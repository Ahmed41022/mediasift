"""Match runs, their groups, and the decisions about the files in them."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from mediasift.core.errors import LibraryError
from mediasift.core.types import Choice, DecisionSource, MatchGroup, MatchKind, MatchMember
from mediasift.db._util import PAGE_SIZE, chunked, clean_text, dump_json, load_json, marks, now
from mediasift.db.connection import Database

# Carry the user's decisions from older runs of a kind to the groups of the new run.
#
# Keep and undecided follow the file anywhere. Remove is followed only when the new group
# still holds another file of the old group that was not to be removed, that is, the reason
# the file could go. Otherwise the user would be taken to agree to losing it in favour of
# files never looked at. On disagreement between old groups, keeping wins.
_CARRY = """
INSERT INTO decisions (group_id, file_id, choice, source, reason)
SELECT nm.group_id, nm.file_id, od.choice, 'user', od.reason
FROM match_runs r
JOIN match_groups og ON og.run_id = r.id
JOIN decisions od ON od.group_id = og.id AND od.source = 'user'
JOIN match_members nm ON nm.file_id = od.file_id
JOIN match_groups ng ON ng.id = nm.group_id AND ng.run_id = :run
WHERE r.kind = :kind AND r.id <> :run
  AND (od.choice <> 'remove' OR EXISTS (
    SELECT 1 FROM match_members om
    JOIN match_members partner ON partner.group_id = nm.group_id AND partner.file_id = om.file_id
    LEFT JOIN decisions pd ON pd.group_id = om.group_id AND pd.file_id = om.file_id
    WHERE om.group_id = od.group_id AND om.file_id <> od.file_id
      AND coalesce(pd.choice, 'undecided') <> 'remove'))
ORDER BY CASE od.choice WHEN 'keep' THEN 0 WHEN 'undecided' THEN 1 ELSE 2 END, od.group_id DESC
ON CONFLICT DO NOTHING
"""

_GROUP_COLUMNS = "g.id, g.kind, g.confidence, g.reclaimable_bytes"


@dataclass(frozen=True, slots=True)
class MatchRun:
    """One finished run of a matcher."""

    id: int
    kind: MatchKind
    matcher: str
    matcher_version: int
    params: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True, slots=True)
class Decision:
    """What to do with one file of a group."""

    group_id: int
    file_id: int
    choice: Choice
    source: DecisionSource
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GroupCursor:
    """Where a page of groups ended. Pass it to get the next page."""

    reclaimable_bytes: int
    group_id: int

    @classmethod
    def of(cls, group: MatchGroup) -> GroupCursor:
        """Cursor just after this group."""
        return cls(group.reclaimable_bytes, group.id)


class MatchesRepo:
    """Rows of the match_runs, match_groups, match_members and decisions tables."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # --- runs ---

    def replace_run(
        self,
        kind: MatchKind,
        groups: Iterable[MatchGroup],
        *,
        matcher: str,
        matcher_version: int,
        params: Mapping[str, Any] | None = None,
    ) -> MatchRun:
        """Store the groups of a run and drop older runs of this kind, all or nothing.

        ``groups`` is consumed inside the transaction. User decisions are carried over (see
        _CARRY); automatic ones are not, they are cheap to make again.
        """
        with self._db.transaction():
            cursor = self._db.execute(
                "INSERT INTO match_runs (kind, matcher, matcher_version, params, state,"
                " created_at) VALUES (?, ?, ?, ?, 'running', ?)",
                (kind.value, matcher, matcher_version, dump_json(params or {}), now()),
            )
            run_id = cursor.lastrowid or 0
            for chunk in chunked(groups):
                self._add_groups(run_id, kind, chunk)
            self._db.execute(_CARRY, {"run": run_id, "kind": kind.value})
            self._db.execute(
                "DELETE FROM match_runs WHERE kind = ? AND id <> ?", (kind.value, run_id)
            )
            self._db.execute("UPDATE match_runs SET state = 'done' WHERE id = ?", (run_id,))
            run = self.latest_run(kind)
        assert run is not None
        return run

    def _add_groups(self, run_id: int, kind: MatchKind, groups: list[MatchGroup]) -> None:
        members: list[tuple[Any, ...]] = []
        for group in groups:
            if group.kind is not kind:
                raise ValueError(f"a {group.kind} group in a {kind} run")
            if len(group.members) < 2:
                raise ValueError("a group needs at least two members")
            cursor = self._db.execute(
                "INSERT INTO match_groups (run_id, kind, confidence, reclaimable_bytes,"
                " member_count) VALUES (?, ?, ?, ?, ?)",
                (run_id, kind.value, group.confidence, group.reclaimable_bytes, len(group.members)),
            )
            group_id = cursor.lastrowid
            members.extend(
                (group_id, member.file_id, member.distance, dump_json(member.evidence))
                for member in group.members
            )
        self._db.executemany(
            "INSERT INTO match_members (group_id, file_id, distance, evidence) VALUES (?, ?, ?, ?)",
            members,
        )

    def latest_run(self, kind: MatchKind) -> MatchRun | None:
        """The stored run of a kind, if any."""
        row = self._db.execute(
            "SELECT id, matcher, matcher_version, params, created_at FROM match_runs"
            " WHERE kind = ? AND state = 'done' ORDER BY id DESC LIMIT 1",
            (kind.value,),
        ).fetchone()
        if row is None:
            return None
        run_id, matcher, version, params, created = row
        return MatchRun(run_id, kind, matcher, version, load_json(params), created)

    # --- groups ---

    def count_groups(self, kind: MatchKind | None = None) -> tuple[int, int]:
        """Number of groups and the bytes they could free."""
        where, params = _kind_filter(kind)
        row = self._db.execute(
            f"SELECT count(*), coalesce(sum(g.reclaimable_bytes), 0) FROM match_groups g{where}",
            params,
        ).fetchone()
        return row[0], row[1]

    def list_groups(
        self,
        kind: MatchKind | None = None,
        *,
        limit: int = 100,
        after: GroupCursor | None = None,
    ) -> list[MatchGroup]:
        """A page of groups, the ones that free most first."""
        where, params = _kind_filter(kind)
        if after is not None:
            where += " AND" if where else " WHERE"
            # Redundant upper bound first, which lets the index skip to the page.
            where += (
                " g.reclaimable_bytes <= :bytes AND (g.reclaimable_bytes < :bytes OR g.id > :group)"
            )
            params.update(bytes=after.reclaimable_bytes, group=after.group_id)
        rows = self._db.execute(
            f"SELECT {_GROUP_COLUMNS} FROM match_groups g{where}"
            " ORDER BY g.reclaimable_bytes DESC, g.id LIMIT :limit",
            {**params, "limit": limit},
        ).fetchall()
        return self._with_members(rows)

    def iter_groups(
        self, kind: MatchKind | None = None, *, page_size: int = PAGE_SIZE
    ) -> Iterator[MatchGroup]:
        """All groups in the order of ``list_groups``, a page at a time."""
        after = None
        while page := self.list_groups(kind, limit=page_size, after=after):
            yield from page
            after = GroupCursor.of(page[-1])

    def get_group(self, group_id: int) -> MatchGroup:
        """A group by id. Ids are never reused, so an outdated one is an error."""
        rows = self._db.execute(
            f"SELECT {_GROUP_COLUMNS} FROM match_groups g WHERE g.id = ?", (group_id,)
        ).fetchall()
        if not rows:
            raise LibraryError(f"there is no group with number {group_id}")
        return self._with_members(rows)[0]

    def groups_of_file(self, file_id: int) -> list[int]:
        """Ids of the groups a file is in."""
        rows = self._db.execute(
            "SELECT group_id FROM match_members WHERE file_id = ? ORDER BY group_id", (file_id,)
        ).fetchall()
        return [row[0] for row in rows]

    def _with_members(self, rows: list[tuple[Any, ...]]) -> list[MatchGroup]:
        members: dict[int, list[MatchMember]] = {row[0]: [] for row in rows}
        for ids in chunked(members):
            found = self._db.execute(
                "SELECT group_id, file_id, distance, evidence FROM match_members"
                f" WHERE group_id IN ({marks(len(ids))}) ORDER BY group_id, distance, file_id",
                ids,
            )
            for group_id, file_id, distance, evidence in found.fetchall():
                members[group_id].append(MatchMember(file_id, distance, load_json(evidence)))
        return [
            MatchGroup(
                kind=MatchKind(kind),
                members=tuple(members[group_id]),
                confidence=confidence,
                reclaimable_bytes=reclaimable,
                id=group_id,
            )
            for group_id, kind, confidence, reclaimable in rows
        ]

    # --- decisions ---

    def set_decisions(self, decisions: Iterable[Decision]) -> int:
        """Store decisions. An automatic one never replaces the user's.

        Returns the number stored. Whether a group keeps a file is checked when planning.
        """
        rows = (
            (d.group_id, d.file_id, d.choice.value, d.source.value, clean_text(d.reason))
            for d in decisions
        )
        with self._db.transaction():
            try:
                return self._db.executemany(
                    "INSERT INTO decisions (group_id, file_id, choice, source, reason)"
                    " VALUES (?, ?, ?, ?, ?)"
                    " ON CONFLICT (group_id, file_id) DO UPDATE SET choice = excluded.choice,"
                    " source = excluded.source, reason = excluded.reason"
                    " WHERE excluded.source = 'user' OR decisions.source = 'auto'",
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                raise LibraryError("a decision is about a file that is not in the group") from exc

    def decisions(self, group_id: int) -> dict[int, Decision]:
        """Decisions of a group by file id. Files without one are left out."""
        return self.decisions_many([group_id]).get(group_id, {})

    def decisions_many(self, group_ids: Iterable[int]) -> dict[int, dict[int, Decision]]:
        """Decisions by group id and file id."""
        found: dict[int, dict[int, Decision]] = {}
        for ids in chunked(group_ids):
            rows = self._db.execute(
                "SELECT group_id, file_id, choice, source, reason FROM decisions"
                f" WHERE group_id IN ({marks(len(ids))})",
                ids,
            )
            for group_id, file_id, choice, source, reason in rows.fetchall():
                found.setdefault(group_id, {})[file_id] = Decision(
                    group_id, file_id, Choice(choice), DecisionSource(source), reason
                )
        return found

    def clear_decisions(
        self,
        *,
        kind: MatchKind | None = None,
        group_id: int | None = None,
        file_id: int | None = None,
        source: DecisionSource | None = None,
    ) -> int:
        """Forget the decisions that match every given condition. Return how many."""
        conditions = {
            "group_id IN (SELECT id FROM match_groups WHERE kind = ?)": kind,
            "group_id = ?": group_id,
            "file_id = ?": file_id,
            "source = ?": source,
        }
        used = {sql: value for sql, value in conditions.items() if value is not None}
        where = f" WHERE {' AND '.join(used)}" if used else ""
        with self._db.transaction():
            cursor = self._db.execute(f"DELETE FROM decisions{where}", list(used.values()))
            return cursor.rowcount


def _kind_filter(kind: MatchKind | None) -> tuple[str, dict[str, Any]]:
    return ("", {}) if kind is None else (" WHERE g.kind = :kind", {"kind": kind.value})
