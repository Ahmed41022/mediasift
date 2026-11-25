from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from mediasift.core.errors import LibraryError
from mediasift.core.types import Choice, DecisionSource, MatchGroup, MatchKind, MatchMember
from mediasift.db.connection import Database
from mediasift.db.files import FilesRepo
from mediasift.db.matches import Decision, GroupCursor, MatchesRepo
from tests.db.conftest import AddFiles

EXACT = MatchKind.EXACT
BURST = MatchKind.BURST
KEEP, REMOVE, UNDECIDED = Choice.KEEP, Choice.REMOVE, Choice.UNDECIDED
USER, AUTO = DecisionSource.USER, DecisionSource.AUTO


@pytest.fixture
def matches(db: Database) -> MatchesRepo:
    return MatchesRepo(db)


@pytest.fixture
def ids(add_files: AddFiles) -> list[int]:
    return [record.id for record in add_files(*(f"{n}.jpg" for n in range(8)))]


def group(*file_ids: int, kind: MatchKind = EXACT, size: int = 10) -> MatchGroup:
    members = tuple(MatchMember(file_id) for file_id in file_ids)
    return MatchGroup(kind, members, reclaimable_bytes=size)


def store(matches: MatchesRepo, *groups: MatchGroup, kind: MatchKind = EXACT) -> list[MatchGroup]:
    """Replace the run and return its groups in the order given."""
    matches.replace_run(kind, groups, matcher="test", matcher_version=1)
    stored = sorted(matches.list_groups(kind), key=lambda g: g.id)
    assert len(stored) == len(groups)
    return stored


def user(group_id: int, file_id: int, choice: Choice, reason: str = "") -> Decision:
    return Decision(group_id, file_id, choice, USER, reason)


def choices(matches: MatchesRepo, group_id: int) -> dict[int, Choice]:
    return {f: d.choice for f, d in matches.decisions(group_id).items()}


# --- runs and groups ---


def test_round_trip(matches: MatchesRepo, ids: list[int]) -> None:
    members = (
        MatchMember(ids[2], 0.0, {"hash": "ab"}),
        MatchMember(ids[0], 1.5, {"why": ["phash", "صورة"]}),
    )
    given = MatchGroup(MatchKind.SIMILAR_IMAGE, members, 0.75, 1234)
    run = matches.replace_run(
        MatchKind.SIMILAR_IMAGE,
        iter([given]),
        matcher="m",
        matcher_version=3,
        params={"threshold": 10},
    )
    assert (run.matcher, run.matcher_version, run.params) == ("m", 3, {"threshold": 10})
    assert matches.latest_run(MatchKind.SIMILAR_IMAGE) == run
    assert matches.latest_run(EXACT) is None
    (stored,) = matches.list_groups()
    assert stored.id > 0
    assert stored == MatchGroup(given.kind, members, 0.75, 1234, stored.id)
    assert matches.get_group(stored.id) == stored
    assert matches.groups_of_file(ids[0]) == [stored.id]


def test_replace_touches_one_kind(matches: MatchesRepo, ids: list[int]) -> None:
    (burst,) = store(matches, group(ids[0], ids[1], kind=BURST), kind=BURST)
    (old,) = store(matches, group(ids[0], ids[1]))
    (new,) = store(matches, group(ids[2], ids[3]))
    assert new.id > old.id
    with pytest.raises(LibraryError, match=f"number {old.id}"):
        matches.get_group(old.id)
    assert matches.get_group(burst.id) == burst
    assert store(matches) == []
    assert matches.latest_run(EXACT) is not None
    assert matches.count_groups(EXACT) == (0, 0)


def test_group_ids_are_never_reused(matches: MatchesRepo, ids: list[int]) -> None:
    (old,) = store(matches, group(ids[0], ids[1]))
    store(matches)
    (new,) = store(matches, group(ids[2], ids[3]))
    assert new.id > old.id


def test_replace_is_all_or_nothing(matches: MatchesRepo, ids: list[int]) -> None:
    (old,) = store(matches, group(ids[0], ids[1]))
    matches.set_decisions([user(old.id, ids[0], REMOVE)])
    run = matches.latest_run(EXACT)

    def failing() -> Iterator[MatchGroup]:
        yield group(ids[2], ids[3])
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        matches.replace_run(EXACT, failing(), matcher="t", matcher_version=1)
    for bad in (group(ids[2]), group(ids[2], ids[3], kind=BURST)):
        with pytest.raises(ValueError, match="group"):
            matches.replace_run(EXACT, [bad], matcher="t", matcher_version=1)
    with pytest.raises(sqlite3.IntegrityError):  # an unknown file
        matches.replace_run(EXACT, [group(ids[2], 10**6)], matcher="t", matcher_version=1)
    assert matches.latest_run(EXACT) == run
    assert matches.list_groups() == [old]
    assert choices(matches, old.id) == {ids[0]: REMOVE}


@pytest.mark.parametrize("page", [1, 2, 3, 100])
def test_pages_by_reclaimable_bytes(matches: MatchesRepo, ids: list[int], page: int) -> None:
    sizes = [5, 900, 5, 0, 900, 70]
    store(matches, *(group(ids[0], ids[1], size=size) for size in sizes))
    store(matches, group(ids[0], ids[1], kind=BURST, size=60), kind=BURST)

    def listed(kind: MatchKind | None) -> list[tuple[int, int]]:
        found: list[MatchGroup] = []
        after = None
        while got := matches.list_groups(kind, limit=page, after=after):
            assert len(got) <= page
            assert all(len(g.members) == 2 for g in got)
            found += got
            after = GroupCursor.of(got[-1])
        assert found == list(matches.iter_groups(kind, page_size=page))
        return [(g.reclaimable_bytes, g.id) for g in found]

    exact = listed(EXACT)
    assert [size for size, _ in exact] == [900, 900, 70, 5, 5, 0]
    assert exact == sorted(exact, key=lambda item: (-item[0], item[1]))
    assert [size for size, _ in listed(None)] == [900, 900, 70, 60, 5, 5, 0]
    assert matches.count_groups(EXACT) == (6, 1880)
    assert matches.count_groups() == (7, 1940)


def test_listing_never_sorts(matches: MatchesRepo, db: Database) -> None:
    statements: list[str] = []
    db._conn.set_trace_callback(statements.append)
    for kind in (EXACT, None):
        matches.list_groups(kind, after=GroupCursor(5, 5))
    db._conn.set_trace_callback(None)
    assert len(statements) == 2
    for sql in statements:
        plan = [row[3] for row in db.execute(f"EXPLAIN QUERY PLAN {sql}")]
        assert len(plan) == 1 and plan[0].startswith("SEARCH"), plan


# --- decisions ---


def test_decisions(matches: MatchesRepo, ids: list[int]) -> None:
    first, second = store(matches, group(*ids[:3]), group(*ids[3:5]))
    auto = Decision(first.id, ids[1], REMOVE, AUTO, "smaller \udcff")
    assert matches.set_decisions([user(first.id, ids[0], KEEP, "أفضل"), auto]) == 2
    stored = matches.decisions(first.id)
    assert stored[ids[0]] == user(first.id, ids[0], KEEP, "أفضل")
    assert stored[ids[1]].reason == "smaller \\udcff"
    assert set(stored) == set(ids[:2])
    assert matches.decisions(second.id) == {}
    assert matches.decisions_many([first.id, second.id]) == {first.id: stored}


def test_user_wins(matches: MatchesRepo, ids: list[int]) -> None:
    (g,) = store(matches, group(*ids[:2]))
    auto = Decision(g.id, ids[0], REMOVE, AUTO)
    assert matches.set_decisions([auto]) == 1
    assert matches.set_decisions([user(g.id, ids[0], KEEP)]) == 1
    assert matches.set_decisions([auto]) == 0
    assert choices(matches, g.id) == {ids[0]: KEEP}
    matches.set_decisions([user(g.id, ids[0], REMOVE)])
    assert choices(matches, g.id) == {ids[0]: REMOVE}


def test_decision_about_a_stranger(matches: MatchesRepo, ids: list[int]) -> None:
    (g,) = store(matches, group(*ids[:2]))
    with pytest.raises(LibraryError, match="not in the group"):
        matches.set_decisions([user(g.id, ids[0], KEEP), user(g.id, ids[5], KEEP)])
    assert matches.decisions(g.id) == {}


def test_clear_decisions(matches: MatchesRepo, ids: list[int]) -> None:
    (g,) = store(matches, group(*ids[:3]))
    (b,) = store(matches, group(*ids[:2], kind=BURST), kind=BURST)
    matches.set_decisions(
        [
            user(g.id, ids[0], KEEP),
            Decision(g.id, ids[1], REMOVE, AUTO),
            Decision(g.id, ids[2], REMOVE, AUTO),
            Decision(b.id, ids[0], REMOVE, AUTO),
        ]
    )
    assert matches.clear_decisions(group_id=g.id, file_id=ids[2]) == 1
    assert matches.clear_decisions(kind=EXACT, source=AUTO) == 1
    assert choices(matches, g.id) == {ids[0]: KEEP}
    assert choices(matches, b.id) == {ids[0]: REMOVE}


# --- carrying decisions over to a new run ---


def test_user_decisions_are_carried_over(matches: MatchesRepo, ids: list[int]) -> None:
    a, b, c, new = ids[:4]
    (old,) = store(matches, group(a, b, c))
    matches.set_decisions(
        [
            user(old.id, a, KEEP, "sharper"),
            user(old.id, b, REMOVE, "blurred"),
            Decision(old.id, c, REMOVE, AUTO, "automatic"),
        ]
    )
    (found,) = store(matches, group(a, b, c, new))
    assert matches.decisions(found.id) == {
        a: user(found.id, a, KEEP, "sharper"),
        b: user(found.id, b, REMOVE, "blurred"),
    }
    (again,) = store(matches, group(a, b))  # and over one more run
    assert choices(matches, again.id) == {a: KEEP, b: REMOVE}


def test_decision_goes_with_its_file_only(matches: MatchesRepo, ids: list[int]) -> None:
    a, b, c, d = ids[:4]
    (old,) = store(matches, group(a, b))
    matches.set_decisions([user(old.id, a, KEEP), user(old.id, b, REMOVE)])
    first, second = store(matches, group(a, c), group(c, d))
    assert choices(matches, first.id) == {a: KEEP}
    assert choices(matches, second.id) == {}


def test_remove_without_its_reason_is_not_carried(matches: MatchesRepo, ids: list[int]) -> None:
    """The user gave up b because a stays. Without a in the group, ask again."""
    a, b, c, d = ids[:4]
    (old,) = store(matches, group(a, b, c))
    matches.set_decisions([user(old.id, a, KEEP), user(old.id, b, REMOVE), user(old.id, c, REMOVE)])
    (found,) = store(matches, group(b, c, d))
    assert choices(matches, found.id) == {}


def test_remove_is_carried_beside_an_undecided_partner(
    matches: MatchesRepo, ids: list[int]
) -> None:
    a, b, c = ids[:3]
    (old,) = store(matches, group(a, b))
    matches.set_decisions([user(old.id, b, REMOVE)])
    (found,) = store(matches, group(a, b, c))
    assert choices(matches, found.id) == {b: REMOVE}


def test_keeping_wins_when_old_groups_disagree(matches: MatchesRepo, ids: list[int]) -> None:
    a, b, c = ids[:3]
    kept, removed = store(matches, group(a, b), group(a, c))
    matches.set_decisions([user(kept.id, a, KEEP), user(removed.id, a, REMOVE)])
    matches.set_decisions([user(kept.id, b, UNDECIDED)])
    (found,) = store(matches, group(a, b, c))
    assert choices(matches, found.id) == {a: KEEP, b: UNDECIDED}


def test_decisions_do_not_cross_kinds(matches: MatchesRepo, ids: list[int]) -> None:
    a, b = ids[:2]
    (exact,) = store(matches, group(a, b))
    matches.set_decisions([user(exact.id, a, KEEP), user(exact.id, b, REMOVE)])
    (burst,) = store(matches, group(a, b, kind=BURST), kind=BURST)
    assert choices(matches, burst.id) == {}
    (again,) = store(matches, group(a, b))
    assert choices(matches, again.id) == {a: KEEP, b: REMOVE}


def test_groups_lose_forgotten_files(matches: MatchesRepo, db: Database, ids: list[int]) -> None:
    (g,) = store(matches, group(*ids[:3]))
    matches.set_decisions([user(g.id, ids[0], REMOVE)])
    db.execute("DELETE FROM files WHERE id = ?", (ids[0],))
    assert [m.file_id for m in matches.get_group(g.id).members] == ids[1:3]
    assert matches.decisions(g.id) == {}
    assert FilesRepo(db).get(ids[1]).id == ids[1]


def test_many_groups(matches: MatchesRepo, ids: list[int]) -> None:
    groups = (group(*ids[:3], size=n) for n in range(3000))
    matches.replace_run(EXACT, groups, matcher="t", matcher_version=1)
    sizes = [g.reclaimable_bytes for g in matches.iter_groups()]
    assert sizes == list(range(2999, -1, -1))
