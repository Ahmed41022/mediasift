-- The library index (docs/PLAN.md 1.3).
--
-- Paths are blobs holding the exact bytes; *_display columns are for people only.
-- AUTOINCREMENT ids are never reused, so an id kept in the operation log, or shown to the
-- user earlier, can never come to mean another file or group.
-- dev and inode are unsigned 64-bit numbers stored as signed ones; 0 means unknown.

CREATE TABLE roots (
  id             INTEGER PRIMARY KEY AUTOINCREMENT,
  path           BLOB    NOT NULL UNIQUE CHECK (length(path) > 0),
  display        TEXT    NOT NULL,
  path_key       BLOB    NOT NULL UNIQUE,
  volume_id      TEXT    NOT NULL DEFAULT '',
  case_sensitive INTEGER NOT NULL CHECK (case_sensitive IN (0, 1)),
  read_only      INTEGER NOT NULL DEFAULT 0 CHECK (read_only IN (0, 1)),
  prefer_rank    INTEGER NOT NULL DEFAULT 0,
  enabled        INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1))
) STRICT;

CREATE TABLE scans (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  state       TEXT NOT NULL
              CHECK (state IN ('running', 'done', 'cancelled', 'failed', 'interrupted')),
  options     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(options)),
  counters    TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(counters)),
  CHECK ((state = 'running') = (finished_at IS NULL))
) STRICT;

CREATE TABLE files (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  root_id       INTEGER NOT NULL REFERENCES roots (id) ON DELETE CASCADE,
  rel_path      BLOB    NOT NULL CHECK (length(rel_path) > 0),
  rel_display   TEXT    NOT NULL,
  rel_key       BLOB    NOT NULL,
  ext           TEXT    NOT NULL,
  media_type    TEXT    NOT NULL CHECK (media_type IN ('image', 'video', 'audio')),
  size          INTEGER NOT NULL CHECK (size >= 0),
  mtime_ns      INTEGER NOT NULL,
  dev           INTEGER NOT NULL DEFAULT 0,
  inode         INTEGER NOT NULL DEFAULT 0,
  nlink         INTEGER NOT NULL DEFAULT 1 CHECK (nlink >= 1),
  status        TEXT    NOT NULL CHECK (status IN ('present', 'missing', 'error', 'placeholder')),
  first_scan_id INTEGER REFERENCES scans (id) ON DELETE SET NULL,
  last_scan_id  INTEGER REFERENCES scans (id) ON DELETE SET NULL,
  UNIQUE (root_id, rel_path)
) STRICT;

CREATE INDEX files_key ON files (root_id, rel_key);
CREATE INDEX files_size ON files (size, mtime_ns);
CREATE INDEX files_identity ON files (dev, inode) WHERE inode <> 0;

CREATE TABLE features (
  file_id      INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
  feature      TEXT    NOT NULL CHECK (feature <> ''),
  version      INTEGER NOT NULL,
  value        BLOB,
  status       TEXT    NOT NULL CHECK (status IN ('ok', 'error')),
  error        TEXT,
  src_size     INTEGER NOT NULL,
  src_mtime_ns INTEGER NOT NULL,
  PRIMARY KEY (file_id, feature),
  CHECK ((status = 'ok') = (value IS NOT NULL)),
  CHECK ((status = 'error') = (error IS NOT NULL))
) STRICT, WITHOUT ROWID;

CREATE INDEX features_value ON features (feature, value, file_id);

CREATE TABLE metadata (
  file_id        INTEGER PRIMARY KEY REFERENCES files (id) ON DELETE CASCADE,
  reader         TEXT    NOT NULL,
  reader_version INTEGER NOT NULL,
  width          INTEGER,
  height         INTEGER,
  orientation    INTEGER,
  duration_ms    INTEGER,
  taken_at       TEXT,
  taken_source   TEXT,
  camera_make    TEXT,
  camera_model   TEXT,
  gps_lat        REAL,
  gps_lon        REAL,
  codec          TEXT,
  bitrate        INTEGER,
  artist         TEXT,
  album          TEXT,
  title          TEXT,
  track          TEXT,
  extra          TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(extra)),
  src_size       INTEGER NOT NULL,
  src_mtime_ns   INTEGER NOT NULL
) STRICT;

CREATE INDEX metadata_taken ON metadata (taken_at) WHERE taken_at IS NOT NULL;

CREATE TABLE scan_issues (
  id      INTEGER PRIMARY KEY,
  scan_id INTEGER NOT NULL REFERENCES scans (id) ON DELETE CASCADE,
  root_id INTEGER REFERENCES roots (id) ON DELETE SET NULL,
  path    BLOB    NOT NULL,
  kind    TEXT    NOT NULL CHECK (kind <> ''),
  message TEXT    NOT NULL DEFAULT ''
) STRICT;

CREATE INDEX scan_issues_scan ON scan_issues (scan_id, id);
CREATE INDEX scan_issues_root ON scan_issues (root_id);

CREATE TABLE match_runs (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  kind            TEXT    NOT NULL CHECK (kind <> ''),
  matcher         TEXT    NOT NULL,
  matcher_version INTEGER NOT NULL,
  params          TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(params)),
  state           TEXT    NOT NULL CHECK (state IN ('running', 'done')),
  created_at      TEXT    NOT NULL
) STRICT;

CREATE INDEX match_runs_kind ON match_runs (kind);

CREATE TABLE match_groups (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id            INTEGER NOT NULL REFERENCES match_runs (id) ON DELETE CASCADE,
  kind              TEXT    NOT NULL CHECK (kind <> ''),
  confidence        REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  reclaimable_bytes INTEGER NOT NULL CHECK (reclaimable_bytes >= 0),
  member_count      INTEGER NOT NULL CHECK (member_count >= 2)
) STRICT;

CREATE INDEX match_groups_run ON match_groups (run_id);
CREATE INDEX match_groups_kind_size ON match_groups (kind, reclaimable_bytes DESC, id);
CREATE INDEX match_groups_size ON match_groups (reclaimable_bytes DESC, id);

CREATE TABLE match_members (
  group_id INTEGER NOT NULL REFERENCES match_groups (id) ON DELETE CASCADE,
  file_id  INTEGER NOT NULL REFERENCES files (id) ON DELETE CASCADE,
  distance REAL    NOT NULL DEFAULT 0 CHECK (distance >= 0),
  evidence TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(evidence)),
  PRIMARY KEY (group_id, file_id)
) STRICT, WITHOUT ROWID;

CREATE INDEX match_members_file ON match_members (file_id, group_id);

CREATE TABLE decisions (
  group_id INTEGER NOT NULL,
  file_id  INTEGER NOT NULL,
  choice   TEXT    NOT NULL CHECK (choice IN ('keep', 'remove', 'undecided')),
  source   TEXT    NOT NULL CHECK (source IN ('auto', 'user')),
  reason   TEXT    NOT NULL DEFAULT '',
  PRIMARY KEY (group_id, file_id),
  FOREIGN KEY (group_id, file_id) REFERENCES match_members (group_id, file_id) ON DELETE CASCADE
) STRICT, WITHOUT ROWID;

CREATE INDEX decisions_file ON decisions (file_id, source);

-- The operation log. It refers to files by id without a foreign key on purpose: whatever
-- happens to the index, the history of what was moved where stays complete.

CREATE TABLE op_sessions (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  kind        TEXT NOT NULL CHECK (kind <> ''),
  state       TEXT NOT NULL CHECK (state <> ''),
  created_at  TEXT NOT NULL,
  finished_at TEXT,
  summary     TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(summary))
) STRICT;

CREATE TABLE operations (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  INTEGER NOT NULL REFERENCES op_sessions (id) ON DELETE RESTRICT,
  seq         INTEGER NOT NULL,
  action      TEXT    NOT NULL CHECK (action <> ''),
  backend     TEXT    NOT NULL CHECK (backend <> ''),
  state       TEXT    NOT NULL CHECK (state <> ''),
  file_id     INTEGER,
  src         BLOB    NOT NULL,
  dst         BLOB,
  size        INTEGER,
  mtime_ns    INTEGER,
  dev         INTEGER,
  inode       INTEGER,
  full_hash   BLOB,
  keeper_path BLOB,
  detail      TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(detail)),
  error       TEXT,
  started_at  TEXT    NOT NULL,
  finished_at TEXT,
  UNIQUE (session_id, seq)
) STRICT;

CREATE INDEX operations_state ON operations (state);
CREATE INDEX operations_file ON operations (file_id);

CREATE TRIGGER op_sessions_keep BEFORE DELETE ON op_sessions
BEGIN
  SELECT RAISE(ABORT, 'the operation log is append-only');
END;

CREATE TRIGGER operations_keep BEFORE DELETE ON operations
BEGIN
  SELECT RAISE(ABORT, 'the operation log is append-only');
END;
