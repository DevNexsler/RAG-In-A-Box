-- Staging comm-store fixture schema.
-- Loaded once by postgres:16-alpine via /docker-entrypoint-initdb.d/.
-- Modeled on the Comm-Data-Store messages shape used by config.yaml.example,
-- flattened to a single table for staging determinism.

CREATE TABLE messages (
    id                serial PRIMARY KEY,
    source            text NOT NULL,
    source_message_id text NOT NULL,
    channel_name      text,
    sender            text,
    direction         text,
    body              text,
    sent_at           timestamptz NOT NULL,
    updated_at        timestamptz NOT NULL
);

-- 5 deterministic fixture rows: fixed timestamps, distinct senders/directions,
-- distinctive searchable words in each body.
INSERT INTO messages
    (source, source_message_id, channel_name, sender, direction, body, sent_at, updated_at)
VALUES
    ('quo',   'msg-001', 'ops',     'Alice Nguyen', 'inbound',
     'The quarterly zephyr report is ready for review, see the attached spreadsheet.',
     '2026-06-01T10:00:00Z', '2026-06-01T10:00:00Z'),
    ('quo',   'msg-002', 'ops',     'Bob Ramirez',  'outbound',
     'Thanks Alice — the marmalade budget line still looks off by 3 percent.',
     '2026-06-01T10:01:00Z', '2026-06-01T10:01:00Z'),
    ('email', 'msg-003', 'billing', 'Carol Idowu',  'inbound',
     'Invoice 4417 for the obsidian widgets was paid on Friday.',
     '2026-06-01T10:02:00Z', '2026-06-01T10:02:00Z'),
    ('email', 'msg-004', 'billing', 'Dan Park',     'outbound',
     'Confirming receipt of invoice 4417; the ledger now reconciles cleanly.',
     '2026-06-01T10:03:00Z', '2026-06-01T10:03:00Z'),
    ('sms',   'msg-005', 'field',   'Erin Walsh',   'inbound',
     'Crew reached the periwinkle substation, inspection starts at noon.',
     '2026-06-01T10:04:00Z', '2026-06-01T10:04:00Z');
