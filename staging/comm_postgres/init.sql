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
    subject           text,
    body              text,
    sent_at           timestamptz NOT NULL,
    updated_at        timestamptz NOT NULL
);

CREATE TABLE participants (
    source          text NOT NULL,
    participant_key text NOT NULL,
    display_name    text,
    PRIMARY KEY (source, participant_key)
);

INSERT INTO participants (source, participant_key, display_name)
VALUES
    ('zoho_cliq', '720844989', 'Dan Park'),
    ('zoho_cliq', '918334727', 'Nigel Pine');

-- 7 deterministic fixture rows: fixed timestamps, distinct senders/directions,
-- distinctive searchable words, one subject-only email, and one tracking-heavy
-- HTML email that exercises normalization through consumer-visible Lance rows.
INSERT INTO messages
    (source, source_message_id, channel_name, sender, direction, subject, body, sent_at, updated_at)
VALUES
    ('quo',   'msg-001', 'ops',     'Alice Nguyen', 'inbound', NULL,
     'The quarterly zephyr report is ready for review, see the attached spreadsheet.',
     '2026-06-01T10:00:00Z', '2026-06-01T10:00:00Z'),
    ('quo',   'msg-002', 'ops',     'Bob Ramirez',  'outbound', NULL,
     'Thanks Alice — the marmalade budget line still looks off by 3 percent.',
     '2026-06-01T10:01:00Z', '2026-06-01T10:01:00Z'),
    ('email', 'msg-003', 'billing', 'Carol Idowu',  'inbound', 'Obsidian widget invoice',
     'Invoice 4417 for the obsidian widgets was paid on Friday.',
     '2026-06-01T10:02:00Z', '2026-06-01T10:02:00Z'),
    ('email', 'msg-004', 'billing', 'Dan Park',     'outbound', 'Re: Obsidian widget invoice',
     'Confirming receipt of invoice 4417; the ledger now reconciles cleanly.',
     '2026-06-01T10:03:00Z', '2026-06-01T10:03:00Z'),
    ('sms',   'msg-005', 'field',   'Erin Walsh',   'inbound', NULL,
     'Crew reached the periwinkle substation, inspection starts at noon.',
     '2026-06-01T10:04:00Z', '2026-06-01T10:04:00Z'),
    ('email', 'msg-006', 'legal',   'Joycelyn Reed', 'inbound',
     'Cobalt courthouse filing received', NULL,
     '2026-06-01T10:05:00Z', '2026-06-01T10:05:00Z');

INSERT INTO messages
    (source, source_message_id, channel_name, sender, direction, subject, body, sent_at, updated_at)
SELECT
    'zoho_mail', 'tracking-heavy', 'delivery', 'Delivery Robot', 'inbound',
    'Appliance delivery confirmation',
    '<html><head><style>.hidden{display:none}</style></head><body>'
    || '<div hidden>' || repeat('&zwnj;&#847;', 500) || '</div>'
    || '<h1>Your delivery is scheduled</h1>'
    || '<p>Order STAGE-246565 includes a washer and dryer.</p>'
    || '<p>Delivery: September 4, 8 AM-12 PM, 12 Oak Street.</p>'
    || string_agg(
         '<a href="https://tracker.example/redirect?upn=u001' || sequence
         || '&amp;target=' || repeat('x', 900) || '">Track delivery</a>',
         '' ORDER BY sequence
       )
    || '<img src="https://images.example/pixel.gif?recipient=123">'
    || '<footer><a href="https://social.example/icon.png?campaign=abc">Follow us</a>'
    || '<p>Privacy settings | Unsubscribe | Terms of use</p></footer>'
    || '</body></html>',
    '2026-06-01T10:06:00Z', '2026-06-01T10:06:00Z'
FROM generate_series(1, 18) AS redirects(sequence);

CREATE TABLE "Buildings" (
    id               integer PRIMARY KEY,
    "Nick_Name"      text,
    created_at        timestamptz NOT NULL,
    updated_at        timestamptz
);

CREATE TABLE "Building Units" (
    id               integer PRIMARY KEY,
    "Buildings_id"   integer REFERENCES "Buildings" (id),
    "Unit"           text,
    "Bed"            numeric,
    "Bath"           numeric,
    "Description"    text,
    "Status"         text,
    created_at        timestamptz NOT NULL,
    updated_at        timestamptz
);

INSERT INTO "Buildings" (id, "Nick_Name", created_at, updated_at)
VALUES
    (1, '  South   Main Apartments Unit  ', '2026-06-01T09:00:00Z', '2026-06-01T09:00:00Z'),
    (2, '125 S 13TH STREET LLC',          '2026-06-01T09:00:00Z', '2026-06-01T09:00:00Z');

INSERT INTO "Building Units"
    (id, "Buildings_id", "Unit", "Bed", "Bath", "Description", "Status", created_at, updated_at)
VALUES
    (104, 1, ' Unit   5 ', 2, 1, 'South Main staging unit.', 'Occupied',
     '2026-06-01T09:00:00Z', '2026-06-01T09:00:00Z'),
    (105, 2, 'B',          1, 1, '13th Street staging unit.', 'Vacant',
     '2026-06-01T09:00:00Z', '2026-06-01T09:00:00Z');
