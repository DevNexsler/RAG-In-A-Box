from doc_id_store import DocIDStore


def test_registry_schema_includes_dedupe_columns(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    cols = {
        row[1]: row[2]
        for row in store._conn.execute("PRAGMA table_info(doc_registry)").fetchall()
    }
    assert "size_bytes" in cols
    assert "content_hash" in cols
    assert "hash_algo" in cols
    assert "dedupe_status" in cols
    assert "canonical_doc_id" in cols
    assert "archive_path" in cols
    assert "duplicate_reason" in cols
    assert "duplicate_of_doc_id" in cols
    assert "first_seen_at" in cols
    assert "last_seen_at" in cols
    indexes = {
        row[1]: row[2]
        for row in store._conn.execute("PRAGMA index_list(doc_registry)").fetchall()
    }
    assert "idx_doc_registry_size_hash" in indexes


def test_find_exact_duplicate_returns_first_seen(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=123,
        content_hash=b"\x01" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    hit = store.find_canonical_by_exact_hash(123, b"\x01" * 32, "blake3")
    assert hit["doc_id"] == "documents::00001"


def test_find_exact_duplicate_ignores_duplicate_only_rows(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=123,
        content_hash=b"\x09" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::old",
    )

    assert store.find_canonical_by_exact_hash(123, b"\x09" * 32, "blake3") is None


def test_find_exact_duplicate_filters_by_hash_algo(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=124,
        content_hash=b"\x15" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=124,
        content_hash=b"\x15" * 32,
        hash_algo="sha256",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )

    blake_hit = store.find_canonical_by_exact_hash(124, b"\x15" * 32, "blake3")
    sha_hit = store.find_canonical_by_exact_hash(124, b"\x15" * 32, "sha256")
    missing_hit = store.find_canonical_by_exact_hash(124, b"\x15" * 32, "md5")

    assert blake_hit["doc_id"] == "documents::00001"
    assert sha_hit["doc_id"] == "documents::00002"
    assert missing_hit is None


def test_first_seen_uses_registration_time_not_dedupe_update_time(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")

    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=222,
        content_hash=b"\x05" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::old",
    )
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=222,
        content_hash=b"\x05" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::old",
    )

    hit = store.claim_canonical_by_exact_hash(
        "documents::00002",
        222,
        b"\x05" * 32,
        hash_algo="blake3",
    )
    assert hit["doc_id"] == "documents::00001"


def test_claim_canonical_by_exact_hash_is_first_seen_wins(tmp_path):
    import threading

    db_path = tmp_path / "doc_registry.db"
    store1 = DocIDStore(db_path)
    store2 = DocIDStore(db_path)
    store1.register("documents::00001", "a.pdf", source_name="documents")
    store2.register("documents::00002", "b.pdf", source_name="documents")
    store1.update_dedupe_identity(
        "documents::00001",
        size_bytes=123,
        content_hash=b"\x02" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::old",
    )
    store2.update_dedupe_identity(
        "documents::00002",
        size_bytes=123,
        content_hash=b"\x02" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::old",
    )

    barrier = threading.Barrier(3)
    results = []

    def claim(store, doc_id):
        barrier.wait()
        results.append(
            store.claim_canonical_by_exact_hash(
                doc_id,
                123,
                b"\x02" * 32,
                hash_algo="blake3",
            )
        )

    t1 = threading.Thread(target=claim, args=(store1, "documents::00001"))
    t2 = threading.Thread(target=claim, args=(store2, "documents::00002"))
    t1.start()
    t2.start()
    barrier.wait()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert {row["doc_id"] for row in results} == {"documents::00001"}
    assert all(row["dedupe_status"] == "canonical" for row in results)

    rows = store1._conn.execute(
        """
        SELECT doc_id, dedupe_status, canonical_doc_id, duplicate_of_doc_id
        FROM doc_registry
        WHERE size_bytes = ? AND content_hash = ?
        ORDER BY doc_id
        """,
        (123, b"\x02" * 32),
    ).fetchall()
    assert len(rows) == 2
    assert sum(row[1] == "canonical" for row in rows) == 1
    assert sum(row[1] == "duplicate" for row in rows) == 1
    assert rows[0][1] == "canonical"
    assert rows[0][2] is None
    assert rows[0][3] is None
    assert rows[1][2] == "documents::00001"
    assert rows[1][3] == "documents::00001"


def test_canonical_promotion_clears_duplicate_only_fields(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=333,
        content_hash=b"\x06" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::old",
        archive_path="archive/a.pdf",
        duplicate_reason="legacy_copy",
    )

    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=333,
        content_hash=b"\x06" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id="documents::old",
    )

    row = store._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00001'
        """
    ).fetchone()
    assert row == ("canonical", None, None, None, None)


def test_update_dedupe_identity_canonical_race_enforces_first_seen(tmp_path):
    import threading

    db_path = tmp_path / "doc_registry.db"
    store1 = DocIDStore(db_path)
    store2 = DocIDStore(db_path)
    store1.register("documents::00001", "a.pdf", source_name="documents")
    store2.register("documents::00002", "b.pdf", source_name="documents")
    store1.update_dedupe_identity(
        "documents::00001",
        size_bytes=444,
        content_hash=b"\x08" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
    )
    store2.update_dedupe_identity(
        "documents::00002",
        size_bytes=444,
        content_hash=b"\x08" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
    )

    barrier = threading.Barrier(3)
    results = []

    def promote(store, doc_id):
        barrier.wait()
        try:
            store.update_dedupe_identity(
                doc_id,
                size_bytes=444,
                content_hash=b"\x08" * 32,
                hash_algo="blake3",
                dedupe_status="canonical",
                canonical_doc_id="documents::bad",
            )
        except Exception as exc:
            results.append(("error", type(exc).__name__))
        else:
            results.append(("ok", doc_id))

    t1 = threading.Thread(target=promote, args=(store1, "documents::00001"))
    t2 = threading.Thread(target=promote, args=(store2, "documents::00002"))
    t1.start()
    t2.start()
    barrier.wait()
    t1.join()
    t2.join()

    assert len(results) == 2
    assert results.count(("ok", "documents::00001")) == 1
    assert any(kind == "error" for kind, _ in results)

    row1 = store1._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00001'
        """
    ).fetchone()
    row2 = store1._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00002'
        """
    ).fetchone()
    assert row1 == ("canonical", None, None, None, None)
    assert row2 == ("duplicate", "documents::00001", None, None, "documents::00001")


def test_update_dedupe_identity_canonical_includes_untagged_older_caller_in_election(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "older.pdf", source_name="documents")
    store.register("documents::00002", "newer.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=445,
        content_hash=b"\x0b" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
    )

    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=445,
        content_hash=b"\x0b" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id="documents::ignored",
        archive_path="archive/ignored.pdf",
        duplicate_reason="ignored",
    )

    row1 = store._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00001'
        """
    ).fetchone()
    row2 = store._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00002'
        """
    ).fetchone()
    assert row1 == ("canonical", None, None, None, None)
    assert row2 == ("duplicate", "documents::00001", None, None, "documents::00001")


def test_update_dedupe_identity_canonical_repoints_sibling_duplicates(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "older.pdf", source_name="documents")
    store.register("documents::00002", "newer.pdf", source_name="documents")
    store.register("documents::00003", "third.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=446,
        content_hash=b"\x0d" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
        archive_path="archive/b.pdf",
        duplicate_reason="seeded",
    )
    store.update_dedupe_identity(
        "documents::00003",
        size_bytes=446,
        content_hash=b"\x0d" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
        archive_path="archive/c.pdf",
        duplicate_reason="seeded",
    )

    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=446,
        content_hash=b"\x0d" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id="documents::ignored",
    )

    refs = store.duplicate_refs_for_canonical("documents::00001")
    assert {ref["doc_id"] for ref in refs} == {"documents::00002", "documents::00003"}
    assert all(ref["canonical_doc_id"] == "documents::00001" for ref in refs)
    assert all(ref["duplicate_of_doc_id"] == "documents::00001" for ref in refs)


def test_claim_canonical_preserves_existing_duplicate_provenance(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.register("documents::00003", "c.pdf", source_name="documents")

    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=789,
        content_hash=b"\x07" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=789,
        content_hash=b"\x07" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::00001",
        archive_path="archive/old.pdf",
        duplicate_reason="legacy_copy",
    )
    store.update_dedupe_identity(
        "documents::00003",
        size_bytes=789,
        content_hash=b"\x07" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::00001",
    )

    store.claim_canonical_by_exact_hash(
        "documents::00003",
        789,
        b"\x07" * 32,
        hash_algo="blake3",
        archive_path="archive/new.pdf",
        duplicate_reason="exact_match",
    )

    row2 = store._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00002'
        """
    ).fetchone()
    assert row2 == ("duplicate", "documents::00001", "archive/old.pdf", "legacy_copy", "documents::00001")

    row3 = store._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00003'
        """
    ).fetchone()
    assert row3 == ("duplicate", "documents::00001", "archive/new.pdf", "exact_match", "documents::00001")


def test_claim_canonical_clears_duplicate_fields_on_promotion(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=790,
        content_hash=b"\x0a" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::old",
        archive_path="archive/a.pdf",
        duplicate_reason="legacy_copy",
    )

    store.claim_canonical_by_exact_hash(
        "documents::00001",
        790,
        b"\x0a" * 32,
        hash_algo="blake3",
        archive_path="archive/new.pdf",
        duplicate_reason="exact_match",
    )

    row = store._conn.execute(
        """
        SELECT dedupe_status, canonical_doc_id, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'documents::00001'
        """
    ).fetchone()
    assert row == ("canonical", None, None, None, None)


def test_migration_adds_dedupe_columns_and_index(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE doc_registry (
            doc_id TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            created REAL NOT NULL
        );
        CREATE TABLE counter (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            value INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO counter VALUES (1, 0);
        INSERT INTO doc_registry VALUES ('old1', 'old.md', 0.0);
    """)
    conn.commit()
    conn.close()

    store = DocIDStore(db_path)
    cols = {
        row[1]: row[2]
        for row in store._conn.execute("PRAGMA table_info(doc_registry)").fetchall()
    }
    assert "size_bytes" in cols
    assert "content_hash" in cols
    assert "hash_algo" in cols
    assert "dedupe_status" in cols
    assert "canonical_doc_id" in cols
    assert "archive_path" in cols
    assert "duplicate_reason" in cols
    assert "duplicate_of_doc_id" in cols
    assert "first_seen_at" in cols
    assert "last_seen_at" in cols
    indexes = {
        row[1]: row[2]
        for row in store._conn.execute("PRAGMA index_list(doc_registry)").fetchall()
    }
    assert "idx_doc_registry_size_hash" in indexes
    assert store._conn.execute("PRAGMA index_list(doc_registry)").fetchall()

    store.update_dedupe_identity(
        "old1",
        size_bytes=11,
        content_hash=b"\x04" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    row = store._conn.execute(
        """
        SELECT created, first_seen_at, last_seen_at, archive_path, duplicate_reason, duplicate_of_doc_id
        FROM doc_registry
        WHERE doc_id = 'old1'
        """
    ).fetchone()
    assert row[0] == 0.0
    assert row[1] == 0.0
    assert row[2] is not None
    assert row[3] is None
    assert row[4] is None
    assert row[5] is None


def test_mark_duplicate_records_duplicate_ref_metadata(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.claim_canonical_by_exact_hash(
        "documents::00001",
        456,
        b"\x03" * 32,
        hash_algo="blake3",
    )
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=456,
        content_hash=b"\x03" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
    )

    store.mark_duplicate(
        "documents::00002",
        "documents::00001",
        archive_path="archive/b.pdf",
        duplicate_reason="exact_match",
    )

    refs = store.duplicate_refs_for_canonical("documents::00001")
    assert len(refs) == 1
    ref = refs[0]
    assert ref["doc_id"] == "documents::00002"
    assert ref["dedupe_status"] == "duplicate"
    assert ref["canonical_doc_id"] == "documents::00001"
    assert ref["archive_path"] == "archive/b.pdf"
    assert ref["duplicate_reason"] == "exact_match"
    assert ref["duplicate_of_doc_id"] == "documents::00001"
    assert ref["first_seen_at"] is not None
    assert ref["last_seen_at"] is not None


def test_mark_duplicate_missing_doc_raises_keyerror(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    try:
        store.mark_duplicate("missing", "documents::00001")
    except KeyError as exc:
        assert exc.args[0] == "missing"
    else:
        raise AssertionError("expected KeyError")


def test_mark_duplicate_rejects_self_reference(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    try:
        store.mark_duplicate("documents::00001", "documents::00001")
    except ValueError as exc:
        assert "self" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_mark_duplicate_missing_canonical_target_raises_keyerror(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=777,
        content_hash=b"\x20" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
    )
    try:
        store.mark_duplicate("documents::00001", "documents::99999")
    except KeyError as exc:
        assert exc.args[0] == "documents::99999"
    else:
        raise AssertionError("expected KeyError")


def test_mark_duplicate_rejects_noncanonical_target(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=501,
        content_hash=b"\x0c" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::other",
    )
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=501,
        content_hash=b"\x0c" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
    )
    try:
        store.mark_duplicate("documents::00001", "documents::00002")
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_mark_duplicate_rejects_canonical_source_row(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.register("documents::00003", "c.pdf", source_name="documents")
    store.claim_canonical_by_exact_hash(
        "documents::00001",
        601,
        b"\x0e" * 32,
        hash_algo="blake3",
    )
    store.claim_canonical_by_exact_hash(
        "documents::00002",
        601,
        b"\x0e" * 32,
        hash_algo="blake3",
    )
    store.claim_canonical_by_exact_hash(
        "documents::00003",
        999,
        b"\x1f" * 32,
        hash_algo="blake3",
    )

    try:
        store.mark_duplicate("documents::00001", "documents::00001")
    except ValueError as exc:
        assert "self" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        store.mark_duplicate("documents::00001", "documents::00003")
    except ValueError as exc:
        assert "canonical" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_mark_duplicate_rejects_cross_cluster_hash_mismatch(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=602,
        content_hash=b"\x0f" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=603,
        content_hash=b"\x10" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::seed",
    )

    try:
        store.mark_duplicate("documents::00002", "documents::00001")
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_update_dedupe_identity_rejects_canonical_hash_move_that_strands_old_cohort(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=604,
        content_hash=b"\x11" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=604,
        content_hash=b"\x11" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::00001",
        archive_path="archive/b.pdf",
        duplicate_reason="same_hash",
    )

    try:
        store.update_dedupe_identity(
            "documents::00001",
            size_bytes=605,
            content_hash=b"\x12" * 32,
            hash_algo="blake3",
            dedupe_status="canonical",
            canonical_doc_id=None,
        )
    except ValueError as exc:
        assert "cohort" in str(exc) or "canonical" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    rows = store._conn.execute(
        """
        SELECT doc_id, dedupe_status, canonical_doc_id
        FROM doc_registry
        ORDER BY doc_id
        """
    ).fetchall()
    assert rows == [
        ("documents::00001", "canonical", None),
        ("documents::00002", "duplicate", "documents::00001"),
    ]


def test_claim_canonical_by_exact_hash_rejects_hash_move_that_strands_old_cohort(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=606,
        content_hash=b"\x13" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=606,
        content_hash=b"\x13" * 32,
        hash_algo="blake3",
        dedupe_status="duplicate",
        canonical_doc_id="documents::00001",
        archive_path="archive/b.pdf",
        duplicate_reason="same_hash",
    )

    try:
        store.claim_canonical_by_exact_hash(
            "documents::00001",
            607,
            b"\x14" * 32,
            hash_algo="blake3",
        )
    except ValueError as exc:
        assert "cohort" in str(exc) or "canonical" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    rows = store._conn.execute(
        """
        SELECT doc_id, dedupe_status, canonical_doc_id
        FROM doc_registry
        ORDER BY doc_id
        """
    ).fetchall()
    assert rows == [
        ("documents::00001", "canonical", None),
        ("documents::00002", "duplicate", "documents::00001"),
    ]


def test_claim_canonical_by_exact_hash_keeps_mixed_hash_algos_separate(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")

    winner1 = store.claim_canonical_by_exact_hash(
        "documents::00001",
        701,
        b"\x21" * 32,
        hash_algo="blake3",
    )
    winner2 = store.claim_canonical_by_exact_hash(
        "documents::00002",
        701,
        b"\x21" * 32,
        hash_algo="sha256",
    )

    assert winner1["doc_id"] == "documents::00001"
    assert winner2["doc_id"] == "documents::00002"
    rows = store._conn.execute(
        """
        SELECT doc_id, dedupe_status, canonical_doc_id, hash_algo
        FROM doc_registry
        ORDER BY doc_id
        """
    ).fetchall()
    assert rows == [
        ("documents::00001", "canonical", None, "blake3"),
        ("documents::00002", "canonical", None, "sha256"),
    ]


def test_update_dedupe_identity_canonical_does_not_mutate_other_hash_algo_row(tmp_path):
    store = DocIDStore(tmp_path / "doc_registry.db")
    store.register("documents::00001", "a.pdf", source_name="documents")
    store.register("documents::00002", "b.pdf", source_name="documents")
    store.update_dedupe_identity(
        "documents::00002",
        size_bytes=702,
        content_hash=b"\x22" * 32,
        hash_algo="sha256",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )

    store.update_dedupe_identity(
        "documents::00001",
        size_bytes=702,
        content_hash=b"\x22" * 32,
        hash_algo="blake3",
        dedupe_status="canonical",
        canonical_doc_id=None,
    )

    rows = store._conn.execute(
        """
        SELECT doc_id, dedupe_status, canonical_doc_id, hash_algo
        FROM doc_registry
        ORDER BY doc_id
        """
    ).fetchall()
    assert rows == [
        ("documents::00001", "canonical", None, "blake3"),
        ("documents::00002", "canonical", None, "sha256"),
    ]
