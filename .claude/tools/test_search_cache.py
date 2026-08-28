#!/usr/bin/env python3
"""Tests for search_cache scoring and pruning. No network, no ollama.

    python3 test_search_cache.py

Guards the thing most likely to rot silently: the numpy fast path in
_score_rows must rank identically to the pure-Python fallback.
"""
from __future__ import annotations

import math
import os
import random
import sqlite3
import sys
import threading
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search_cache as sc  # noqa: E402

DIM = 768
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        FAILURES.append(name)


def make_db(n: int = 400, seed: int = 7, model: str | None = None) -> tuple[str, list]:
    """Temp cache DB of n pages with random unit-ish vectors."""
    rng = random.Random(seed)
    path = os.path.join(tempfile.mkdtemp(prefix="sc-test-"), "cache.db")
    cache = sc.SearchCache(db_path=path)
    conn = cache._connect()
    now = time.time()
    label = sc.embed_space() if model is None else model
    vectors = []
    for i in range(n):
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        vectors.append(vec)
        conn.execute(
            "INSERT INTO pages (url, domain, title, content, query, embedding, "
            "authority, fetched_at, content_hash, model) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"https://e{i}.test/p", f"e{i}.test", f"title {i}", f"content body {i}",
             "q", sc._embed_to_blob(vec), rng.choice([0.1, 0.5, 0.9, 1.0]),
             now - rng.uniform(0, 160 * 3600), f"h{i}", label),
        )
    conn.commit()
    return path, vectors


def pin_query(vec):
    """Force the query embedding, so tests never touch the network."""
    sc.embed_one = lambda _text, _task=None, _v=vec: _v


def test_numpy_matches_python() -> None:
    path, vectors = make_db()
    cache = sc.SearchCache(db_path=path)
    conn = cache._connect()
    rows = conn.execute(
        "SELECT url, embedding, authority, fetched_at FROM pages"
    ).fetchall()
    now = time.time()

    # A query vector close to one stored row, so some rows clear the threshold.
    query = [v + 0.15 for v in vectors[3]]

    for min_sim in (0.0, 0.5, 0.75, 0.99):
        fast = sc._score_rows_numpy(rows, query, now, 168, min_sim)
        slow = sc._score_rows_python(rows, query, now, 168, min_sim)
        check(f"numpy path returns a result (min_sim={min_sim})", fast is not None)
        if fast is None:
            continue
        fast.sort(key=lambda s: (-s[1], s[0]))
        slow.sort(key=lambda s: (-s[1], s[0]))
        check(f"same row count (min_sim={min_sim})", len(fast) == len(slow),
              f"{len(fast)} vs {len(slow)}")
        if len(fast) != len(slow):
            continue
        worst = max((max(abs(a[1] - b[1]), abs(a[2] - b[2])) for a, b in zip(fast, slow)),
                    default=0.0)
        same_urls = all(a[0] == b[0] for a, b in zip(fast, slow))
        check(f"same order (min_sim={min_sim})", same_urls)
        check(f"scores agree to 1e-5 (min_sim={min_sim})", worst < 1e-5, f"max delta {worst:.2e}")


def test_ragged_blob_falls_back() -> None:
    """A row stored at another width must not crash or corrupt the ranking."""
    path, vectors = make_db(n=20)
    cache = sc.SearchCache(db_path=path)
    conn = cache._connect()
    conn.execute("UPDATE pages SET embedding = ? WHERE url = ?",
                 (sc._embed_to_blob([0.5] * 64), "https://e0.test/p"))
    conn.commit()
    rows = conn.execute("SELECT url, embedding, authority, fetched_at FROM pages").fetchall()
    now = time.time()
    check("ragged blob -> numpy declines",
          sc._score_rows_numpy(rows, vectors[1], now, 168, 0.0) is None)
    # _score_rows must then give exactly what the pure-Python path gives.
    check("ragged blob -> _score_rows falls back to Python",
          sc._score_rows(rows, vectors[1], now, 168, 0.0)
          == sc._score_rows_python(rows, vectors[1], now, 168, 0.0))


def test_lookup_returns_content_in_score_order() -> None:
    path, vectors = make_db()
    cache = sc.SearchCache(db_path=path)
    pin_query(vectors[5])
    hits = cache.lookup("ignored", min_similarity=0.0, top_k=5)
    check("lookup returns top_k", len(hits) == 5, str(len(hits)))
    check("lookup fills content", all(h.content for h in hits))
    check("lookup fills title/domain", all(h.title and h.domain for h in hits))
    check("lookup is score-ordered",
          all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)))
    check("best hit is the exact-match row", hits[0].url == "https://e5.test/p", hits[0].url)
    check("similarity of exact match is 1.0", math.isclose(hits[0].similarity, 1.0, abs_tol=1e-5),
          str(hits[0].similarity))


def test_min_similarity_is_enforced() -> None:
    path, vectors = make_db()
    cache = sc.SearchCache(db_path=path)
    pin_query(vectors[5])
    hits = cache.lookup("ignored", min_similarity=0.99)
    check("threshold keeps only the exact match", len(hits) == 1, str(len(hits)))
    check("no hit below threshold", all(h.similarity >= 0.99 for h in hits))


def test_threshold_is_calibrated_per_space() -> None:
    """A cut-off tuned for one model silently halves recall on another."""
    check("nomic keeps its tuned 0.75",
          sc.MIN_SIMILARITY_BY_SPACE["ollama:nomic-embed-text"] == 0.75)
    check("the active space has a calibrated cut-off",
          sc.embed_space() in sc.MIN_SIMILARITY_BY_SPACE, sc.embed_space())
    check("default resolves to the active space's value",
          sc.min_similarity_default() == sc.MIN_SIMILARITY_BY_SPACE[sc.embed_space()])

    os.environ["WEB_RESEARCH_MIN_SIMILARITY"] = "0.42"
    try:
        check("env overrides the default", sc.min_similarity_default() == 0.42)
        os.environ["WEB_RESEARCH_MIN_SIMILARITY"] = "not-a-number"
        check("junk override falls back rather than crashing",
              sc.min_similarity_default() == sc.MIN_SIMILARITY_BY_SPACE[sc.embed_space()])
    finally:
        del os.environ["WEB_RESEARCH_MIN_SIMILARITY"]

    # lookup with no explicit threshold must use it
    path, vectors = make_db(n=30)
    cache = sc.SearchCache(db_path=path)
    pin_query(vectors[7])
    hits = cache.lookup("ignored")
    check("lookup applies the calibrated default",
          all(h.similarity >= sc.min_similarity_default() for h in hits))


def test_project_resolves_once_under_threads() -> None:
    """The backfill calls this from a thread pool.

    Publishing a half-resolved value let sibling threads treat it as final and
    return None, which failed 175 of 200 rows on the first real run.
    """
    saved_run, saved_project = sc.subprocess.run, sc._project
    saved_env = {k: os.environ.pop(k, None) for k in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT")}

    calls = []

    class Result:
        stdout = "resolved-project\n"

    def slow_run(*a, **kw):
        calls.append(1)
        time.sleep(0.2)  # widen the window a real subprocess would open
        return Result()

    sc.subprocess.run = slow_run
    sc._project = None
    try:
        out: list = []
        threads = [threading.Thread(target=lambda: out.append(sc._gcp_project()))
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sc.subprocess.run = saved_run
        sc._project = saved_project
        for k, v in saved_env.items():
            if v is not None:
                os.environ[k] = v

    check("every thread gets the project", out == ["resolved-project"] * 8,
          f"{sorted(set(map(str, out)))}")
    check("gcloud is consulted once, not per thread", len(calls) == 1, str(len(calls)))


def test_prune() -> None:
    path, _ = make_db(n=200)
    cache = sc.SearchCache(db_path=path)
    conn = cache._connect()
    old = time.time() - 400 * 3600
    conn.execute("UPDATE pages SET fetched_at = ? WHERE url LIKE 'https://e1_.test/p'", (old,))
    conn.commit()
    stale = conn.execute("SELECT COUNT(*) FROM pages WHERE fetched_at <= ?",
                         (time.time() - 168 * 3600,)).fetchone()[0]

    r = cache.prune(max_age_hours=168)
    check("prune deletes exactly the stale rows", r["rows_deleted"] == stale,
          f"{r['rows_deleted']} vs {stale}")
    check("prune reports consistent totals", r["rows_after"] == r["rows_before"] - r["rows_deleted"])
    left = conn.execute("SELECT COUNT(*) FROM pages WHERE fetched_at <= ?",
                        (time.time() - 168 * 3600,)).fetchone()[0]
    check("no stale rows survive", left == 0, str(left))
    check("second prune is a no-op", cache.prune(max_age_hours=168)["rows_deleted"] == 0)


def test_foreign_model_rows_are_never_ranked() -> None:
    """The whole point of the model column.

    A row embedded by another model has a real, comparable-looking vector of
    the same width. Ranking it would return confident nonsense.
    """
    path, vectors = make_db(n=50, model="some-other-model@768")
    cache = sc.SearchCache(db_path=path)
    pin_query(vectors[5])
    check("foreign-model rows are invisible", cache.lookup("ignored", min_similarity=0.0) == [])

    conn = cache._connect()
    conn.execute("UPDATE pages SET model = ? WHERE url = ?",
                 (sc.embed_space(), "https://e5.test/p"))
    conn.commit()
    hits = cache.lookup("ignored", min_similarity=0.0)
    check("only the relabelled row is ranked", len(hits) == 1, str(len(hits)))
    check("and it is the right one", hits and hits[0].url == "https://e5.test/p")


def test_migration_labels_legacy_rows() -> None:
    """A cache written before the model column existed holds Ollama vectors."""
    path = os.path.join(tempfile.mkdtemp(prefix="sc-legacy-"), "cache.db")
    legacy = sqlite3.connect(path)
    legacy.execute("""
        CREATE TABLE pages (
            url TEXT PRIMARY KEY, domain TEXT NOT NULL, title TEXT,
            content TEXT NOT NULL, query TEXT NOT NULL, embedding BLOB,
            authority REAL DEFAULT 0.5, fetched_at REAL NOT NULL,
            content_hash TEXT NOT NULL)
    """)
    legacy.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)",
                   ("https://old.test/p", "old.test", "t", "c" * 200, "q",
                    sc._embed_to_blob([0.1] * DIM), 0.5, time.time(), "h"))
    legacy.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?,?)",
                   ("https://noemb.test/p", "noemb.test", "t", "c" * 200, "q",
                    None, 0.5, time.time(), "h"))
    legacy.commit()
    legacy.close()

    conn = sc.SearchCache(db_path=path)._connect()
    cols = {r[1] for r in conn.execute("PRAGMA table_info(pages)")}
    check("migration adds the model column", "model" in cols)
    labels = dict(conn.execute("SELECT url, model FROM pages"))
    check("embedded legacy row is labelled ollama",
          labels["https://old.test/p"] == "ollama:nomic-embed-text",
          str(labels.get("https://old.test/p")))
    check("row without a vector stays unlabelled", labels["https://noemb.test/p"] is None)
    check("migration is idempotent",
          sc.SearchCache(db_path=path)._connect().execute(
              "SELECT COUNT(*) FROM pages").fetchone()[0] == 2)


def test_store_many_records_the_space() -> None:
    path = os.path.join(tempfile.mkdtemp(prefix="sc-store-"), "cache.db")
    cache = sc.SearchCache(db_path=path)
    saved = sc.embed_batch
    sc.embed_batch = lambda texts, task=None: [[0.3] * DIM for _ in texts]
    try:
        n = cache.store_many([
            (f"https://s{i}.test/p", "s.test", f"t{i}", "body " * 40, "q") for i in range(3)
        ])
    finally:
        sc.embed_batch = saved
    check("store_many reports what it wrote", n == 3, str(n))
    conn = cache._connect()
    models = [r[0] for r in conn.execute("SELECT model FROM pages")]
    check("every stored row carries the active space",
          models == [sc.embed_space()] * 3, str(set(models)))
    check("store_many skips thin content",
          cache.store_many([("https://tiny.test/p", "t.test", "t", "short", "q")]) == 0)


def test_failed_embed_is_stored_unsearchable() -> None:
    """A failed embed must not lose the page, and must not fake a vector."""
    path = os.path.join(tempfile.mkdtemp(prefix="sc-fail-"), "cache.db")
    cache = sc.SearchCache(db_path=path)
    saved = sc.embed_batch
    sc.embed_batch = lambda texts, task=None: [None for _ in texts]
    try:
        cache.store_many([("https://f.test/p", "f.test", "t", "body " * 40, "q")])
    finally:
        sc.embed_batch = saved
    row = cache._connect().execute(
        "SELECT embedding, model, LENGTH(content) FROM pages").fetchone()
    check("content is kept", row[2] > 100)
    check("no vector is invented", row[0] is None)
    check("no space is claimed", row[1] is None)


if __name__ == "__main__":
    print(f"numpy available: {sc._np is not None}   space: {sc.embed_space()}\n")
    for fn in (test_numpy_matches_python, test_ragged_blob_falls_back,
               test_lookup_returns_content_in_score_order,
               test_min_similarity_is_enforced,
               test_threshold_is_calibrated_per_space,
               test_project_resolves_once_under_threads, test_prune,
               test_foreign_model_rows_are_never_ranked,
               test_migration_labels_legacy_rows,
               test_store_many_records_the_space,
               test_failed_embed_is_stored_unsearchable):
        print(f"-- {fn.__name__}")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all passed")
