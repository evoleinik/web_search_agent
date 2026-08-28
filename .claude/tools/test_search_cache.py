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
import sys
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


def make_db(n: int = 400, seed: int = 7) -> tuple[str, list]:
    """Temp cache DB of n pages with random unit-ish vectors."""
    rng = random.Random(seed)
    path = os.path.join(tempfile.mkdtemp(prefix="sc-test-"), "cache.db")
    cache = sc.SearchCache(db_path=path)
    conn = cache._connect()
    now = time.time()
    vectors = []
    for i in range(n):
        vec = [rng.gauss(0, 1) for _ in range(DIM)]
        vectors.append(vec)
        conn.execute(
            "INSERT INTO pages (url, domain, title, content, query, embedding, "
            "authority, fetched_at, content_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (f"https://e{i}.test/p", f"e{i}.test", f"title {i}", f"content body {i}",
             "q", sc._embed_to_blob(vec), rng.choice([0.1, 0.5, 0.9, 1.0]),
             now - rng.uniform(0, 160 * 3600), f"h{i}"),
        )
    conn.commit()
    return path, vectors


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
    sc._embed_ollama = lambda _text, _v=vectors[5]: _v  # pin the query vector
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
    sc._embed_ollama = lambda _text, _v=vectors[5]: _v
    hits = cache.lookup("ignored", min_similarity=0.99)
    check("threshold keeps only the exact match", len(hits) == 1, str(len(hits)))
    check("no hit below threshold", all(h.similarity >= 0.99 for h in hits))


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


if __name__ == "__main__":
    print(f"numpy available: {sc._np is not None}\n")
    for fn in (test_numpy_matches_python, test_ragged_blob_falls_back,
               test_lookup_returns_content_in_score_order,
               test_min_similarity_is_enforced, test_prune):
        print(f"-- {fn.__name__}")
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all passed")
