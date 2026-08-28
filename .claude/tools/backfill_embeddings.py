#!/usr/bin/env python3
"""Re-embed cached pages into the active vector space.

    python3 backfill_embeddings.py --dry-run
    python3 backfill_embeddings.py --window 168 --workers 8

Rows carrying another model's vector, or none at all, are invisible to
lookup(). This walks them in batches and rewrites embedding + model.

Resumable: it re-queries the outstanding set each pass, so an interrupted run
loses at most one in-flight batch. Safe to run while web_research is working;
writes are short and the connection waits on locks.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import search_cache as sc  # noqa: E402


def outstanding(conn: sqlite3.Connection, space: str, cutoff: float) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM pages WHERE fetched_at > ? AND (model IS NOT ? OR embedding IS NULL)",
        (cutoff, space),
    ).fetchone()[0]


def fetch_chunk(conn: sqlite3.Connection, space: str, cutoff: float, limit: int):
    return conn.execute(
        "SELECT url, title, content FROM pages "
        "WHERE fetched_at > ? AND (model IS NOT ? OR embedding IS NULL) "
        "ORDER BY fetched_at DESC LIMIT ?",
        (cutoff, space, limit),
    ).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=sc.DB_PATH, help="cache path")
    ap.add_argument("--window", type=int, default=168,
                    help="only rows fetched within HOURS (default 168, what lookup reads)")
    ap.add_argument("--workers", type=int, default=8, help="parallel Vertex calls (default 8)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, then exit")
    args = ap.parse_args()

    space = sc.embed_space()
    cache = sc.SearchCache(db_path=args.db)
    conn = cache._connect()
    cutoff = time.time() - args.window * 3600

    todo = outstanding(conn, space, cutoff)
    total = conn.execute("SELECT COUNT(*) FROM pages WHERE fetched_at > ?", (cutoff,)).fetchone()[0]
    print(f"target space : {space}")
    print(f"window       : {args.window}h  ({total} rows)")
    print(f"to re-embed  : {todo}")
    for model, count in conn.execute(
        "SELECT COALESCE(model,'(none)'), COUNT(*) FROM pages WHERE fetched_at > ? "
        "GROUP BY 1 ORDER BY 2 DESC", (cutoff,)
    ):
        print(f"   {count:>7}  {model}")
    if args.dry_run or not todo:
        return 0
    if args.limit:
        todo = min(todo, args.limit)

    batch = sc.VERTEX_MAX_BATCH
    pool = args.workers * batch
    done = failed = 0
    started = time.time()

    while done + failed < todo:
        rows = fetch_chunk(conn, space, cutoff, min(pool, todo - done - failed))
        if not rows:
            break
        chunks = [rows[i:i + batch] for i in range(0, len(rows), batch)]

        def run(chunk):
            texts = [cache._embed_text(t or "", c or "") for _, t, c in chunk]
            return chunk, sc.embed_batch(texts, sc.TASK_DOCUMENT)

        writes = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool_exec:
            for chunk, embeddings in pool_exec.map(run, chunks):
                for (url, _, _), emb in zip(chunk, embeddings):
                    if emb:
                        writes.append((sc._embed_to_blob(emb), space, url))
                    else:
                        failed += 1

        if writes:
            conn.executemany(
                "UPDATE pages SET embedding = ?, model = ? WHERE url = ?", writes)
            conn.commit()
            done += len(writes)

        if not writes:  # nothing is landing; stop rather than spin
            print(f"\naborting: a full pass of {len(rows)} rows produced no embeddings",
                  file=sys.stderr)
            return 1

        rate = done / max(time.time() - started, 1e-6)
        eta = (todo - done - failed) / rate if rate else 0
        print(f"\r  {done}/{todo} re-embedded  {failed} failed  "
              f"{rate:.0f}/s  eta {eta/60:.1f}m", end="", flush=True)

    print(f"\ndone: {done} re-embedded, {failed} failed, "
          f"{time.time()-started:.0f}s")
    print(f"remaining in window: {outstanding(conn, space, cutoff)}")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())
