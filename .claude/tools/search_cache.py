"""
Search Memory — SQLite cache with vector embeddings for web research.

Stores fetched pages with nomic-embed-text embeddings (via Ollama).
On cache lookup, finds semantically similar cached pages ranked by
similarity * 0.6 + authority * 0.3 + freshness * 0.1.

Usage (from web_research.py):
    from search_cache import SearchCache
    cache = SearchCache()
    hits = cache.lookup(query, max_age_hours=168, top_k=10)
    cache.store(url, domain, title, content, query)
    cache.print_stats()
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Sequence, Tuple

try:
    import numpy as _np
except ImportError:  # scoring falls back to pure Python
    _np = None

# =============================================================================
# DOMAIN AUTHORITY MAP
# =============================================================================

# Scores 0.0–1.0. Unknown domains default to 0.5.
DOMAIN_AUTHORITY: dict[str, float] = {
    # 1.0 — Official docs & standards
    "docs.python.org": 1.0,
    "docs.rs": 1.0,
    "doc.rust-lang.org": 1.0,
    "go.dev": 1.0,
    "pkg.go.dev": 1.0,
    "developer.mozilla.org": 1.0,
    "developer.apple.com": 1.0,
    "developer.android.com": 1.0,
    "learn.microsoft.com": 1.0,
    "cloud.google.com": 1.0,
    "docs.aws.amazon.com": 1.0,
    "kubernetes.io": 1.0,
    "nodejs.org": 1.0,
    "react.dev": 1.0,
    "nextjs.org": 1.0,
    "vuejs.org": 1.0,
    "angular.dev": 1.0,
    "svelte.dev": 1.0,
    "w3.org": 1.0,
    "rfc-editor.org": 1.0,
    "ietf.org": 1.0,
    "tc39.es": 1.0,
    "ecma-international.org": 1.0,
    "spec.graphql.org": 1.0,
    "www.postgresql.org": 1.0,
    "sqlite.org": 1.0,
    "dev.mysql.com": 1.0,
    "redis.io": 1.0,
    "docs.docker.com": 1.0,

    # 0.9 — Research & top-tier platforms
    "arxiv.org": 0.9,
    "github.com": 0.9,
    "stackoverflow.com": 0.9,
    "huggingface.co": 0.9,
    "arstechnica.com": 0.9,
    "lwn.net": 0.9,
    "acm.org": 0.9,
    "dl.acm.org": 0.9,
    "ieee.org": 0.9,
    "ieeexplore.ieee.org": 0.9,
    "nature.com": 0.9,
    "science.org": 0.9,
    "openai.com": 0.9,
    "anthropic.com": 0.9,
    "deepmind.google": 0.9,
    "ai.meta.com": 0.9,
    "research.google": 0.9,
    "blog.google": 0.9,
    "engineering.fb.com": 0.9,
    "netflixtechblog.com": 0.9,
    "aws.amazon.com": 0.9,
    "azure.microsoft.com": 0.9,
    "engineering.atspotify.com": 0.9,
    "uber.com/blog": 0.9,
    "engineering.linkedin.com": 0.9,
    "discord.com/blog": 0.9,

    # 0.8 — High quality tech blogs & tools
    "news.ycombinator.com": 0.8,
    "lobste.rs": 0.8,
    "martinfowler.com": 0.8,
    "joelonsoftware.com": 0.8,
    "paulgraham.com": 0.8,
    "simonwillison.net": 0.8,
    "jvns.ca": 0.8,
    "danluu.com": 0.8,
    "brooker.co.za": 0.8,
    "brandur.org": 0.8,
    "fasterthanli.me": 0.8,
    "web.dev": 0.8,
    "css-tricks.com": 0.8,
    "smashingmagazine.com": 0.8,
    "infoq.com": 0.8,
    "thenewstack.io": 0.8,
    "theregister.com": 0.8,
    "pypi.org": 0.8,
    "npmjs.com": 0.8,
    "www.npmjs.com": 0.8,
    "crates.io": 0.8,
    "vercel.com": 0.8,
    "cloudflare.com": 0.8,
    "blog.cloudflare.com": 0.8,
    "fastly.com": 0.8,
    "fly.io": 0.8,
    "stripe.com": 0.8,
    "docs.stripe.com": 0.8,
    "twilio.com": 0.8,
    "auth0.com": 0.8,
    "hashicorp.com": 0.8,
    "terraform.io": 0.8,
    "docker.com": 0.8,
    "grafana.com": 0.8,
    "prometheus.io": 0.8,
    "elastic.co": 0.8,
    "kafka.apache.org": 0.8,
    "spark.apache.org": 0.8,
    "airflow.apache.org": 0.8,
    "pytorch.org": 0.8,
    "tensorflow.org": 0.8,
    "jupyter.org": 0.8,
    "pandas.pydata.org": 0.8,
    "numpy.org": 0.8,
    "scipy.org": 0.8,

    # 0.7 — Good quality
    "dev.to": 0.7,
    "medium.com": 0.7,
    "substack.com": 0.7,
    "realpython.com": 0.7,
    "learnxinyminutes.com": 0.7,
    "freecodecamp.org": 0.7,
    "www.freecodecamp.org": 0.7,
    "digitalocean.com": 0.7,
    "linode.com": 0.7,
    "baeldung.com": 0.7,
    "geeksforgeeks.org": 0.7,
    "www.geeksforgeeks.org": 0.7,
    "techcrunch.com": 0.7,
    "wired.com": 0.7,
    "www.wired.com": 0.7,
    "theverge.com": 0.7,
    "www.theverge.com": 0.7,
    "bloomberg.com": 0.7,
    "reuters.com": 0.7,
    "reddit.com": 0.7,
    "www.reddit.com": 0.7,
    "old.reddit.com": 0.7,
    "wikipedia.org": 0.7,
    "en.wikipedia.org": 0.7,
    "testdriven.io": 0.7,
    "codecademy.com": 0.7,
    "exercism.org": 0.7,
    "brilliant.org": 0.7,

    # 0.6 — Decent / forums
    "superuser.com": 0.6,
    "askubuntu.com": 0.6,
    "serverfault.com": 0.6,
    "unix.stackexchange.com": 0.6,
    "apple.stackexchange.com": 0.6,
    "dba.stackexchange.com": 0.6,
    "security.stackexchange.com": 0.6,
    "softwareengineering.stackexchange.com": 0.6,
    "slashdot.org": 0.6,
    "linuxquestions.org": 0.6,
    "forums.docker.com": 0.6,
    "discuss.hashicorp.com": 0.6,
    "community.cloudflare.com": 0.6,
    "forum.nginx.org": 0.6,

    # 0.3 — Low quality / content farms
    "w3schools.com": 0.3,
    "www.w3schools.com": 0.3,
    "tutorialspoint.com": 0.3,
    "www.tutorialspoint.com": 0.3,
    "javatpoint.com": 0.3,
    "www.javatpoint.com": 0.3,
    "programiz.com": 0.3,
    "www.programiz.com": 0.3,
    "guru99.com": 0.3,
    "www.guru99.com": 0.3,
    "makeuseof.com": 0.3,
    "www.makeuseof.com": 0.3,
    "about.com": 0.3,
    "ehow.com": 0.3,
    "www.ehow.com": 0.3,
    "wikihow.com": 0.3,
    "www.wikihow.com": 0.3,
    "educba.com": 0.3,
    "www.educba.com": 0.3,
    "simplilearn.com": 0.3,
    "www.simplilearn.com": 0.3,

    # 0.1 — Known spam / clickbait
    "answersq.com": 0.1,
    "quillbot.com": 0.1,
    "copyleaks.com": 0.1,
}

DEFAULT_AUTHORITY = 0.5
DB_PATH = os.path.join(os.path.expanduser("~"), ".web-research", "cache.db")


def domain_authority(domain: str) -> float:
    """Look up authority score for a domain. Tries exact match, then strips www."""
    if domain in DOMAIN_AUTHORITY:
        return DOMAIN_AUTHORITY[domain]
    # Strip www. prefix
    if domain.startswith("www."):
        bare = domain[4:]
        if bare in DOMAIN_AUTHORITY:
            return DOMAIN_AUTHORITY[bare]
    # Try parent domain (e.g., blog.cloudflare.com -> cloudflare.com)
    parts = domain.split(".")
    if len(parts) > 2:
        parent = ".".join(parts[-2:])
        if parent in DOMAIN_AUTHORITY:
            return DOMAIN_AUTHORITY[parent]
    return DEFAULT_AUTHORITY


# =============================================================================
# EMBEDDINGS VIA OLLAMA
# =============================================================================

def _embed_ollama(text: str) -> Optional[List[float]]:
    """Get embedding from Ollama nomic-embed-text via HTTP API. Returns 768-dim float list or None."""
    import urllib.request
    try:
        payload = json.dumps({"model": "nomic-embed-text", "input": text[:8000]}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        emb = data.get("embeddings", [[]])[0]
        return emb if emb else None
    except Exception:
        return None


def _embed_to_blob(embedding: List[float]) -> bytes:
    """Pack float list to bytes (little-endian float32)."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _blob_to_embed(blob: bytes) -> List[float]:
    """Unpack bytes to float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"<{n}f", blob))


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# =============================================================================
# SCORING
# =============================================================================

# Rows are (url, embedding_blob, authority, fetched_at). Scored as
#     similarity * 0.6 + authority * 0.3 + freshness * 0.1
# and returned as (url, score, similarity) for rows clearing min_similarity.


def _score_rows_python(
    rows: Sequence[tuple],
    query_emb: List[float],
    now: float,
    max_age_hours: int,
    min_similarity: float,
) -> List[Tuple[str, float, float]]:
    out: List[Tuple[str, float, float]] = []
    for url, emb_blob, authority, fetched_at in rows:
        sim = _cosine_similarity(query_emb, _blob_to_embed(emb_blob))
        if sim < min_similarity:
            continue
        freshness = max(0.0, 1.0 - ((now - fetched_at) / 3600.0) / max_age_hours)
        out.append((url, sim * 0.6 + authority * 0.3 + freshness * 0.1, sim))
    return out


def _score_rows_numpy(
    rows: Sequence[tuple],
    query_emb: List[float],
    now: float,
    max_age_hours: int,
    min_similarity: float,
) -> Optional[List[Tuple[str, float, float]]]:
    """One matmul over the whole window. Returns None if the blobs are ragged."""
    dim = len(query_emb)
    mat = _np.frombuffer(b"".join(r[1] for r in rows), dtype="<f4")
    if mat.size != len(rows) * dim:  # a row stored at a different width
        return None
    mat = mat.reshape(len(rows), dim)

    qv = _np.asarray(query_emb, dtype="<f4")
    q_norm = _np.linalg.norm(qv)
    if q_norm == 0:
        return []

    norms = _np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1.0
    sims = (mat @ (qv / q_norm)) / norms

    keep = _np.flatnonzero(sims >= min_similarity)
    if keep.size == 0:
        return []

    kept_sims = sims[keep]
    authority = _np.fromiter((rows[i][2] for i in keep), dtype="<f8", count=keep.size)
    fetched_at = _np.fromiter((rows[i][3] for i in keep), dtype="<f8", count=keep.size)
    freshness = _np.maximum(0.0, 1.0 - ((now - fetched_at) / 3600.0) / max_age_hours)
    scores = kept_sims * 0.6 + authority * 0.3 + freshness * 0.1

    return [
        (rows[i][0], float(scores[k]), float(kept_sims[k]))
        for k, i in enumerate(keep)
    ]


def _score_rows(
    rows: Sequence[tuple],
    query_emb: List[float],
    now: float,
    max_age_hours: int,
    min_similarity: float,
) -> List[Tuple[str, float, float]]:
    if _np is not None:
        scored = _score_rows_numpy(rows, query_emb, now, max_age_hours, min_similarity)
        if scored is not None:
            return scored
    return _score_rows_python(rows, query_emb, now, max_age_hours, min_similarity)


# =============================================================================
# CACHE
# =============================================================================

@dataclass
class CacheHit:
    """A cached page result."""
    url: str
    domain: str
    title: str
    content: str
    query: str
    authority: float
    fetched_at: float
    similarity: float
    score: float  # combined ranking score


class SearchCache:
    """SQLite-backed search cache with vector similarity lookup."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    url TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    query TEXT NOT NULL,
                    embedding BLOB,
                    authority REAL DEFAULT 0.5,
                    fetched_at REAL NOT NULL,
                    content_hash TEXT NOT NULL
                )
            """)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON pages(domain)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_fetched ON pages(fetched_at)")
            self._conn.commit()
        return self._conn

    def lookup(
        self,
        query: str,
        max_age_hours: int = 168,
        top_k: int = 10,
        min_similarity: float = 0.75,
    ) -> List[CacheHit]:
        """Find cached pages similar to query.

        Returns top_k results ranked by:
            similarity * 0.6 + authority * 0.3 + freshness * 0.1

        The ranking scan reads embeddings only. Page content is fetched in a
        second query, for the winners alone. Scanning content made every lookup
        read hundreds of MB it then threw away.
        """
        query_emb = _embed_ollama(query)
        if query_emb is None:
            return []

        conn = self._connect()
        cutoff = time.time() - (max_age_hours * 3600)
        rows = conn.execute(
            "SELECT url, embedding, authority, fetched_at "
            "FROM pages WHERE fetched_at > ? AND embedding IS NOT NULL",
            (cutoff,),
        ).fetchall()
        if not rows:
            return []

        scored = _score_rows(rows, query_emb, time.time(), max_age_hours, min_similarity)
        if not scored:
            return []
        scored.sort(key=lambda s: s[1], reverse=True)
        scored = scored[:top_k]

        urls = [url for url, _, _ in scored]
        detail = {
            row[0]: row
            for row in conn.execute(
                "SELECT url, domain, title, content, query, authority, fetched_at "
                "FROM pages WHERE url IN (%s)" % ",".join("?" * len(urls)),
                urls,
            )
        }

        hits: List[CacheHit] = []
        for url, score, sim in scored:
            row = detail.get(url)
            if row is None:  # deleted between the two queries
                continue
            _, domain, title, content, orig_query, authority, fetched_at = row
            hits.append(CacheHit(
                url=url, domain=domain, title=title, content=content,
                query=orig_query, authority=authority, fetched_at=fetched_at,
                similarity=sim, score=score,
            ))
        return hits

    def store(
        self,
        url: str,
        domain: str,
        title: str,
        content: str,
        query: str,
    ) -> None:
        """Store a fetched page in cache. Embeds content via Ollama."""
        if not content or len(content) < 100:
            return

        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        authority = domain_authority(domain)

        # Embed title + first 500 chars of content (captures topic efficiently)
        embed_text = f"{title} {content[:500]}"
        embedding = _embed_ollama(embed_text)
        emb_blob = _embed_to_blob(embedding) if embedding else None

        conn = self._connect()
        conn.execute(
            "INSERT OR REPLACE INTO pages "
            "(url, domain, title, content, query, embedding, authority, fetched_at, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (url, domain, title, content, query, emb_blob, authority, time.time(), content_hash),
        )
        conn.commit()

    def stats(self) -> dict:
        """Return cache statistics."""
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        with_emb = conn.execute("SELECT COUNT(*) FROM pages WHERE embedding IS NOT NULL").fetchone()[0]
        total_chars = conn.execute("SELECT COALESCE(SUM(LENGTH(content)), 0) FROM pages").fetchone()[0]
        oldest = conn.execute("SELECT MIN(fetched_at) FROM pages").fetchone()[0]
        newest = conn.execute("SELECT MAX(fetched_at) FROM pages").fetchone()[0]
        domains = conn.execute("SELECT COUNT(DISTINCT domain) FROM pages").fetchone()[0]

        # Age distribution
        now = time.time()
        fresh = conn.execute("SELECT COUNT(*) FROM pages WHERE fetched_at > ?", (now - 86400,)).fetchone()[0]
        week = conn.execute("SELECT COUNT(*) FROM pages WHERE fetched_at > ?", (now - 604800,)).fetchone()[0]

        return {
            "total_pages": total,
            "with_embeddings": with_emb,
            "total_chars": total_chars,
            "unique_domains": domains,
            "oldest": time.strftime("%Y-%m-%d %H:%M", time.localtime(oldest)) if oldest else None,
            "newest": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)) if newest else None,
            "fresh_24h": fresh,
            "fresh_7d": week,
            "db_size_mb": round(os.path.getsize(self.db_path) / 1048576, 1) if os.path.exists(self.db_path) else 0,
        }

    def print_stats(self) -> None:
        """Print cache stats to stderr."""
        s = self.stats()
        print(f"Cache: {s['total_pages']} pages ({s['with_embeddings']} with embeddings), "
              f"{s['unique_domains']} domains, {s['db_size_mb']}MB", file=sys.stderr)
        print(f"  Fresh: {s['fresh_24h']} (<24h), {s['fresh_7d']} (<7d)", file=sys.stderr)
        if s['oldest']:
            print(f"  Range: {s['oldest']} — {s['newest']}", file=sys.stderr)

    def prune(self, max_age_hours: int = 168, vacuum: bool = True) -> dict:
        """Delete rows older than max_age_hours and reclaim the file space.

        lookup() never reads outside its age window, so older rows are dead
        weight. Nothing else ever deleted from this table, so the file only grew.
        """
        conn = self._connect()
        cutoff = time.time() - (max_age_hours * 3600)
        rows_before = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        mb_before = round(os.path.getsize(self.db_path) / 1048576, 1)

        deleted = conn.execute("DELETE FROM pages WHERE fetched_at <= ?", (cutoff,)).rowcount
        conn.commit()

        vacuumed = False
        if vacuum and deleted:
            prev = conn.isolation_level
            conn.isolation_level = None  # VACUUM cannot run inside a transaction
            try:
                conn.execute("VACUUM")
                vacuumed = True
            except sqlite3.OperationalError:
                pass  # another process holds the file; space reclaims next run
            finally:
                conn.isolation_level = prev

        return {
            "rows_before": rows_before,
            "rows_deleted": deleted,
            "rows_after": rows_before - deleted,
            "mb_before": mb_before,
            "mb_after": round(os.path.getsize(self.db_path) / 1048576, 1),
            "vacuumed": vacuumed,
        }

    def print_prune(self, max_age_hours: int = 168) -> None:
        """Prune and report to stderr."""
        r = self.prune(max_age_hours)
        busy = ", vacuum skipped (db busy)" if r["rows_deleted"] and not r["vacuumed"] else ""
        print(f"Pruned {r['rows_deleted']} pages older than {max_age_hours}h "
              f"({r['rows_before']} -> {r['rows_after']} rows, "
              f"{r['mb_before']}MB -> {r['mb_after']}MB){busy}", file=sys.stderr)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
