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
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
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
# EMBEDDINGS
# =============================================================================

# Which model produced a row's vector is recorded per row, and rows from
# another model are never ranked. Two models can share a width (nomic and
# text-embedding-005 are both 768) while occupying unrelated spaces, so a
# cosine across them returns plausible nonsense rather than an error.

EMBED_PROVIDER = os.environ.get("WEB_RESEARCH_EMBED_PROVIDER", "vertex").strip().lower()
OLLAMA_MODEL = os.environ.get("WEB_RESEARCH_OLLAMA_MODEL", "nomic-embed-text")
VERTEX_MODEL = os.environ.get("WEB_RESEARCH_VERTEX_MODEL", "gemini-embedding-001")
VERTEX_DIM = int(os.environ.get("WEB_RESEARCH_VERTEX_DIM", "768"))
VERTEX_LOCATION = os.environ.get("WEB_RESEARCH_VERTEX_LOCATION", "us-central1")

# Vertex accepts up to 25 instances per predict call. Batching is the whole
# throughput story: ~1700ms/item at n=1 against ~140ms/item at n=25.
VERTEX_MAX_BATCH = 25
VERTEX_TIMEOUT = 120
VERTEX_ATTEMPTS = 3

TASK_QUERY = "RETRIEVAL_QUERY"
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"

_TOKEN_CACHE = os.path.join(tempfile.gettempdir(), f"web-research-gcp-token-{os.getuid()}")
_TOKEN_TTL = 2700  # ADC tokens last ~3600s; refresh early
_project: Optional[str] = None
_resolve_lock = threading.Lock()


# Cosine scales are model-specific, so the cut-off is too. Measured over
# matched and mismatched query/document pairs: gemini-embedding-001 puts
# relevant pairs at 0.71-0.83 and irrelevant ones at 0.40-0.62, so 0.70 keeps
# every true hit and admits none of the misses. Nomic's inherited 0.75 applied
# to that scale would throw away half the real hits.
MIN_SIMILARITY_BY_SPACE = {
    "ollama:nomic-embed-text": 0.75,
    "vertex:gemini-embedding-001@768": 0.70,
}
FALLBACK_MIN_SIMILARITY = 0.70


def min_similarity_default() -> float:
    """Similarity cut-off calibrated for the active space."""
    override = os.environ.get("WEB_RESEARCH_MIN_SIMILARITY")
    if override:
        try:
            return float(override)
        except ValueError:
            _log(f"ignoring non-numeric WEB_RESEARCH_MIN_SIMILARITY={override!r}")
    return MIN_SIMILARITY_BY_SPACE.get(embed_space(), FALLBACK_MIN_SIMILARITY)


def embed_space() -> str:
    """Identity of the current vector space. Stored per row, compared per query."""
    if EMBED_PROVIDER == "ollama":
        return f"ollama:{OLLAMA_MODEL}"
    return f"vertex:{VERTEX_MODEL}@{VERTEX_DIM}"


def _log(msg: str) -> None:
    """Embedding failures must be visible. A silent miss looks like a cold cache."""
    if os.environ.get("WEB_RESEARCH_EMBED_QUIET"):
        return
    print(f"[search-cache] {msg}", file=sys.stderr)


def _gcp_project() -> Optional[str]:
    """Resolve the project once per process.

    Held under a lock and assigned only when fully resolved. Publishing the
    empty intermediate value let sibling threads read it as "already resolved"
    and give up, which failed 175 of 200 rows on the first threaded backfill.
    """
    global _project
    if _project is not None:
        return _project or None
    with _resolve_lock:
        if _project is not None:
            return _project or None
        value = (os.environ.get("GOOGLE_CLOUD_PROJECT")
                 or os.environ.get("GCP_PROJECT") or "").strip()
        if not value:
            try:
                value = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True, text=True, timeout=60,
                ).stdout.strip()
            except Exception as exc:
                _log(f"gcloud project lookup failed: {exc}")
                value = ""
        _project = value
    return _project or None


def _access_token() -> Optional[str]:
    """ADC access token, cached on disk between processes.

    Dozens of web_research processes run at once. Each shelling out to gcloud
    would cost more than the embed call it is authorising.
    """
    try:
        if time.time() - os.stat(_TOKEN_CACHE).st_mtime < _TOKEN_TTL:
            with open(_TOKEN_CACHE, encoding="utf-8") as fh:
                token = fh.read().strip()
            if token:
                return token
    except OSError:
        pass

    with _resolve_lock:  # one gcloud call, not one per thread
        try:
            if time.time() - os.stat(_TOKEN_CACHE).st_mtime < _TOKEN_TTL:
                with open(_TOKEN_CACHE, encoding="utf-8") as fh:
                    token = fh.read().strip()
                if token:
                    return token
        except OSError:
            pass
        try:
            token = subprocess.run(
                ["gcloud", "auth", "application-default", "print-access-token"],
                capture_output=True, text=True, timeout=60,
            ).stdout.strip()
        except Exception as exc:
            _log(f"gcloud token fetch failed: {exc}")
            return None
    if not token:
        _log("gcloud returned no ADC token; run: gcloud auth application-default login")
        return None

    try:  # atomic, so a concurrent reader never sees a half-written token
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(_TOKEN_CACHE))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token)
        os.chmod(tmp, 0o600)
        os.replace(tmp, _TOKEN_CACHE)
    except OSError:
        pass
    return token


def _vertex_predict(texts: List[str], task_type: str) -> Optional[List[List[float]]]:
    """One predict call for up to VERTEX_MAX_BATCH texts. None on failure."""
    project = _gcp_project()
    if not project:
        _log("no GCP project; set GOOGLE_CLOUD_PROJECT or run gcloud config set project")
        return None
    token = _access_token()
    if not token:
        return None

    url = (f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/projects/{project}"
           f"/locations/{VERTEX_LOCATION}/publishers/google/models/{VERTEX_MODEL}:predict")
    body = {
        "instances": [{"content": t[:8000], "task_type": task_type} for t in texts],
        "parameters": {"outputDimensionality": VERTEX_DIM},
    }
    payload = json.dumps(body).encode()

    for attempt in range(VERTEX_ATTEMPTS):
        req = urllib.request.Request(
            url, data=payload,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=VERTEX_TIMEOUT) as resp:
                data = json.loads(resp.read())
            out = [p["embeddings"]["values"] for p in data["predictions"]]
            if len(out) != len(texts):
                _log(f"vertex returned {len(out)} embeddings for {len(texts)} inputs")
                return None
            return out
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429, 500, 502, 503, 504) and attempt < VERTEX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
                continue
            _log(f"vertex HTTP {exc.code}: {exc.read()[:200]!r}")
            return None
        except Exception as exc:
            if attempt < VERTEX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
                continue
            _log(f"vertex request failed: {exc}")
            return None
    return None


def _embed_ollama(text: str) -> Optional[List[float]]:
    """Get embedding from a local Ollama model. Returns a float list or None."""
    try:
        payload = json.dumps({"model": OLLAMA_MODEL, "input": text[:8000]}).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        emb = data.get("embeddings", [[]])[0]
        return emb if emb else None
    except Exception as exc:
        _log(f"ollama embed failed: {exc}")
        return None


def embed_batch(texts: List[str], task_type: str = TASK_DOCUMENT) -> List[Optional[List[float]]]:
    """Embed many texts. Result is positional; a failed slot is None."""
    if not texts:
        return []
    if EMBED_PROVIDER == "ollama":
        return [_embed_ollama(t) for t in texts]

    out: List[Optional[List[float]]] = []
    for i in range(0, len(texts), VERTEX_MAX_BATCH):
        chunk = texts[i:i + VERTEX_MAX_BATCH]
        got = _vertex_predict(chunk, task_type)
        out.extend(got if got is not None else [None] * len(chunk))
    return out


def embed_one(text: str, task_type: str = TASK_DOCUMENT) -> Optional[List[float]]:
    """Embed a single text."""
    return embed_batch([text], task_type)[0]


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
            # Dozens of writers run concurrently; wait rather than raise on a lock.
            self._conn.execute("PRAGMA busy_timeout=30000")
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
                    content_hash TEXT NOT NULL,
                    model TEXT
                )
            """)
            self._migrate(self._conn)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_domain ON pages(domain)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_fetched ON pages(fetched_at)")
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_model_fetched ON pages(model, fetched_at)")
            self._conn.commit()
        return self._conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Add the model column to a pre-existing cache.

        Every vector written before this column existed came from Ollama
        nomic-embed-text, so that is what those rows are labelled.
        """
        cols = {row[1] for row in conn.execute("PRAGMA table_info(pages)")}
        if "model" in cols:
            return
        try:
            conn.execute("ALTER TABLE pages ADD COLUMN model TEXT")
        except sqlite3.OperationalError:
            return  # another process won the race
        conn.execute(
            "UPDATE pages SET model = ? WHERE embedding IS NOT NULL AND model IS NULL",
            ("ollama:nomic-embed-text",),
        )
        conn.commit()

    def lookup(
        self,
        query: str,
        max_age_hours: int = 168,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> List[CacheHit]:
        """Find cached pages similar to query.

        Returns top_k results ranked by:
            similarity * 0.6 + authority * 0.3 + freshness * 0.1

        The ranking scan reads embeddings only. Page content is fetched in a
        second query, for the winners alone. Scanning content made every lookup
        read hundreds of MB it then threw away.
        """
        if min_similarity is None:
            min_similarity = min_similarity_default()

        query_emb = embed_one(query, TASK_QUERY)
        if query_emb is None:
            return []

        conn = self._connect()
        cutoff = time.time() - (max_age_hours * 3600)
        rows = conn.execute(
            "SELECT url, embedding, authority, fetched_at "
            "FROM pages WHERE fetched_at > ? AND embedding IS NOT NULL AND model = ?",
            (cutoff, embed_space()),
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

    @staticmethod
    def _embed_text(title: str, content: str) -> str:
        """Title plus the opening of the body captures the topic cheaply."""
        return f"{title} {content[:500]}"

    def store(
        self,
        url: str,
        domain: str,
        title: str,
        content: str,
        query: str,
    ) -> None:
        """Store one fetched page. Prefer store_many for a set of results."""
        self.store_many([(url, domain, title, content, query)])

    def store_many(self, items: Sequence[Tuple[str, str, str, str, str]]) -> int:
        """Store pages as (url, domain, title, content, query), embedded in one batch.

        Batching is why this exists: per item the embed cost falls roughly
        twelvefold against one call per page. A page whose embed fails is still
        stored, with a null embedding, for backfill_embeddings.py to pick up.
        """
        rows = [it for it in items if it[3] and len(it[3]) >= 100]
        if not rows:
            return 0

        embeddings = embed_batch(
            [self._embed_text(title, content) for _, _, title, content, _ in rows],
            TASK_DOCUMENT,
        )
        space = embed_space()
        now = time.time()

        payload = []
        for (url, domain, title, content, query), emb in zip(rows, embeddings):
            payload.append((
                url, domain, title, content, query,
                _embed_to_blob(emb) if emb else None,
                domain_authority(domain), now,
                hashlib.sha256(content.encode()).hexdigest()[:16],
                space if emb else None,
            ))

        conn = self._connect()
        conn.executemany(
            "INSERT OR REPLACE INTO pages "
            "(url, domain, title, content, query, embedding, authority, fetched_at, "
            "content_hash, model) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            payload,
        )
        conn.commit()

        failed = sum(1 for e in embeddings if e is None)
        if failed:
            _log(f"{failed}/{len(rows)} embeddings failed; those rows are not searchable yet")
        return len(payload)

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

        by_model = conn.execute(
            "SELECT COALESCE(model, '(none)'), COUNT(*) FROM pages GROUP BY 1 ORDER BY 2 DESC"
        ).fetchall()
        searchable = conn.execute(
            "SELECT COUNT(*) FROM pages WHERE embedding IS NOT NULL AND model = ? "
            "AND fetched_at > ?", (embed_space(), now - 604800),
        ).fetchone()[0]

        return {
            "total_pages": total,
            "with_embeddings": with_emb,
            "total_chars": total_chars,
            "unique_domains": domains,
            "oldest": time.strftime("%Y-%m-%d %H:%M", time.localtime(oldest)) if oldest else None,
            "newest": time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)) if newest else None,
            "fresh_24h": fresh,
            "fresh_7d": week,
            "by_model": by_model,
            "active_model": embed_space(),
            "searchable_7d": searchable,
            "db_size_mb": round(os.path.getsize(self.db_path) / 1048576, 1) if os.path.exists(self.db_path) else 0,
        }

    def print_stats(self) -> None:
        """Print cache stats to stderr."""
        s = self.stats()
        print(f"Cache: {s['total_pages']} pages ({s['with_embeddings']} with embeddings), "
              f"{s['unique_domains']} domains, {s['db_size_mb']}MB", file=sys.stderr)
        print(f"  Fresh: {s['fresh_24h']} (<24h), {s['fresh_7d']} (<7d)", file=sys.stderr)
        print(f"  Active model: {s['active_model']} "
              f"({s['searchable_7d']} searchable in the 7d window)", file=sys.stderr)
        for model, count in s["by_model"]:
            print(f"    {count:>7}  {model}", file=sys.stderr)
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
