import json
import os
import hashlib
from pathlib import Path
from datetime import datetime

CACHE_FILE = Path(__file__).resolve().parents[2] / "data" / "query_cache.json"

def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

def make_cache_key(question: str) -> str:
    """Normalize and hash the question — case insensitive, strip whitespace."""
    normalized = question.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cached_response(question: str, return_full: bool = False):
    cache = load_cache()
    key   = make_cache_key(question)
    if key in cache:
        print(f"   [Cache] HIT — returning cached answer")
        # Update hit count and last accessed
        cache[key]["hits"] = cache[key].get("hits", 0) + 1
        cache[key]["last_accessed"] = datetime.now().isoformat()
        save_cache(cache)
        return cache[key] if return_full else cache[key].get("answer")
    print(f"   [Cache] MISS — running full pipeline")
    return None

def should_cache(question: str) -> bool:
    """Don't cache vague relative-date questions whose answers change over time."""
    q = question.strip().lower()
    relative_terms = ["next month", "this month", "right now", "currently", "today"]
    return not any(term in q for term in relative_terms)

def store_in_cache(question: str, answer: str, sql_data: str | None = None):
    cache = load_cache()
    key   = make_cache_key(question)
    cache[key] = {
        "question":      question.strip(),
        "answer":        answer,
        "sql_data":      sql_data,
        "cached_at":     datetime.now().isoformat(),
        "last_accessed": datetime.now().isoformat(),
        "hits":          0
    }
    save_cache(cache)
    print(f"   [Cache] Stored new entry (total cached: {len(cache)})")
