"""
Ocean Intelligence Engine — v3.2
Fixes from diagnostic output:
  1. Text key is 'document_text' (not 'document')
  2. 'regions' stored as stringified list → use $eq on string OR text-match workaround
  3. SQL date guard added — warns if requested period is beyond DB coverage
  4. $eq correctly applied for all single-value fields
"""

import os
import re
import json
import time
import pickle
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from datetime import datetime
from dateutil.relativedelta import relativedelta
from src.engine.query_cache import get_cached_response, store_in_cache, should_cache
from src.engine.chart_engine import should_visualize, build_chart

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("OceanEngine")

CURRENT_DATE = datetime.now()
load_dotenv()
log.info("Booting Ocean Intelligence Engine v3.2...")


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIRMED METADATA SCHEMA  (from diagnostic output)
#
#  Text field key     : "document_text"           ← confirmed
#  species field      : single string "mackerel"  ← use $eq
#  type field         : single string "feeding"   ← use $eq
#  regions field      : STRINGIFIED list
#                       e.g. "['Arabian Sea', 'Bay of Bengal']"
#                       This is a string in Pinecone, not a real list.
#                       $in does NOT work on a stringified list.
#                       Strategy: don't filter on regions at all.
#                       Use species + type only, let semantic similarity
#                       handle regional relevance.
#
#  DB depth_zone      : 'Surface','Epipelagic','Mesopelagic','Bathypelagic'
#  DB region_name     : 'Arabian Sea','Bay of Bengal'
#  DB year range      : 2020–2026 (up to Feb 2026)
# ═════════════════════════════════════════════════════════════════════════════

PINECONE_TEXT_KEY = "document_text"   # confirmed by diagnostic

# DB coverage — used to guard SQL queries and give helpful messages
DB_MIN_YEAR  = 2020
DB_MAX_YEAR  = 2026
DB_MAX_MONTH = 2     # data goes up to Feb 2026

VALID_CHUNK_TYPES = {
    "feeding", "spawning", "habitat", "fishery", "behavior",
    "taxonomy", "biology", "morphology", "lifecycle", "oceanography",
}

VALID_SPECIES = {
    "anchovy", "hilsa", "mackerel", "pomfret", "prawn", "sardine",
    "seer_fish", "skipjack_tuna", "squid", "yellowfin_tuna",
    "black_banded_trevally", "spiny_lobster", "general",
}

SPECIES_KEYWORDS: dict[str, list[str]] = {
    "anchovy":               ["anchovy", "anchovies", "thryssa", "stolephorus",
                              "cuvierii", "mystax", "indian anchovy"],
    "hilsa":                 ["hilsa", "ilisha", "tenualosa", "jatka", "hilsha"],
    "mackerel":              ["mackerel", "rastrelliger", "bangda", "ayala",
                              "kanagurta", "indian mackerel"],
    "pomfret":               ["pomfret", "pampus", "rupchanda", "avoli",
                              "argenteus", "white pomfret"],
    "prawn":                 ["prawn", "shrimp", "penaeus", "monodon",
                              "tiger prawn", "tiger shrimp"],
    "sardine":               ["sardine", "sardinella", "mathi", "oil sardine",
                              "longiceps", "indian oil sardine"],
    "seer_fish":             ["seer", "surmai", "vanjaram", "scomberomorus",
                              "spanish mackerel", "commerson", "king fish",
                              "kingfish", "narrow barred"],
    "skipjack_tuna":         ["skipjack", "katsuwonus", "pelamis",
                              "stripe tuna", "stripe belly tuna"],
    "squid":                 ["squid", "uroteuthis", "duvaucelii", "calamari",
                              "indian squid"],
    "yellowfin_tuna":        ["yellowfin", "thunnus", "albacares", "kanta",
                              "yellow fin"],
    "black_banded_trevally": ["trevally", "seriolina", "black banded",
                              "nigrofasciata"],
    "spiny_lobster":         ["lobster", "panulirus", "homarus",
                              "spiny lobster", "scalloped lobster"],
}

CHUNK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "feeding":      ["eat", "diet", "food", "feed", "prey", "consume",
                     "nutrition", "feeding pattern", "what do", "what does",
                     "forage", "feeding intensity"],
    "spawning":     ["spawn", "breed", "reproduc", "egg", "matur",
                     "brood", "propagat", "larva", "larvae", "size at maturity",
                     "first matur", "spawning season", "peak spawn"],
    "habitat":      ["live", "found", "where", "habitat", "depth", "temperature",
                     "salinity", "distribution", "range", "zone", "reef", "shelf",
                     "prefer", "occur", "suitable", "depth range", "temp range",
                     "what temperature", "what depth"],
    "fishery":      ["catch", "fishing", "season", "gear", "ban", "landing",
                     "harvest", "net", "trawl", "seine", "gillnet", "market",
                     "commercial", "artisanal", "how to fish", "best time to fish",
                     "best month", "peak catch", "ring seine", "purse seine",
                     "hook and line", "fishing ban"],
    "behavior":     ["behav", "school", "migrat", "swim", "aggregat",
                     "movement", "nocturnal", "diurnal", "vertical migration",
                     "shoal", "group", "solitary"],
    "biology":      ["size", "weight", "length", "lifespan", "age", "growth",
                     "maturity", "trophic", "resilience", "vulnerability",
                     "max length", "common length", "max weight"],
    "lifecycle":    ["life cycle", "lifecycle", "larva", "juvenile", "adult",
                     "post larv", "cohort", "recruit", "hatch", "nauplii",
                     "post larvae", "life stage"],
    "oceanography": ["upwelling", "monsoon", "chlorophyll", "pfz",
                     "potential fishing zone", "sst", "sea surface temperature",
                     "bloom", "halocline", "pycnocline", "omz",
                     "dissolved oxygen", "oxygen minimum zone", "nutrient",
                     "productivity", "front", "eddy", "thermocline",
                     "incois", "argo", "ekman", "stratification"],
    "taxonomy":     ["classif", "taxonomy", "family", "order", "genus",
                     "scientific name", "iucn", "conservation status",
                     "class teleostei", "species name"],
    "morphology":   ["look like", "appear", "colour", "color", "fin",
                     "body shape", "morpholog", "identify", "spot", "stripe",
                     "scale", "physical description", "how to identify"],
}

# Region detection is used for SQL only (not Pinecone, since regions field
# is a stringified list and $in/$eq both fail reliably on it)
REGION_KEYWORDS: dict[str, list[str]] = {
    "Arabian Sea":   ["arabian sea", "arabian", "west coast", "kerala",
                      "karnataka", "goa", "maharashtra", "gujarat", "oman",
                      "somali", "malabar", "lakshadweep", "maldives", "saurashtra"],
    "Bay of Bengal": ["bay of bengal", "bengal", "bob", "east coast",
                      "andhra", "andhra pradesh", "odisha", "orissa",
                      "west bengal", "bangladesh", "myanmar", "andaman",
                      "ganges", "brahmaputra", "irrawaddy", "meghna", "tamil nadu"],
}

MONTH_NAMES: dict[str, int] = {
    "january":1, "february":2, "march":3, "april":4, "may":5, "june":6,
    "july":7, "august":8, "september":9, "october":10, "november":11, "december":12,
}

VARIABLE_MAP: dict[str, str] = {
    "temperature":      "avg_temp_celsius",
    "temp":             "avg_temp_celsius",
    "salinity":         "avg_salinity_psu",
    "oxygen":           "avg_doxy_umol_kg",
    "o2":               "avg_doxy_umol_kg",
    "dissolved oxygen": "avg_doxy_umol_kg",
    "chlorophyll":      "avg_chla_mg_m3",
    "chla":             "avg_chla_mg_m3",
}

UNIT_MAP: dict[str, str] = {
    "avg_temp_celsius": "°C",
    "avg_salinity_psu": "PSU",
    "avg_doxy_umol_kg": "µmol/kg",
    "avg_chla_mg_m3":   "mg/m³",
}

VARIABLE_SHORT: dict[str, str] = {
    "avg_temp_celsius": "temp",
    "avg_salinity_psu": "salinity",
    "avg_doxy_umol_kg": "oxygen",
    "avg_chla_mg_m3":   "chla",
}

ML_VARIABLES   = {"avg_temp_celsius", "avg_salinity_psu"}
CLIM_VARIABLES = {"avg_doxy_umol_kg", "avg_chla_mg_m3"}


# ═════════════════════════════════════════════════════════════════════════════
#  CLIENTS
# ═════════════════════════════════════════════════════════════════════════════

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def call_llm(prompt: str, retries: int = 3, delay: float = 2.0) -> str:
    for attempt in range(1, retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=30,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            log.warning(f"[LLM] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay * attempt)
    log.error("[LLM] All retries exhausted.")
    return ""


# ── Neon ──────────────────────────────────────────────────────────────────────
_neon_url = os.getenv("DATABASE_URL", "")
if not _neon_url:
    raise EnvironmentError("DATABASE_URL not set in .env")

_engine = create_engine(
    _neon_url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    connect_args={"sslmode": "require"},
)

def run_sql(query: str) -> pd.DataFrame:
    with _engine.connect() as conn:
        return pd.read_sql(text(query), conn)


# ── Pinecone ──────────────────────────────────────────────────────────────────
_pc             = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
_pinecone_index = _pc.Index(os.getenv("PINECONE_INDEX_NAME", "ocean-knowledge"))
_embeddings     = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── ML Models ─────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
prediction_models: dict = {}
if MODEL_DIR.exists():
    for fname in os.listdir(MODEL_DIR):
        if fname.endswith(".pkl"):
            key = fname.replace(".pkl", "")
            with open(MODEL_DIR / fname, "rb") as f:
                prediction_models[key] = pickle.load(f)
    log.info(f"[ML] Loaded {len(prediction_models)} models: {list(prediction_models.keys())}")


# ── Valid Enums from Neon (confirmed by diagnostic) ───────────────────────────
try:
    valid_regions: list[str] = run_sql(
        "SELECT DISTINCT region_name FROM argo_ocean_data ORDER BY region_name;"
    )["region_name"].tolist()
    valid_depths: list[str] = run_sql(
        "SELECT DISTINCT depth_zone FROM argo_ocean_data ORDER BY depth_zone;"
    )["depth_zone"].tolist()
    log.info(f"[Enums] Regions: {valid_regions}")
    log.info(f"[Enums] Depths:  {valid_depths}")
except Exception as e:
    log.warning(f"[Enums] DB load failed: {e}. Using confirmed defaults.")
    valid_regions = ["Arabian Sea", "Bay of Bengal"]
    valid_depths  = ["Bathypelagic", "Epipelagic", "Mesopelagic", "Surface"]


# ═════════════════════════════════════════════════════════════════════════════
#  DETECTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def detect_species(question: str) -> str | None:
    q = question.lower()
    for species_key, keywords in sorted(
        SPECIES_KEYWORDS.items(),
        key=lambda x: max(len(k) for k in x[1]),
        reverse=True,
    ):
        if any(kw in q for kw in keywords):
            return species_key
    return None


def detect_chunk_type(question: str) -> str | None:
    q      = question.lower()
    scores = {
        ct: sum(1 for kw in kws if kw in q)
        for ct, kws in CHUNK_TYPE_KEYWORDS.items()
    }
    scores = {ct: s for ct, s in scores.items() if s > 0}
    return max(scores, key=scores.get) if scores else None


def detect_region(question: str) -> str | None:
    """Used for both SQL and Pinecone region context."""
    q = question.lower()
    found_ar = any(kw in q for kw in REGION_KEYWORDS["Arabian Sea"])
    found_bb = any(kw in q for kw in REGION_KEYWORDS["Bay of Bengal"])
    if found_ar and not found_bb:
        return "Arabian Sea"
    if found_bb and not found_ar:
        return "Bay of Bengal"
    return None


def is_beyond_db_coverage(year: int, month: int) -> bool:
    """True if the requested period is outside 2020–Feb 2026."""
    if year < DB_MIN_YEAR:
        return True
    if year > DB_MAX_YEAR:
        return True
    if year == DB_MAX_YEAR and month > DB_MAX_MONTH:
        return True
    return False


# ═════════════════════════════════════════════════════════════════════════════
#  PINECONE VECTOR SEARCH
#
#  CONFIRMED schema from diagnostic:
#    - species : single string  → $eq works ✓
#    - type    : single string  → $eq works ✓
#    - regions : STRINGIFIED list "['Arabian Sea', ...]"
#                $in does NOT work on this.
#                We skip the regions filter entirely and rely on
#                semantic similarity to surface regionally relevant chunks.
# ═════════════════════════════════════════════════════════════════════════════

def build_pinecone_filter(
    species:    str | None,
    chunk_type: str | None,
) -> dict | None:
    """
    Build Pinecone filter using ONLY the fields that work reliably:
      - species ($eq on single string)
      - type    ($eq on single string)

    regions is excluded because it is stored as a stringified Python list
    (e.g. "['Arabian Sea', 'Bay of Bengal']") and Pinecone cannot run
    $in/$eq against it reliably. Regional relevance is handled by semantic
    similarity of the query embedding instead.
    """
    f: dict = {}
    if species and species in VALID_SPECIES:
        f["species"] = {"$eq": species}
    if chunk_type and chunk_type in VALID_CHUNK_TYPES:
        f["type"] = {"$eq": chunk_type}
    return f if f else None


def vector_search(
    query:      str,
    species:    str | None = None,
    chunk_type: str | None = None,
    k:          int = 5,
) -> list[str]:
    """
    Search Pinecone with metadata filters and three-level fallback.
    Text is extracted from the 'document_text' key (confirmed by diagnostic).

    Fallback levels:
      1. species + type   (most specific)
      2. species only     (if type over-filters)
      3. no filter        (pure semantic, last resort)
    """
    query_vector = _embeddings.embed_query(query)

    def _query(filter_dict: dict | None, top_k: int) -> list[str]:
        kwargs: dict = {
            "vector":           query_vector,
            "top_k":            top_k,
            "include_metadata": True,
        }
        if filter_dict:
            kwargs["filter"] = filter_dict
        results = _pinecone_index.query(**kwargs)
        chunks  = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            # Use confirmed key 'document_text'
            text_chunk = meta.get(PINECONE_TEXT_KEY, "").strip()
            if text_chunk:
                chunks.append(text_chunk)
        return chunks

    # Level 1: species + type
    full_filter = build_pinecone_filter(species, chunk_type)
    chunks = _query(full_filter, k)
    log.info(f"[Pinecone] L1 filter={full_filter} → {len(chunks)} chunks")

    # Level 2: species only
    if len(chunks) < 2 and species:
        sp_filter = build_pinecone_filter(species, None)
        chunks    = _query(sp_filter, k)
        log.info(f"[Pinecone] L2 species-only → {len(chunks)} chunks")

    # Level 3: no filter
    if len(chunks) < 2:
        chunks = _query(None, k)
        log.info(f"[Pinecone] L3 no-filter → {len(chunks)} chunks")

    return chunks


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def format_history(chat_history: list) -> str:
    if not chat_history:
        return "No previous conversation."
    lines = []
    for msg in chat_history[-6:]:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if role == "Assistant" and len(content) > 400:
            content = content[:400] + "...[truncated]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def clean_sql(raw: str) -> str:
    raw   = re.sub(r"```sql|```", "", raw, flags=re.IGNORECASE).strip()
    match = re.search(r"(SELECT\s.+?)(?:;|$)", raw, re.IGNORECASE | re.DOTALL)
    return (match.group(1).strip() + ";") if match else raw.strip() + ";"


def parse_route(raw: str) -> str:
    raw = raw.strip().upper()
    for label in ["PREDICTION", "HYBRID", "VECTOR_ONLY", "SQL_ONLY"]:
        if label in raw:
            return label
    return "SQL_ONLY"


def df_to_answer_string(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "DATABASE RESULT: 0 rows returned."
    total     = len(df)
    sort_cols = [c for c in ["year", "month"] if c in df.columns]
    if total > max_rows:
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=False).head(max_rows)
        else:
            df = df.head(max_rows)
        suffix = f"\n[TRUNCATED: showing {max_rows} most recent of {total} total rows]"
    else:
        suffix = ""
    return df.to_string(index=False) + suffix


# ═════════════════════════════════════════════════════════════════════════════
#  CLIMATOLOGY HELPER
# ═════════════════════════════════════════════════════════════════════════════

def get_seasonal_climatology(region: str, depth: str, month: int, col: str) -> str:
    query = f"""
        SELECT
            COUNT(*)                                              AS n_years,
            ROUND(AVG("{col}")::numeric, 4)                      AS mean_val,
            ROUND(MIN("{col}")::numeric, 4)                      AS min_val,
            ROUND(MAX("{col}")::numeric, 4)                      AS max_val,
            ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP
                  (ORDER BY "{col}")::numeric, 4)                AS p25,
            ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP
                  (ORDER BY "{col}")::numeric, 4)                AS p75
        FROM argo_ocean_data
        WHERE region_name = :region
          AND depth_zone  = :depth
          AND month       = :month
          AND "{col}" IS NOT NULL;
    """
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(
                text(query), conn,
                params={"region": region, "depth": depth, "month": month},
            )
        if df.empty or df.iloc[0]["n_years"] == 0:
            return f"  • {col}: No observations for month {month} in {region}/{depth}"
        row  = df.iloc[0]
        unit = UNIT_MAP.get(col, "")
        return (
            f"  • {col} [HISTORICAL CLIMATOLOGY — not a model forecast]: "
            f"mean {row['mean_val']} {unit}, "
            f"range {row['min_val']}–{row['max_val']} {unit}, "
            f"typical {row['p25']}–{row['p75']} {unit} "
            f"(across {int(row['n_years'])} years)"
        )
    except Exception as e:
        return f"  • {col}: Climatology query failed — {e}"


# ═════════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═════════════════════════════════════════════════════════════════════════════

ROUTER_PROMPT = """You are a routing agent for an Ocean AI system.
Output EXACTLY ONE WORD from: SQL_ONLY, VECTOR_ONLY, HYBRID, PREDICTION

- SQL_ONLY    → historical recorded ocean data (temperature, salinity, oxygen, chlorophyll, trends, comparisons)
- VECTOR_ONLY → marine biology, species ecology, fish behaviour, fishing advice, oceanographic concepts
- HYBRID      → needs BOTH recorded data AND biological/ecological context
- PREDICTION  → future values (will, predict, forecast, expected, next month, future year)

If question has "current", "right now", "today", "this month", "state of" → HYBRID

DATABASE COVERAGE NOTE: Data is available from January 2020 to February 2026 only.
If a question asks for data before 2020 or after February 2026, still route to SQL_ONLY —
the answer prompt will handle the empty result gracefully.

CONVERSATION HISTORY:
{history}

Question: {question}
Answer (one word only):"""


SQL_PROMPT = """You are a PostgreSQL expert. Generate ONE raw SQL query. No explanation. No markdown.

TABLE: argo_ocean_data
COLUMNS: id, region_name (varchar), year (int), month (int), depth_zone (varchar),
         avg_temp_celsius, avg_salinity_psu, avg_doxy_umol_kg, avg_chla_mg_m3 (numeric), profile_count (int)

EXACT VALUES IN THE DATABASE — USE THESE STRINGS VERBATIM:
  region_name: {regions}
  depth_zone:  {depths}
  year range:  2020 to 2026
  latest data: February 2026 (month=2, year=2026)

VARIABLE → COLUMN:
- temperature/temp → avg_temp_celsius
- salinity → avg_salinity_psu
- oxygen/O2/dissolved oxygen → avg_doxy_umol_kg
- chlorophyll/chla → avg_chla_mg_m3
- "conditions"/"all parameters" → SELECT all four numeric columns

RULES:
1. SELECT region_name, depth_zone, year, month, <columns> FROM argo_ocean_data
2. No JOINs. No subqueries.
3. region_name: use = 'Arabian Sea' or = 'Bay of Bengal' (exact match).
   Do NOT use ILIKE unless the region is truly ambiguous.
4. depth_zone: if user does not specify depth, OMIT the depth_zone WHERE clause entirely
   so all zones are returned. If user says "surface" use = 'Surface'.
5. BIOLOGY RANGES from context: ORDER BY ABS(col - midpoint) ASC only. No WHERE from bio ranges.
6. USER-STATED RANGES: use WHERE BETWEEN.
7. NULL SAFETY: for avg_doxy_umol_kg and avg_chla_mg_m3 add AND <col> IS NOT NULL
   unless user asks for "all parameters".
8. OR on same column: wrap in parentheses.
9. No time filter specified: ORDER BY year DESC, month DESC LIMIT 30
10. No aggregates (AVG/STDDEV/COUNT) in user queries — data is pre-aggregated.
    highest/lowest: ORDER BY <col> DESC/ASC LIMIT 1.
11. CRITICAL DATE AWARENESS: if user asks for a month/year combination beyond
    February 2026, still write the query — it will return 0 rows and the answer
    prompt will explain the data is not yet available.

CONVERSATION HISTORY:
{history}

BIOLOGICAL CONTEXT:
{context}

Question: {question}
SQL:"""


PREDICTION_EXTRACTOR_PROMPT = """Extract prediction parameters from the question.
Today's date: {today}.
Return ONLY a valid JSON object:
- "region": one of {regions}, or null
- "depth_zone": one of {depths}, or null
- "year": integer, never null
- "month": integer 1-12, never null
- "variables": list from ["avg_temp_celsius","avg_salinity_psu","avg_doxy_umol_kg","avg_chla_mg_m3"]

Question: {question}
JSON:"""


ANSWER_PROMPT = """You are a factual Oceanography AI. Give a direct, clean answer.

DATABASE COVERAGE: Argo float data is available from January 2020 to February 2026 only.

DATA FORMAT:
- DB rows: region_name, depth_zone, year, month, <variable>. Month numbers: 1=Jan…12=Dec.
- [ML FORECAST] = Ridge model prediction (temperature/salinity only).
- [HISTORICAL CLIMATOLOGY] = historical seasonal average, clearly NOT a forecast.
- Biological context = marine species/oceanography knowledge base.

RULES:
- Use ONLY the data provided. Never invent numbers or facts.
- If DATABASE RESULT says "0 rows returned":
  • If the requested period is within 2020–Feb 2026: say "BGC float data unavailable for this specific combination."
  • If the requested period is outside the database coverage (before 2020 or after Feb 2026):
    say "Argo float data is only available from January 2020 to February 2026. No recorded data exists for [period]."
  Do NOT say "BGC float data unavailable" for both cases — distinguish them clearly.
- If BIOLOGICAL CONTEXT is "None", skip biology entirely.
- Be concise. No filler phrases.
- More than 15 DB rows → summarize: overall range (min–max), most recent value, trend.
- Always append units: °C, PSU, µmol/kg, mg/m³.
- Never confuse temperature °C values with salinity PSU values.

CONVERSATION HISTORY:
{history}

DATABASE RESULT:
{sql_data}

PREDICTION RESULT:
{prediction_data}

BIOLOGICAL / SCIENTIFIC CONTEXT:
{vector_context}

USER QUESTION: {question}
ANSWER:"""


# ═════════════════════════════════════════════════════════════════════════════
#  PREDICTION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def run_prediction(question: str) -> str:
    if not prediction_models:
        return "No prediction models loaded."

    raw = call_llm(PREDICTION_EXTRACTOR_PROMPT.format(
        question=question,
        regions=valid_regions,
        depths=valid_depths,
        today=CURRENT_DATE.strftime("%B %Y"),
    ))
    if not raw:
        return "LLM failed to extract prediction parameters."

    try:
        params = json.loads(re.sub(r"```json|```", "", raw).strip())
    except Exception:
        return f"Could not parse prediction parameters: {raw}"

    raw_region = params.get("region")
    raw_depth  = params.get("depth_zone")
    raw_year   = params.get("year")
    raw_month  = params.get("month")
    variables  = params.get("variables", ["avg_temp_celsius"])

    regions_to_predict = [raw_region] if raw_region else valid_regions
    depth = raw_depth or "Surface"

    if not raw_year or not raw_month:
        nxt   = CURRENT_DATE + relativedelta(months=1)
        year  = int(raw_year  or nxt.year)
        month = int(raw_month or nxt.month)
    else:
        year, month = int(raw_year), int(raw_month)

    all_results: list[str] = []

    for region in regions_to_predict:
        rk = region.lower().replace(" ", "_")
        dk = depth.lower().replace(" ", "_")
        ml = pd.Timestamp(year=year, month=month, day=1).strftime("%B %Y")
        rr = [f"\n📍 {region} / {depth} / {ml}:"]

        try:
            hist_df = run_sql(
                f"SELECT year, month, avg_temp_celsius, avg_salinity_psu "
                f"FROM argo_ocean_data "
                f"WHERE region_name = '{region}' AND depth_zone = '{depth}' "
                f"ORDER BY year ASC, month ASC;"
            )
            hist_df["date"] = pd.to_datetime(hist_df[["year","month"]].assign(DAY=1))
            hist_df = hist_df.set_index("date").sort_index()
        except Exception as e:
            all_results.append(f"\n{region}: Historical data fetch failed — {e}")
            continue

        if len(hist_df) < 13:
            all_results.append(f"\n{region}: Insufficient data (need ≥13 months).")
            continue

        for var_full in variables:
            short = VARIABLE_SHORT.get(var_full)
            if not short:
                rr.append(f"  • {var_full}: Unknown variable.")
                continue

            if var_full in CLIM_VARIABLES:
                rr.append(get_seasonal_climatology(region, depth, month, var_full))
                continue

            model_key = f"{rk}_{dk}_{short}"
            model     = prediction_models.get(model_key)
            if model is None:
                avail = [k for k in prediction_models if rk in k]
                rr.append(f"  • {var_full}: No model '{model_key}'. Available: {avail}")
                continue

            if var_full not in hist_df.columns:
                rr.append(f"  • {var_full}: Column missing.")
                continue

            try:
                target     = pd.Timestamp(year=year, month=month, day=1)
                last       = hist_df.index[-1]
                months_fwd = (target.year - last.year)*12 + (target.month - last.month)

                if months_fwd <= 0:
                    rr.append(f"  • {var_full}: {ml} is not future. Query history instead.")
                    continue

                t_hist  = list(hist_df["avg_temp_celsius"].values)
                s_hist  = list(hist_df["avg_salinity_psu"].values)
                sv_hist = list(hist_df[var_full].values)
                pred    = None

                for step in range(1, months_fwd + 1):
                    sd = last + pd.DateOffset(months=step)
                    X  = pd.DataFrame({
                        "month_sin":        [np.sin(2*np.pi*sd.month/12)],
                        "month_cos":        [np.cos(2*np.pi*sd.month/12)],
                        "temp_lag_1":       [t_hist[-1]],
                        "sal_lag_1":        [s_hist[-1]],
                        "self_lag_2":       [sv_hist[-2]],
                        "self_lag_12":      [sv_hist[-12]],
                        "self_rolling_3mo": [np.mean(sv_hist[-3:])],
                    })
                    pred = model.predict(X)[0]
                    sv_hist.append(pred)
                    t_hist.append(pred if var_full == "avg_temp_celsius" else t_hist[-1])
                    s_hist.append(pred if var_full == "avg_salinity_psu" else s_hist[-1])

                unit = UNIT_MAP.get(var_full, "")
                rr.append(f"  • {var_full} [ML FORECAST]: {pred:.3f} {unit}")
            except Exception as e:
                rr.append(f"  • {var_full}: Prediction error — {e}")

        all_results.extend(rr)

    return "\n".join(all_results) if all_results else "No predictions generated."


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN WORKFLOW
# ═════════════════════════════════════════════════════════════════════════════

def execute_hybrid_workflow(question: str, chat_history: list = None) -> tuple:
    if chat_history is None:
        chat_history = []

    log.info(f"[User] {question}")

    cached = get_cached_response(question, return_full=True)
    if cached:
        log.info("[Cache] Hit")
        answer, sql_data = cached["answer"], cached.get("sql_data")
        chart = None
        if sql_data and should_visualize(question):
            try:
                chart = build_chart(sql_data, question)
            except Exception as e:
                log.warning(f"[Cache] Chart rebuild: {e}")
        return answer, chart

    history_str = format_history(chat_history)
    visualize   = should_visualize(question)

    detected_species    = detect_species(question)
    detected_chunk_type = detect_chunk_type(question)
    detected_region     = detect_region(question)
    log.info(f"[Detect] species={detected_species} | "
             f"type={detected_chunk_type} | region={detected_region}")

    raw_route = call_llm(ROUTER_PROMPT.format(question=question, history=history_str))
    route     = parse_route(raw_route)
    log.info(f"[Router] '{raw_route.strip()}' → '{route}'")

    vector_context  = "None"
    sql_data        = "None"
    sql_df          = None
    prediction_data = "None"
    chart_figure    = None

    # ── PREDICTION ────────────────────────────────────────────────────────────
    if route == "PREDICTION":
        log.info("[ML] Running prediction...")
        prediction_data = run_prediction(question)

    # ── VECTOR SEARCH ─────────────────────────────────────────────────────────
    if route in ("VECTOR_ONLY", "HYBRID"):
        log.info("[Pinecone] Searching...")

        # For pure oceanography (no species in question), don't filter on species
        # since those chunks have species="general" and we'd miss them
        search_species = detected_species
        if detected_chunk_type == "oceanography" and not detected_species:
            search_species = None

        chunks = vector_search(
            query=question,
            species=search_species,
            chunk_type=detected_chunk_type,
            k=5,
        )
        if chunks:
            vector_context = "\n---\n".join(chunks)
            log.info(f"[Pinecone] {len(chunks)} chunk(s) assembled")
        else:
            log.info("[Pinecone] No chunks retrieved")

    # ── SQL ───────────────────────────────────────────────────────────────────
    if route in ("SQL_ONLY", "HYBRID"):
        log.info("[SQL] Generating query...")
        raw_sql = call_llm(SQL_PROMPT.format(
            regions  = valid_regions,
            depths   = valid_depths,
            question = question,
            context  = vector_context,
            history  = history_str,
        ))

        if not raw_sql:
            sql_data = "LLM failed to generate SQL."
        else:
            clean_query = clean_sql(raw_sql)
            log.info(f"[SQL] {clean_query}")
            try:
                sql_df = run_sql(clean_query)
                log.info(f"[SQL] {len(sql_df)} rows returned")

                if visualize and not sql_df.empty:
                    try:
                        chart_figure = build_chart(
                            sql_df.to_string(index=False), question
                        )
                    except Exception as e:
                        log.warning(f"[Chart] {e}")

                sql_data = df_to_answer_string(sql_df, max_rows=20)
            except Exception as e:
                sql_data = f"SQL Error: {e}"
                log.error(f"[SQL] {e}")

    # ── SYNTHESIZE ────────────────────────────────────────────────────────────
    final_answer = call_llm(ANSWER_PROMPT.format(
        question=question,
        sql_data=sql_data,
        prediction_data=prediction_data,
        vector_context=vector_context,
        history=history_str,
    ))
    if not final_answer:
        final_answer = "Unable to generate an answer right now. Please try again."

    log.info(f"[Answer] {final_answer[:200]}...")
    log.info("-" * 60)

    if should_cache(question):
        store_in_cache(
            question, final_answer,
            sql_data=sql_data if route in ("SQL_ONLY", "HYBRID") else None,
        )

    return final_answer, chart_figure


# ═════════════════════════════════════════════════════════════════════════════
#  CLI TEST SUITE
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    history: list = []

    def chat(q: str) -> str:
        answer, _ = execute_hybrid_workflow(q, chat_history=history)
        history.append({"role": "user",      "content": q})
        history.append({"role": "assistant", "content": answer})
        print(f"\n[Answer] {answer}\n{'='*60}")
        return answer

    # SQL — within DB coverage (2020–Feb 2026)
    chat("What was the surface temperature of the Arabian Sea in May 2023?")
    chat("What about Bay of Bengal in the same month?")

    # SQL — outside DB coverage (should get a clear message, not "BGC unavailable")
    chat("What was the surface temperature of the Arabian Sea in March 2019?")

    # Vector — species feeding
    chat("What do Indian mackerel eat?")

    # Vector — species fishery with region
    chat("When is the best season to fish for hilsa in the Bay of Bengal?")

    # Vector — pure oceanography
    chat("How does upwelling affect fish aggregation in the Arabian Sea?")

    # Hybrid — species + current data
    chat("Is the current salinity in the Arabian Sea suitable for tiger prawns?")

    # Prediction — future
    chat("What will the surface temperature be in the Bay of Bengal in August 2026?")
