import json
import os
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

CHUNKS_FILE = Path(__file__).resolve().parents[2] / "data" / "chunks_clean.json"
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME   = "ocean-knowledge"
EMBED_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE   = 50
DIMENSION    = 384  # all-MiniLM-L6-v2 outputs 384 dimensions

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD CHUNKS
# ─────────────────────────────────────────────────────────────

print("\n📂 Loading chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    raw_chunks = json.load(f)

print(f"✅ Loaded {len(raw_chunks)} chunks")

# ─────────────────────────────────────────────────────────────
# STEP 2: SANITIZE METADATA
# Pinecone only accepts str, int, float, bool, list of str
# ─────────────────────────────────────────────────────────────

def sanitize_metadata(meta):
    clean = {}
    for key, value in meta.items():
        if value is None:
            continue
        elif isinstance(value, list):
            flat = [str(v) for v in value if v is not None]
            if flat:
                clean[key] = flat  # Pinecone accepts list of strings
        elif isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and value.strip() == "":
                continue
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean

# ─────────────────────────────────────────────────────────────
# STEP 3: PREPARE DATA
# ─────────────────────────────────────────────────────────────

print("🔧 Validating chunks...")

ids       = []
documents = []
metadatas = []
seen_ids  = set()
skipped   = 0

for chunk in raw_chunks:
    chunk_id = str(chunk.get("id", "")).strip()
    document = str(chunk.get("document", "")).strip()
    metadata = chunk.get("metadata", {})

    if not chunk_id or not document:
        skipped += 1
        continue

    if chunk_id in seen_ids:
        skipped += 1
        continue

    seen_ids.add(chunk_id)
    ids.append(chunk_id)
    documents.append(document)
    metadatas.append(sanitize_metadata(metadata))

print(f"✅ {len(ids)} valid chunks ready")
if skipped:
    print(f"⚠️  {skipped} chunks skipped")

# ─────────────────────────────────────────────────────────────
# STEP 4: CONNECT TO PINECONE
# ─────────────────────────────────────────────────────────────

print("\n🔌 Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_KEY)

# Delete old index if exists — fresh start
existing = [i.name for i in pc.list_indexes()]

if INDEX_NAME in existing:
    pc.delete_index(INDEX_NAME)
    print(f"🗑️  Deleted old index: {INDEX_NAME}")

# Create new index
pc.create_index(
    name=INDEX_NAME,
    dimension=DIMENSION,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

print(f"✅ Fresh index created: '{INDEX_NAME}'")

index = pc.Index(INDEX_NAME)

# ─────────────────────────────────────────────────────────────
# STEP 5: GENERATE EMBEDDINGS
# ─────────────────────────────────────────────────────────────

print("\n🧠 Generating embeddings...")

model = SentenceTransformer(EMBED_MODEL)
embeddings = model.encode(documents, show_progress_bar=True)

print(f"✅ Embeddings generated: {embeddings.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 6: UPSERT IN BATCHES
# ─────────────────────────────────────────────────────────────

print(f"\n🚀 Uploading to Pinecone in batches of {BATCH_SIZE}...")

total_loaded = 0

for i in range(0, len(ids), BATCH_SIZE):
    batch_ids   = ids[i:i+BATCH_SIZE]
    batch_embs  = embeddings[i:i+BATCH_SIZE].tolist()
    batch_meta  = metadatas[i:i+BATCH_SIZE]
    batch_docs  = documents[i:i+BATCH_SIZE]

    # Store document text inside metadata so we can retrieve it
    for j in range(len(batch_meta)):
        batch_meta[j]["document_text"] = batch_docs[j]

    vectors = [
        {
            "id": batch_ids[j],
            "values": batch_embs[j],
            "metadata": batch_meta[j]
        }
        for j in range(len(batch_ids))
    ]

    try:
        index.upsert(vectors=vectors)
        total_loaded += len(batch_ids)
        print(f"   ✅ Uploaded: {total_loaded} / {len(ids)}")
    except Exception as e:
        print(f"   ❌ Batch failed at index {i}: {e}")

print(f"\n✅ Total vectors in index: {index.describe_index_stats()['total_vector_count']}")

# ─────────────────────────────────────────────────────────────
# STEP 7: VERIFICATION TEST
# ─────────────────────────────────────────────────────────────

print("\n🧪 Running verification queries...")

test_queries = [
    ("When do anchovies spawn in Arabian Sea?", {"type": {"$eq": "spawning"}}),
    ("Best fish to catch in June?",             {"regions": {"$in": ["Arabian Sea"]}}),
    ("What is the oxygen minimum zone?",        {"type": {"$eq": "oceanography"}}),
]

for query, filter_dict in test_queries:
    query_embedding = model.encode([query]).tolist()[0]

    results = index.query(
        vector=query_embedding,
        top_k=1,
        filter=filter_dict,
        include_metadata=True
    )

    if results["matches"]:
        match = results["matches"][0]
        print(f"\n  Query : {query}")
        print(f"  Match : [{match['id']}] score={match['score']:.3f}")
        print(f"  Text  : {match['metadata'].get('document_text','')[:100]}...")
    else:
        print(f"\n  Query : {query}")
        print(f"  Match : ❌ No results — check metadata tags")

print("\n✅ Pinecone knowledge base ready to use.\n")
