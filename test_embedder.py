# Quick test — run this to see if embeddings work on Hindi text
from bharatrag.embeddings.indic_embeddings import IndicEmbedder

# Load Hindi model
embedder = IndicEmbedder(language="hindi")

# Test 1 — Similar sentences (should score HIGH)
score1 = embedder.similarity(
    "भारत की राजधानी क्या है?",
    "भारत की राजधानी नई दिल्ली है।"
)
print(f"Similar sentences score: {score1:.4f}  (should be HIGH > 0.7)")

# Test 2 — Unrelated sentences (should score LOW)
score2 = embedder.similarity(
    "भारत की राजधानी क्या है?",
    "आज मौसम बहुत अच्छा है।"
)
print(f"Unrelated sentences score: {score2:.4f}  (should be LOW < 0.4)")

# Test 3 — One question vs many chunks
chunks = [
    "भारत की राजधानी नई दिल्ली है।",          # relevant
    "मुंबई भारत की आर्थिक राजधानी है।",        # somewhat relevant  
    "आज क्रिकेट मैच बहुत अच्छा था।",           # not relevant
]
scores = embedder.similarity_one_to_many(
    "भारत की राजधानी क्या है?",
    chunks
)
print(f"\nOne-to-many scores:")
for chunk, score in zip(chunks, scores):
    print(f"  {score:.4f} — {chunk}")