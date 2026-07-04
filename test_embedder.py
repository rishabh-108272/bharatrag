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





# Test Context Relevance Metric
print("\n" + "="*50)
print("TESTING CONTEXT RELEVANCE METRIC")
print("="*50)

from bharatrag.metrics.context_relevance import ContextRelevance

cr = ContextRelevance(language="hindi")


# Test 1 — Relevant context (should score HIGH)
score1 = cr.score(
    question="भारत की राजधानी क्या है?",
    contexts=[
        "भारत की राजधानी नई दिल्ली है।",
        "नई दिल्ली भारत का सबसे बड़ा शहर है।"
    ]
)
print(f"\nTest 1 - Relevant context: {score1}  (should be HIGH > 0.5)")

# Test 2 — Irrelevant context (should score LOW)
score2 = cr.score(
    question="भारत की राजधानी क्या है?",
    contexts=[
        "आज मौसम बहुत अच्छा है।",
        "क्रिकेट भारत का लोकप्रिय खेल है।"
    ]
)
print(f"Test 2 - Irrelevant context: {score2}  (should be LOW < 0.3)")

# Test 3 — Detailed breakdown
print("\nTest 3 - Detailed breakdown:")
details = cr.score_detailed(
    question="पीएम किसान योजना क्या है?",
    contexts=[
        "पीएम किसान सम्मान निधि योजना भारत सरकार की एक योजना है।",
        "इस योजना के तहत किसानों को हर साल 6000 रुपये मिलते हैं।",
        "आज बारिश हो रही है।"
    ]
)
print(f"Overall score: {details['overall']}")
for chunk in details['chunks']:
    print(f"  {chunk['score']} — {chunk['chunk']}")