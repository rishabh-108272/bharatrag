"""
BharatRAG Benchmark Runner
Runs evaluation on the benchmark dataset and shows results.
"""

import json
from bharatrag import evaluate


def load_benchmark(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(data_path: str = "data/benchmark.json"):
    print("=" * 60)
    print("BharatRAG Benchmark Evaluation")
    print("=" * 60)

    dataset = load_benchmark(data_path)
    print(f"Loaded {dataset['total']} examples")
    print(f"Languages: {dataset['languages']}")

    # Split by language
    hindi_data = [d for d in dataset["data"] if d["language"] == "hindi"]
    marathi_data = [d for d in dataset["data"] if d["language"] == "marathi"]

    # ── HINDI EVALUATION ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("HINDI EVALUATION")
    print("─" * 60)

    # Test 1: Correct answers
    hindi_results = evaluate(
        questions=[d["question"] for d in hindi_data],
        contexts=[d["context"] for d in hindi_data],
        answers=[d["ground_truth_answer"] for d in hindi_data],
        language="hindi"
    )
    print("\n✅ With CORRECT answers:")
    print(f"   Context Relevance:  {hindi_results['context_relevance']}")
    print(f"   Groundedness:       {hindi_results['groundedness']}")
    print(f"   Answer Relevance:   {hindi_results['answer_relevance']}")
    print(f"   Overall:            {hindi_results['overall']}")

    # Test 2: Hallucinated answers
    hindi_hallucinated = evaluate(
        questions=[d["question"] for d in hindi_data],
        contexts=[d["context"] for d in hindi_data],
        answers=[d["hallucinated_answer"] for d in hindi_data],
        language="hindi"
    )
    print("\n❌ With HALLUCINATED answers:")
    print(f"   Context Relevance:  {hindi_hallucinated['context_relevance']}")
    print(f"   Groundedness:       {hindi_hallucinated['groundedness']}")
    print(f"   Answer Relevance:   {hindi_hallucinated['answer_relevance']}")
    print(f"   Overall:            {hindi_hallucinated['overall']}")

    # ── MARATHI EVALUATION ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("MARATHI EVALUATION")
    print("─" * 60)

    marathi_results = evaluate(
        questions=[d["question"] for d in marathi_data],
        contexts=[d["context"] for d in marathi_data],
        answers=[d["ground_truth_answer"] for d in marathi_data],
        language="marathi"
    )
    print("\n✅ With CORRECT answers:")
    print(f"   Context Relevance:  {marathi_results['context_relevance']}")
    print(f"   Groundedness:       {marathi_results['groundedness']}")
    print(f"   Answer Relevance:   {marathi_results['answer_relevance']}")
    print(f"   Overall:            {marathi_results['overall']}")

    marathi_hallucinated = evaluate(
        questions=[d["question"] for d in marathi_data],
        contexts=[d["context"] for d in marathi_data],
        answers=[d["hallucinated_answer"] for d in marathi_data],
        language="marathi"
    )
    print("\n❌ With HALLUCINATED answers:")
    print(f"   Context Relevance:  {marathi_hallucinated['context_relevance']}")
    print(f"   Groundedness:       {marathi_hallucinated['groundedness']}")
    print(f"   Answer Relevance:   {marathi_hallucinated['answer_relevance']}")
    print(f"   Overall:            {marathi_hallucinated['overall']}")

    # ── SUMMARY TABLE ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<25} {'Hindi✅':>10} {'Hindi❌':>10} {'Marathi✅':>10} {'Marathi❌':>10}")
    print("-" * 65)
    print(f"{'Context Relevance':<25} {hindi_results['context_relevance']:>10} {hindi_hallucinated['context_relevance']:>10} {marathi_results['context_relevance']:>10} {marathi_hallucinated['context_relevance']:>10}")
    print(f"{'Groundedness':<25} {hindi_results['groundedness']:>10} {hindi_hallucinated['groundedness']:>10} {marathi_results['groundedness']:>10} {marathi_hallucinated['groundedness']:>10}")
    print(f"{'Answer Relevance':<25} {hindi_results['answer_relevance']:>10} {hindi_hallucinated['answer_relevance']:>10} {marathi_results['answer_relevance']:>10} {marathi_hallucinated['answer_relevance']:>10}")
    print(f"{'Overall':<25} {hindi_results['overall']:>10} {hindi_hallucinated['overall']:>10} {marathi_results['overall']:>10} {marathi_hallucinated['overall']:>10}")
    print("\n✅ = correct answers   ❌ = hallucinated answers")
    print("BharatRAG should score LOWER for hallucinated answers.")


if __name__ == "__main__":
    run_benchmark()