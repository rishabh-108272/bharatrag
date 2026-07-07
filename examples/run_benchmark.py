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

    results = {}

    for language in dataset["languages"]:
        lang_data = [d for d in dataset["data"] if d["language"] == language]

        # ── LANGUAGE EVALUATION ──────────────────────────────────────
        print("\n" + "─" * 60)
        print(f"{language.upper()} EVALUATION")
        print("─" * 60)

        # Test 1: Correct answers
        correct_results = evaluate(
            questions=[d["question"] for d in lang_data],
            contexts=[d["context"] for d in lang_data],
            answers=[d["ground_truth_answer"] for d in lang_data],
            language=language
        )
        print("\n✅ With CORRECT answers:")
        print(f"   Context Relevance:  {correct_results['context_relevance']}")
        print(f"   Groundedness:       {correct_results['groundedness']}")
        print(f"   Answer Relevance:   {correct_results['answer_relevance']}")
        print(f"   Overall:            {correct_results['overall']}")

        # Test 2: Hallucinated answers
        hallucinated_results = evaluate(
            questions=[d["question"] for d in lang_data],
            contexts=[d["context"] for d in lang_data],
            answers=[d["hallucinated_answer"] for d in lang_data],
            language=language
        )
        print("\n❌ With HALLUCINATED answers:")
        print(f"   Context Relevance:  {hallucinated_results['context_relevance']}")
        print(f"   Groundedness:       {hallucinated_results['groundedness']}")
        print(f"   Answer Relevance:   {hallucinated_results['answer_relevance']}")
        print(f"   Overall:            {hallucinated_results['overall']}")

        results[language] = {
            "correct": correct_results,
            "hallucinated": hallucinated_results,
        }

    # ── SUMMARY TABLE ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    header = f"\n{'Metric':<25}"
    for language in dataset["languages"]:
        header += f" {language.capitalize() + '✅':>10} {language.capitalize() + '❌':>10}"
    print(header)
    print("-" * (25 + 21 * len(dataset["languages"])))

    for metric, label in [
        ("context_relevance", "Context Relevance"),
        ("groundedness", "Groundedness"),
        ("answer_relevance", "Answer Relevance"),
        ("overall", "Overall"),
    ]:
        row = f"{label:<25}"
        for language in dataset["languages"]:
            row += f" {results[language]['correct'][metric]:>10} {results[language]['hallucinated'][metric]:>10}"
        print(row)

    print("\n✅ = correct answers   ❌ = hallucinated answers")
    print("BharatRAG should score LOWER for hallucinated answers.")


if __name__ == "__main__":
    run_benchmark()