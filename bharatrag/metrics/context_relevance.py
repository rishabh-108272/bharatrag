"""
Context Relevance Metric
Measures: Did we retrieve the right context for this question?
Score 0-1. Higher = more relevant context retrieved.
"""

from bharatrag.embeddings.indic_embeddings import IndicEmbedder


class ContextRelevance:
    """
    Computes how relevant the retrieved context chunks are
    to the user's question.

    How it works:
        1. Embed the question
        2. Embed each context chunk
        3. Compute similarity between question and each chunk
        4. Return average similarity as the score

    Example:
        >>> cr = ContextRelevance(language="hindi")
        >>> score = cr.score(
        ...     question="भारत की राजधानी क्या है?",
        ...     contexts=["भारत की राजधानी नई दिल्ली है।",
        ...               "मुंबई एक बड़ा शहर है।"]
        ... )
        >>> print(score)  # 0.61
    """

    def __init__(self, language: str = "hindi"):
        """
        Args:
            language: "hindi", "marathi", or "english"
        """
        self.language = language
        self.embedder = IndicEmbedder(language=language)

    def score(self, question: str, contexts: list) -> float:
        """
        Compute context relevance score.

        Args:
            question: the user's question (string)
            contexts: list of retrieved context chunks (list of strings)

        Returns:
            float between 0 and 1
            0 = contexts completely irrelevant to question
            1 = contexts perfectly relevant to question
        """
        if not contexts:
            return 0.0

        # Get similarity of question vs every context chunk
        similarities = self.embedder.similarity_one_to_many(
            question, contexts
        )

        # Average score across all chunks
        avg_score = sum(similarities) / len(similarities)
        return round(avg_score, 4)

    def score_detailed(self, question: str, contexts: list) -> dict:
        """
        Same as score() but returns details for each chunk.
        Useful for debugging — see which chunks are relevant.

        Returns:
            dict with overall score + per-chunk breakdown
        """
        if not contexts:
            return {"overall": 0.0, "chunks": []}

        similarities = self.embedder.similarity_one_to_many(
            question, contexts
        )

        chunks_detail = [
            {
                "chunk": ctx[:100] + "..." if len(ctx) > 100 else ctx,
                "score": round(sim, 4)
            }
            for ctx, sim in zip(contexts, similarities)
        ]

        return {
            "overall": round(sum(similarities) / len(similarities), 4),
            "chunks": chunks_detail,
            "language": self.language
        }