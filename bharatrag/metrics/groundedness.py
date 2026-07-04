"""
Groundedness Metric
Measures: Is the answer actually based on the context,
or is the LLM hallucinating?
Score 0-1. Higher = answer is more grounded in context.
"""

from bharatrag.embeddings.indic_embeddings import IndicEmbedder


class Groundedness:
    """
    Checks if every claim in the answer is supported
    by the retrieved context.

    How it works:
        1. Split the answer into individual sentences (claims)
        2. For each claim, check similarity against all context chunks
        3. If best similarity > threshold → claim is supported
        4. Score = supported_claims / total_claims

    Example:
        >>> gr = Groundedness(language="hindi")
        >>> score = gr.score(
        ...     answer="भारत की राजधानी नई दिल्ली है।",
        ...     contexts=["भारत की राजधानी नई दिल्ली है।"]
        ... )
        >>> print(score)  # 1.0 — fully grounded
    """

    def __init__(self, language: str = "hindi", threshold: float = 0.45):
        """
        Args:
            language:  "hindi", "marathi", or "english"
            threshold: similarity score above which a claim
                       is considered supported (default 0.45)
        """
        self.language = language
        self.threshold = threshold
        self.embedder = IndicEmbedder(language=language)

    def _split_into_claims(self, text: str) -> list:
        """
        Split text into individual claims/sentences.
        We treat each sentence as one claim to verify.

        Args:
            text: answer string

        Returns:
            list of sentence strings
        """
        # Split on Hindi/English sentence endings
        import re
        sentences = re.split(r'[।\.!\?]+', text)
        # Remove empty strings and strip whitespace
        claims = [s.strip() for s in sentences if s.strip()]
        return claims

    def score(self, answer: str, contexts: list) -> float:
        """
        Compute groundedness score.

        Args:
            answer:   the generated answer to check
            contexts: list of retrieved context chunks

        Returns:
            float between 0 and 1
            0 = answer is completely hallucinated
            1 = every claim in answer is supported by context
        """
        if not answer or not contexts:
            return 0.0

        claims = self._split_into_claims(answer)

        if not claims:
            return 0.0

        supported = 0

        for claim in claims:
            # Compare this claim against ALL context chunks
            similarities = self.embedder.similarity_one_to_many(
                claim, contexts
            )
            # Best match score across all chunks
            best_score = max(similarities)

            if best_score >= self.threshold:
                supported += 1

        score = supported / len(claims)
        return round(score, 4)

    def score_detailed(self, answer: str, contexts: list) -> dict:
        """
        Same as score() but shows which claims are
        supported and which are hallucinated.
        """
        if not answer or not contexts:
            return {"overall": 0.0, "claims": []}

        claims = self._split_into_claims(answer)
        claims_detail = []

        for claim in claims:
            similarities = self.embedder.similarity_one_to_many(
                claim, contexts
            )
            best_score = max(similarities)
            is_supported = best_score >= self.threshold

            claims_detail.append({
                "claim": claim,
                "best_similarity": round(best_score, 4),
                "supported": is_supported
            })

        supported_count = sum(
            1 for c in claims_detail if c["supported"]
        )

        return {
            "overall": round(supported_count / len(claims), 4),
            "supported": supported_count,
            "total_claims": len(claims),
            "claims": claims_detail,
            "language": self.language
        }