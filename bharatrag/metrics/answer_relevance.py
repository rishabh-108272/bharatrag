"""
Answer Relevance Metric
Measures: Does the answer actually address the question?
Score 0-1. Higher = answer is more relevant to the question.
"""

from bharatrag.embeddings.indic_embeddings import IndicEmbedder


class AnswerRelevance:
    """
    Checks if the generated answer actually addresses
    the user's original question.

    How it works:
        1. Embed the original question
        2. Embed the answer
        3. Compute similarity between question and answer
        4. High similarity = answer addresses the question

    Note: This is simpler than RAGAS's approach (which generates
    reverse questions). We use direct similarity which works
    well for Hindi/Marathi without needing an LLM API call.

    Example:
        >>> ar = AnswerRelevance(language="hindi")
        >>> score = ar.score(
        ...     question="भारत की राजधानी क्या है?",
        ...     answer="भारत की राजधानी नई दिल्ली है।"
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

    def score(self, question: str, answer: str) -> float:
        """
        Compute answer relevance score.

        Args:
            question: the original user question
            answer:   the generated answer

        Returns:
            float between 0 and 1
            0 = answer is completely irrelevant to question
            1 = answer perfectly addresses the question
        """
        if not question or not answer:
            return 0.0

        similarity = self.embedder.similarity(question, answer)
        return round(similarity, 4)

    def score_detailed(self, question: str, answer: str) -> dict:
        """
        Same as score() but returns more details.
        """
        if not question or not answer:
            return {"overall": 0.0, "question": question, "answer": answer}

        similarity = self.embedder.similarity(question, answer)

        # Interpretation
        if similarity >= 0.7:
            interpretation = "Highly relevant"
        elif similarity >= 0.45:
            interpretation = "Moderately relevant"
        elif similarity >= 0.25:
            interpretation = "Weakly relevant"
        else:
            interpretation = "Not relevant"

        return {
            "overall": round(similarity, 4),
            "interpretation": interpretation,
            "question": question,
            "answer": answer[:100] + "..." if len(answer) > 100 else answer,
            "language": self.language
        }