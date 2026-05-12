"""
Topic coherence calculation.

Calculates coherence scores for topic models.
"""

from typing import List


class CoherenceCalculator:
    """
        Calculate topic coherence scores.

        Measures how coherent topic keywords are based on
        their co-occurrence in documents.

        Example:
    ```python
            calculator = CoherenceCalculator()
            score = calculator.calculate(
                keywords=["machine", "learning", "AI"],
                documents=corpus
            )
            print(f"Coherence: {score:.3f}")
    ```
    """

    def calculate(self, keywords: List[str], documents: List[str]) -> float:
        """
        Calculate coherence score.

        Args:
            keywords: Topic keywords
            documents: Document corpus

        Returns:
            Coherence score (0-1)

        Raises:
            ValueError: If keywords or documents are empty
        """
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        if not documents:
            raise ValueError("Documents list cannot be empty")

        if len(keywords) < 2:
            return 0.0

        # Calculate pairwise co-occurrence
        cooccurrence_count = 0
        total_pairs = 0

        for i, word1 in enumerate(keywords):
            for word2 in keywords[i + 1 :]:
                total_pairs += 1
                for doc in documents:
                    doc_lower = doc.lower()
                    if word1.lower() in doc_lower and word2.lower() in doc_lower:
                        cooccurrence_count += 1
                        break

        coherence = cooccurrence_count / total_pairs if total_pairs > 0 else 0.0
        return coherence

    def calculate_umass(self, keywords: List[str], documents: List[str]) -> float:
        """
        Calculate UMass coherence.

        Args:
            keywords: Topic keywords
            documents: Document corpus

        Returns:
            UMass coherence score

        Raises:
            ValueError: If keywords or documents are empty
        """
        if not keywords:
            raise ValueError("Keywords list cannot be empty")
        if not documents:
            raise ValueError("Documents list cannot be empty")

        # Simplified UMass coherence
        return self.calculate(keywords, documents)
