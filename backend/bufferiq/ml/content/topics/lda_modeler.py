"""
LDA-based topic modeling.

Uses Latent Dirichlet Allocation for topic extraction.
"""

from typing import List

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from bufferiq.ml.content.topics.nmf_modeler import Topic


class LDATopicModeler:
    """
        Extract topics using LDA.

        Latent Dirichlet Allocation is a probabilistic model that
        assumes documents are mixtures of topics.

        Example:
    ```python
            modeler = LDATopicModeler(n_topics=5)
            docs = ["AI and machine learning", "Data science analytics"]
            topics = modeler.fit_transform(docs)
            for topic in topics:
                print(f"Topic {topic.id}: {topic.keywords[:5]}")
    ```
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.95,
        n_top_words: int = 10,
    ) -> None:
        """
        Initialize LDA topic modeler.

        Args:
            n_topics: Number of topics to extract
            max_features: Maximum vocabulary size
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            n_top_words: Number of top words per topic

        Raises:
            ValueError: If n_topics < 1
        """
        if n_topics < 1:
            raise ValueError("n_topics must be >= 1")

        self.n_topics = n_topics
        self.n_top_words = n_top_words

        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )

        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100,
        )

        self.feature_names_: List[str] = []

    def fit_transform(self, documents: List[str]) -> List[Topic]:
        """
        Fit model and extract topics.

        Args:
            documents: List of text documents

        Returns:
            List of extracted topics

        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        if len(documents) < self.n_topics:
            raise ValueError(
                f"Need at least {self.n_topics} documents for {self.n_topics} topics"
            )

        # Vectorize documents
        count_matrix = self.vectorizer.fit_transform(documents)
        self.feature_names_ = self.vectorizer.get_feature_names_out().tolist()

        # Fit LDA
        self.lda.fit(count_matrix)

        # Extract topics
        topics = []
        for topic_idx, topic_weights in enumerate(self.lda.components_):
            # Get top words
            top_indices = topic_weights.argsort()[-self.n_top_words :][::-1]
            keywords = [self.feature_names_[i] for i in top_indices]
            weights = [float(topic_weights[i]) for i in top_indices]

            # Calculate coherence (simplified)
            coherence = self._calculate_coherence(keywords, documents)

            # Create description
            description = ", ".join(keywords[:5])

            topics.append(
                Topic(
                    id=topic_idx,
                    keywords=keywords,
                    weights=weights,
                    coherence=coherence,
                    description=description,
                )
            )

        return topics

    def _calculate_coherence(self, keywords: List[str], documents: List[str]) -> float:
        """
        Calculate topic coherence (simplified).

        Args:
            keywords: Topic keywords
            documents: Document corpus

        Returns:
            Coherence score (0-1)
        """
        if len(keywords) < 2:
            return 0.0

        top_words = keywords[:5]
        cooccurrence_count = 0
        total_pairs = 0

        for i, word1 in enumerate(top_words):
            for word2 in top_words[i + 1 :]:
                total_pairs += 1
                for doc in documents:
                    if word1 in doc.lower() and word2 in doc.lower():
                        cooccurrence_count += 1
                        break

        return cooccurrence_count / total_pairs if total_pairs > 0 else 0.0
