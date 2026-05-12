"""
Tests for LDA topic modeler.
"""

import pytest

from bufferiq.ml.content.topics.lda_modeler import LDATopicModeler
from bufferiq.ml.content.topics.nmf_modeler import Topic


class TestLDATopicModeler:
    """Test LDATopicModeler class."""

    @pytest.fixture
    def sample_documents(self) -> list:
        """Create sample documents for testing."""
        return [
            "machine learning artificial intelligence data science",
            "deep learning neural networks AI technology",
            "data analytics statistics machine learning",
            "artificial intelligence deep learning models",
            "big data analytics business intelligence",
            "neural networks deep learning AI systems",
            "machine learning algorithms data mining",
            "AI technology artificial intelligence future",
            "data science analytics visualization",
            "deep learning convolutional neural networks",
        ]

    def test_fit_transform_basic(self, sample_documents: list) -> None:
        """Test basic topic extraction."""
        modeler = LDATopicModeler(n_topics=3)
        topics = modeler.fit_transform(sample_documents)

        assert len(topics) == 3
        assert all(isinstance(t, Topic) for t in topics)

    def test_topic_has_keywords(self, sample_documents: list) -> None:
        """Test that topics have keywords."""
        modeler = LDATopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert len(topic.keywords) > 0
            assert all(isinstance(kw, str) for kw in topic.keywords)

    def test_topic_has_weights(self, sample_documents: list) -> None:
        """Test that topics have weights."""
        modeler = LDATopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert len(topic.weights) > 0
            assert len(topic.weights) == len(topic.keywords)
            assert all(w > 0 for w in topic.weights)

    def test_topic_has_coherence(self, sample_documents: list) -> None:
        """Test that topics have coherence scores."""
        modeler = LDATopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert 0.0 <= topic.coherence <= 1.0

    def test_empty_documents_raises_error(self) -> None:
        """Test that empty documents list raises error."""
        modeler = LDATopicModeler(n_topics=2)

        with pytest.raises(ValueError, match="cannot be empty"):
            modeler.fit_transform([])

    def test_too_few_documents_raises_error(self) -> None:
        """Test that too few documents raises error."""
        modeler = LDATopicModeler(n_topics=5)
        docs = ["doc1", "doc2"]

        with pytest.raises(ValueError, match="Need at least"):
            modeler.fit_transform(docs)

    def test_invalid_n_topics_raises_error(self) -> None:
        """Test that invalid n_topics raises error."""
        with pytest.raises(ValueError, match="n_topics must be >= 1"):
            LDATopicModeler(n_topics=0)

    def test_custom_n_topics(self, sample_documents: list) -> None:
        """Test custom number of topics."""
        modeler = LDATopicModeler(n_topics=5)
        topics = modeler.fit_transform(sample_documents)

        assert len(topics) == 5

    def test_topic_ids_unique(self, sample_documents: list) -> None:
        """Test that topic IDs are unique."""
        modeler = LDATopicModeler(n_topics=3)
        topics = modeler.fit_transform(sample_documents)

        topic_ids = [t.id for t in topics]
        assert len(topic_ids) == len(set(topic_ids))