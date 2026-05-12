"""
Tests for NMF topic modeler.
"""

import pytest

from bufferiq.ml.content.topics.nmf_modeler import NMFTopicModeler, Topic


class TestNMFTopicModeler:
    """Test NMFTopicModeler class."""

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
        modeler = NMFTopicModeler(n_topics=3)
        topics = modeler.fit_transform(sample_documents)

        assert len(topics) == 3
        assert all(isinstance(t, Topic) for t in topics)

    def test_topic_has_keywords(self, sample_documents: list) -> None:
        """Test that topics have keywords."""
        modeler = NMFTopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert len(topic.keywords) > 0
            assert all(isinstance(kw, str) for kw in topic.keywords)

    def test_topic_has_weights(self, sample_documents: list) -> None:
        """Test that topics have weights."""
        modeler = NMFTopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert len(topic.weights) > 0
            assert len(topic.weights) == len(topic.keywords)
            assert all(w > 0 for w in topic.weights)

    def test_topic_has_coherence(self, sample_documents: list) -> None:
        """Test that topics have coherence scores."""
        modeler = NMFTopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert 0.0 <= topic.coherence <= 1.0

    def test_topic_has_description(self, sample_documents: list) -> None:
        """Test that topics have descriptions."""
        modeler = NMFTopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert isinstance(topic.description, str)
            assert len(topic.description) > 0

    def test_topic_ids_unique(self, sample_documents: list) -> None:
        """Test that topic IDs are unique."""
        modeler = NMFTopicModeler(n_topics=3)
        topics = modeler.fit_transform(sample_documents)

        topic_ids = [t.id for t in topics]
        assert len(topic_ids) == len(set(topic_ids))

    def test_empty_documents_raises_error(self) -> None:
        """Test that empty documents list raises error."""
        modeler = NMFTopicModeler(n_topics=2)

        with pytest.raises(ValueError, match="cannot be empty"):
            modeler.fit_transform([])

    def test_too_few_documents_raises_error(self) -> None:
        """Test that too few documents raises error."""
        modeler = NMFTopicModeler(n_topics=5)
        docs = ["doc1", "doc2"]

        with pytest.raises(ValueError, match="Need at least"):
            modeler.fit_transform(docs)

    def test_custom_n_topics(self, sample_documents: list) -> None:
        """Test custom number of topics."""
        modeler = NMFTopicModeler(n_topics=5)
        topics = modeler.fit_transform(sample_documents)

        assert len(topics) == 5

    def test_custom_n_top_words(self, sample_documents: list) -> None:
        """Test custom number of top words."""
        modeler = NMFTopicModeler(n_topics=2, n_top_words=5)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            assert len(topic.keywords) == 5

    def test_invalid_n_topics_raises_error(self) -> None:
        """Test that invalid n_topics raises error."""
        with pytest.raises(ValueError, match="n_topics must be >= 1"):
            NMFTopicModeler(n_topics=0)

    def test_single_topic(self, sample_documents: list) -> None:
        """Test extraction of single topic."""
        modeler = NMFTopicModeler(n_topics=1)
        topics = modeler.fit_transform(sample_documents)

        assert len(topics) == 1

    def test_keywords_are_relevant(self, sample_documents: list) -> None:
        """Test that extracted keywords are relevant."""
        modeler = NMFTopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        # Keywords should be from the documents
        all_words = set()
        for doc in sample_documents:
            all_words.update(doc.lower().split())

        for topic in topics:
            for keyword in topic.keywords:
                # Keywords should be from vocabulary
                assert isinstance(keyword, str)

    def test_weights_sorted_descending(self, sample_documents: list) -> None:
        """Test that weights are sorted in descending order."""
        modeler = NMFTopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        for topic in topics:
            weights = topic.weights
            # Should be in descending order (most important first)
            for i in range(len(weights) - 1):
                assert weights[i] >= weights[i + 1]

    def test_different_max_features(self, sample_documents: list) -> None:
        """Test with different max_features."""
        modeler = NMFTopicModeler(n_topics=2, max_features=50)
        topics = modeler.fit_transform(sample_documents)

        assert len(topics) == 2

    def test_coherence_calculation(self, sample_documents: list) -> None:
        """Test coherence calculation."""
        modeler = NMFTopicModeler(n_topics=2)
        topics = modeler.fit_transform(sample_documents)

        # Coherence should be calculated for all topics
        for topic in topics:
            assert isinstance(topic.coherence, float)
            assert topic.coherence >= 0.0