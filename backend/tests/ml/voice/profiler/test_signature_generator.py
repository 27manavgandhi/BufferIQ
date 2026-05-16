"""Tests for voice signature generator."""

import pytest
from bufferiq.ml.voice.profiler.signature_generator import VoiceSignatureGenerator


class TestVoiceSignatureGenerator:
    """Test voice signature generator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return VoiceSignatureGenerator()
    
    def test_generate_basic_signature(self, generator):
        """Test basic signature generation."""
        profile_data = {
            "lexical": {"ttr": 0.8, "density": 0.6},
            "syntactic": {"avg_len": 15},
            "stylistic": {"formality": 80},
        }
        
        signature = generator.generate(profile_data)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA-256 hex length
    
    def test_generate_empty_data_raises_error(self, generator):
        """Test empty data raises error."""
        with pytest.raises(ValueError, match="empty"):
            generator.generate({})
    
    def test_generate_deterministic(self, generator):
        """Test signature generation is deterministic."""
        profile_data = {"key": "value"}
        
        sig1 = generator.generate(profile_data)
        sig2 = generator.generate(profile_data)
        
        assert sig1 == sig2
    
    def test_generate_different_data_different_signature(self, generator):
        """Test different data produces different signatures."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        
        sig1 = generator.generate(data1)
        sig2 = generator.generate(data2)
        
        assert sig1 != sig2
    
    def test_generate_order_independent(self, generator):
        """Test signature is independent of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        
        sig1 = generator.generate(data1)
        sig2 = generator.generate(data2)
        
        assert sig1 == sig2
    
    def test_verify_correct_signature(self, generator):
        """Test verifying correct signature."""
        profile_data = {"key": "value"}
        signature = generator.generate(profile_data)
        
        is_valid = generator.verify(profile_data, signature)
        
        assert is_valid is True
    
    def test_verify_incorrect_signature(self, generator):
        """Test verifying incorrect signature."""
        profile_data = {"key": "value"}
        wrong_signature = "0" * 64
        
        is_valid = generator.verify(profile_data, wrong_signature)
        
        assert is_valid is False
    
    def test_verify_modified_data(self, generator):
        """Test verification fails for modified data."""
        original_data = {"key": "value"}
        signature = generator.generate(original_data)
        
        modified_data = {"key": "different_value"}
        is_valid = generator.verify(modified_data, signature)
        
        assert is_valid is False
    
    def test_compare_signatures_equal(self, generator):
        """Test comparing equal signatures."""
        sig1 = "abc123"
        sig2 = "abc123"
        
        assert generator.compare_signatures(sig1, sig2) is True
    
    def test_compare_signatures_different(self, generator):
        """Test comparing different signatures."""
        sig1 = "abc123"
        sig2 = "def456"
        
        assert generator.compare_signatures(sig1, sig2) is False
    
    def test_generate_with_nested_data(self, generator):
        """Test signature generation with nested data."""
        profile_data = {
            "lexical": {
                "ttr": 0.8,
                "nested": {
                    "deep": "value"
                }
            }
        }
        
        signature = generator.generate(profile_data)
        
        assert len(signature) == 64
    
    def test_generate_with_numeric_values(self, generator):
        """Test signature with various numeric types."""
        profile_data = {
            "int": 42,
            "float": 3.14,
            "negative": -10,
        }
        
        signature = generator.generate(profile_data)
        
        assert len(signature) == 64
    
    def test_generate_with_lists(self, generator):
        """Test signature generation with lists."""
        profile_data = {
            "items": [1, 2, 3],
            "words": ["a", "b", "c"],
        }
        
        signature = generator.generate(profile_data)
        
        assert len(signature) == 64
    
    def test_signature_hex_format(self, generator):
        """Test signature is valid hex string."""
        profile_data = {"key": "value"}
        signature = generator.generate(profile_data)
        
        # Should be valid hex
        int(signature, 16)  # Will raise if not valid hex
        assert all(c in '0123456789abcdef' for c in signature)