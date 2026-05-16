"""
Voice signature generation.

Generates unique cryptographic signatures
for voice profiles.
"""

import hashlib
import json
from typing import Dict, Any


class VoiceSignatureGenerator:
    """
    Generate unique voice signatures.
    
    Creates SHA-256 hashes from voice profile data
    for identity verification and change detection.
    
    Example:
```python
        generator = VoiceSignatureGenerator()
        signature = generator.generate(profile_data)
        print(f"Signature: {signature}")
```
    """
    
    def __init__(self):
        """Initialize signature generator."""
        pass
    
    def generate(self, profile_data: Dict[str, Any]) -> str:
        """
        Generate unique voice signature hash.
        
        Args:
            profile_data: Profile characteristics
        
        Returns:
            SHA-256 signature (hex string)
        
        Raises:
            ValueError: If profile_data is empty
        """
        if not profile_data:
            raise ValueError("Cannot generate signature from empty profile data")
        
        # Serialize profile data (sorted for consistency)
        serialized = json.dumps(profile_data, sort_keys=True)
        
        # Generate SHA-256 hash
        signature = hashlib.sha256(serialized.encode()).hexdigest()
        
        return signature
    
    def verify(self, profile_data: Dict[str, Any], signature: str) -> bool:
        """
        Verify signature matches profile data.
        
        Args:
            profile_data: Profile characteristics
            signature: Expected signature
        
        Returns:
            True if signature matches
        """
        calculated = self.generate(profile_data)
        return calculated == signature
    
    def compare_signatures(self, sig1: str, sig2: str) -> bool:
        """
        Compare two signatures for equality.
        
        Args:
            sig1: First signature
            sig2: Second signature
        
        Returns:
            True if signatures match
        """
        return sig1 == sig2