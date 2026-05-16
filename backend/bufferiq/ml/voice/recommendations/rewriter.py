"""
Content rewriting for voice alignment.

Rewrites content to better align with brand voice.
"""

from typing import List, Optional
import re


class VoiceRewriter:
    """
    Rewrite content for voice alignment.
    
    Applies transformations to align content with
    target voice characteristics.
    
    Example:
```python
        rewriter = VoiceRewriter()
        rewritten = rewriter.adjust_formality(
            text="Hey! Check this out!",
            target_formality=80
        )
```
    """
    
    def __init__(self):
        """Initialize voice rewriter."""
        self.casual_to_formal = {
            "hey": "hello",
            "yeah": "yes",
            "nope": "no",
            "gonna": "going to",
            "wanna": "want to",
            "kinda": "kind of",
            "sorta": "sort of",
        }
        
        self.formal_to_casual = {v: k for k, v in self.casual_to_formal.items()}
    
    def adjust_formality(self, text: str, target_formality: float) -> str:
        """
        Adjust text formality to match target.
        
        Args:
            text: Text to adjust
            target_formality: Target formality (0-100)
        
        Returns:
            Adjusted text
        """
        if target_formality > 70:
            # Make more formal
            return self._make_formal(text)
        elif target_formality < 40:
            # Make more casual
            return self._make_casual(text)
        else:
            # Keep as is
            return text
    
    def _make_formal(self, text: str) -> str:
        """Make text more formal."""
        # Replace casual words
        for casual, formal in self.casual_to_formal.items():
            text = re.sub(r'\b' + casual + r'\b', formal, text, flags=re.IGNORECASE)
        
        # Replace exclamation marks with periods
        text = text.replace('!', '.')
        
        # Remove emojis (simple approach)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "]+",
            flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        return text.strip()
    
    def _make_casual(self, text: str) -> str:
        """Make text more casual."""
        # Replace formal words
        for formal, casual in self.formal_to_casual.items():
            text = re.sub(r'\b' + formal + r'\b', casual, text, flags=re.IGNORECASE)
        
        # Add occasional exclamation
        if '.' in text and '!' not in text:
            text = text.replace('.', '!', 1)
        
        return text
    
    def adjust_sentence_length(
        self, text: str, target_length: float
    ) -> str:
        """
        Adjust average sentence length.
        
        Args:
            text: Text to adjust
            target_length: Target average length in words
        
        Returns:
            Adjusted text
        """
        from nltk import sent_tokenize
        
        sentences = sent_tokenize(text)
        
        # Calculate current average
        words_per_sent = [len(s.split()) for s in sentences]
        if not words_per_sent:
            return text
        
        import statistics
        current_avg = statistics.mean(words_per_sent)
        
        if current_avg > target_length + 5:
            # Break long sentences
            return self._break_long_sentences(text)
        elif current_avg < target_length - 5:
            # Combine short sentences
            return self._combine_short_sentences(text)
        
        return text
    
    def _break_long_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones."""
        from nltk import sent_tokenize
        
        sentences = sent_tokenize(text)
        result = []
        
        for sent in sentences:
            words = sent.split()
            if len(words) > 20:
                # Try to break at conjunctions
                if ' and ' in sent:
                    parts = sent.split(' and ', 1)
                    result.append(parts[0] + '.')
                    result.append('And ' + parts[1])
                elif ', ' in sent:
                    parts = sent.split(', ', 1)
                    result.append(parts[0] + '.')
                    result.append(parts[1].capitalize())
                else:
                    result.append(sent)
            else:
                result.append(sent)
        
        return ' '.join(result)
    
    def _combine_short_sentences(self, text: str) -> str:
        """Combine short sentences."""
        from nltk import sent_tokenize
        
        sentences = sent_tokenize(text)
        result = []
        buffer = ""
        
        for sent in sentences:
            words = sent.split()
            if len(words) < 8:
                if buffer:
                    buffer += ", " + sent[0].lower() + sent[1:]
                else:
                    buffer = sent
            else:
                if buffer:
                    result.append(buffer)
                    buffer = ""
                result.append(sent)
        
        if buffer:
            result.append(buffer)
        
        return ' '.join(result)