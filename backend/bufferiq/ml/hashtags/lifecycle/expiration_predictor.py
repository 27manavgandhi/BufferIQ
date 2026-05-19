"""
Expiration predictor for hashtags.

Predicts when hashtags will become dormant.
"""

from datetime import datetime, timedelta
from typing import List, Optional


class ExpirationPredictor:
    """
    Predict hashtag expiration.

    Example:
```python
        predictor = ExpirationPredictor()

        expiration = predictor.predict_expiration(
            volume_history=[1000, 900, 700, 500, 300],
            current_date=datetime.now()
        )

        if expiration:
            print(f"Predicted expiration: {expiration}")
            days_left = (expiration - datetime.now()).days
            print(f"Days remaining: {days_left}")
```
    """

    def __init__(self, dormant_threshold: int = 10) -> None:
        """
        Initialize expiration predictor.

        Args:
            dormant_threshold: Volume threshold for dormant status
        """
        self.dormant_threshold = dormant_threshold

    def predict_expiration(
        self,
        volume_history: List[int],
        current_date: datetime,
    ) -> Optional[datetime]:
        """
        Predict when hashtag will expire (become dormant).

        Args:
            volume_history: Volume over time
            current_date: Current date

        Returns:
            Predicted expiration date or None
        """
        if not volume_history:
            return None

        current_volume = volume_history[-1]

        # Already dormant
        if current_volume <= self.dormant_threshold:
            return current_date

        # Not declining
        if len(volume_history) < 3:
            return None

        # Calculate decline rate
        decline_rate = self._calculate_decline_rate(volume_history)

        if decline_rate <= 0:
            # Not declining
            return None

        # Estimate days to dormant
        volume_to_lose = current_volume - self.dormant_threshold
        days_to_dormant = int(volume_to_lose / decline_rate)

        # Predict expiration
        expiration_date = current_date + timedelta(days=days_to_dormant)

        return expiration_date

    def _calculate_decline_rate(self, volume_history: List[int]) -> float:
        """
        Calculate decline rate per period.

        Args:
            volume_history: Volume history

        Returns:
            Decline rate (positive number)
        """
        if len(volume_history) < 2:
            return 0.0

        # Calculate average decline
        declines = []
        for i in range(1, len(volume_history)):
            decline = volume_history[i - 1] - volume_history[i]
            if decline > 0:
                declines.append(decline)

        if not declines:
            return 0.0

        avg_decline = sum(declines) / len(declines)

        return avg_decline