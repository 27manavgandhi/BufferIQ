"""
Engagement predictor for hashtags.

Predicts engagement lift from using hashtags.
"""

from typing import List
import numpy as np
from sklearn.linear_model import LinearRegression


class EngagementPredictor:
    """
    Predict engagement from hashtags.

    Uses simple linear regression to predict engagement lift.

    Example:
```python
        predictor = EngagementPredictor()

        # Train on historical data
        predictor.train(
            hashtag_counts=[0, 1, 3, 5, 7],
            engagements=[80, 100, 140, 150, 145]
        )

        # Predict
        predicted = predictor.predict(hashtag_count=5)
        print(f"Predicted engagement: {predicted:.1f}")
```
    """

    def __init__(self) -> None:
        """Initialize predictor."""
        self.model = LinearRegression()
        self.is_trained = False

    def train(
        self, hashtag_counts: List[int], engagements: List[float]
    ) -> float:
        """
        Train predictor on historical data.

        Args:
            hashtag_counts: List of hashtag counts
            engagements: Corresponding engagements

        Returns:
            R² score
        """
        X = np.array(hashtag_counts).reshape(-1, 1)
        y = np.array(engagements)

        self.model.fit(X, y)
        self.is_trained = True

        # Return R² score
        return float(self.model.score(X, y))

    def predict(self, hashtag_count: int) -> float:
        """
        Predict engagement for hashtag count.

        Args:
            hashtag_count: Number of hashtags

        Returns:
            Predicted engagement
        """
        if not self.is_trained:
            # Return simple estimate if not trained
            return 100.0 + (hashtag_count * 10.0)

        X = np.array([[hashtag_count]])
        prediction = self.model.predict(X)[0]

        return float(prediction)

    def predict_lift(
        self, baseline_engagement: float, with_hashtag_count: int
    ) -> float:
        """
        Predict engagement lift.

        Args:
            baseline_engagement: Engagement without hashtags
            with_hashtag_count: Number of hashtags to use

        Returns:
            Predicted lift (e.g., 0.25 for 25% increase)
        """
        predicted_engagement = self.predict(with_hashtag_count)

        if baseline_engagement == 0:
            return 0.0

        lift = (predicted_engagement - baseline_engagement) / baseline_engagement

        return lift