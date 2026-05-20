"""
Anomaly detector.

Detects anomalies in experiment metrics.

Example:
```python
    detector = AnomalyDetector()
    
    anomalies = detector.detect_anomalies(
        time_series=[100, 102, 98, 150, 101]
    )
```
"""

from typing import Dict, List

import numpy as np


class AnomalyDetector:
    """
    Detect anomalies in time series.

    Example:
```python
        detector = AnomalyDetector(z_threshold=3.0)

        anomalies = detector.detect_anomalies(
            time_series=[100, 102, 98, 101, 200, 99]
        )

        print(f"Found {anomalies['num_anomalies']} anomalies")
        print(f"Indices: {anomalies['anomaly_indices']}")
```
    """

    def __init__(self, z_threshold: float = 3.0) -> None:
        """
        Initialize anomaly detector.

        Args:
            z_threshold: Z-score threshold for anomalies
        """
        self.z_threshold = z_threshold

    def detect_anomalies(self, time_series: List[float]) -> Dict[str, any]:
        """
        Detect anomalies using z-score method.

        Args:
            time_series: Time series data

        Returns:
            Anomaly detection result
        """
        if len(time_series) < 3:
            return {
                "num_anomalies": 0,
                "anomaly_indices": [],
                "anomaly_values": [],
            }

        values = np.array(time_series)
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return {
                "num_anomalies": 0,
                "anomaly_indices": [],
                "anomaly_values": [],
            }

        # Calculate z-scores
        z_scores = np.abs((values - mean) / std)

        # Find anomalies
        anomaly_mask = z_scores > self.z_threshold
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_values = values[anomaly_mask].tolist()

        return {
            "num_anomalies": len(anomaly_indices),
            "anomaly_indices": anomaly_indices,
            "anomaly_values": anomaly_values,
            "z_scores": z_scores.tolist(),
            "mean": float(mean),
            "std": float(std),
        }

    def detect_anomalies_mad(
        self, time_series: List[float], threshold: float = 3.5
    ) -> Dict[str, any]:
        """
        Detect anomalies using MAD (Median Absolute Deviation).

        More robust to outliers than z-score method.

        Args:
            time_series: Time series data
            threshold: MAD threshold

        Returns:
            Anomaly detection result
        """
        if len(time_series) < 3:
            return {
                "num_anomalies": 0,
                "anomaly_indices": [],
                "anomaly_values": [],
            }

        values = np.array(time_series)
        median = np.median(values)

        # MAD
        mad = np.median(np.abs(values - median))

        if mad == 0:
            return {
                "num_anomalies": 0,
                "anomaly_indices": [],
                "anomaly_values": [],
            }

        # Modified z-score
        modified_z_scores = 0.6745 * (values - median) / mad

        # Find anomalies
        anomaly_mask = np.abs(modified_z_scores) > threshold
        anomaly_indices = np.where(anomaly_mask)[0].tolist()
        anomaly_values = values[anomaly_mask].tolist()

        return {
            "num_anomalies": len(anomaly_indices),
            "anomaly_indices": anomaly_indices,
            "anomaly_values": anomaly_values,
            "modified_z_scores": modified_z_scores.tolist(),
            "median": float(median),
            "mad": float(mad),
        }