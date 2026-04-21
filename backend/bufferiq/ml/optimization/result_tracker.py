"""Track and persist hyperparameter optimization results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class OptimizationResultTracker:
    """
    Track optimization trials and export results.
    
    Logs each trial's parameters, score, and duration. Generates
    optimization reports and exports best parameters.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize result tracker.
        
        Args:
            output_dir: Directory to save optimization results
        
        Example:
            >>> tracker = OptimizationResultTracker(Path("outputs/optimizations"))
            >>> tracker.log_trial(1, {'lr': 0.1}, 0.73, 12.5)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.trials: List[Dict[str, Any]] = []
        self.best_trial: Optional[Dict[str, Any]] = None
        self.baseline_score: Optional[float] = None
        
        logger.info(f"Result tracker initialized at {output_dir}")

    def set_baseline(self, score: float) -> None:
        """
        Set baseline score for comparison.
        
        Args:
            score: Baseline model score before optimization
        """
        self.baseline_score = score
        logger.info(f"Baseline score set to {score:.4f}")

    def log_trial(
        self,
        trial_id: int,
        params: Dict[str, Any],
        score: float,
        duration: float,
    ) -> None:
        """
        Log a single optimization trial.
        
        Args:
            trial_id: Unique trial identifier
            params: Hyperparameters tested
            score: Cross-validation score achieved
            duration: Time taken for trial (seconds)
        """
        trial = {
            "trial_id": trial_id,
            "params": params,
            "score": score,
            "duration": duration,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.trials.append(trial)
        
        # Update best trial
        if self.best_trial is None or score > self.best_trial["score"]:
            self.best_trial = trial
            logger.info(
                f"New best trial: {trial_id}, score={score:.4f}, "
                f"params={params}"
            )

    def get_best_trial(self) -> Optional[Dict[str, Any]]:
        """
        Get the best trial found so far.
        
        Returns:
            Best trial dictionary or None if no trials logged
        """
        return self.best_trial

    def get_improvement(self) -> Optional[float]:
        """
        Calculate improvement over baseline.
        
        Returns:
            Improvement percentage, or None if no baseline/trials
        """
        if self.baseline_score is None or self.best_trial is None:
            return None
        
        improvement = (
            (self.best_trial["score"] - self.baseline_score)
            / abs(self.baseline_score)
            * 100
        )
        return improvement

    def save_trials(self, filename: str = "trials.json") -> Path:
        """
        Save all trials to JSON file.
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(self.trials, f, indent=2)
        
        logger.info(f"Saved {len(self.trials)} trials to {filepath}")
        return filepath

    def export_best_params(self, filename: str = "best_params.yaml") -> Path:
        """
        Export best parameters to YAML file.
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved file
        
        Raises:
            ValueError: If no trials have been logged
        """
        if self.best_trial is None:
            raise ValueError("No trials logged yet")
        
        filepath = self.output_dir / filename
        
        output = {
            "best_params": self.best_trial["params"],
            "best_score": self.best_trial["score"],
            "trial_id": self.best_trial["trial_id"],
            "timestamp": self.best_trial["timestamp"],
        }
        
        if self.baseline_score is not None:
            output["baseline_score"] = self.baseline_score
            output["improvement_pct"] = self.get_improvement()
        
        with open(filepath, "w") as f:
            yaml.dump(output, f, default_flow_style=False)
        
        logger.info(f"Exported best params to {filepath}")
        return filepath

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate optimization summary report.
        
        Returns:
            Dictionary containing optimization statistics
        """
        if not self.trials:
            return {"status": "No trials logged"}
        
        scores = [t["score"] for t in self.trials]
        durations = [t["duration"] for t in self.trials]
        
        report = {
            "total_trials": len(self.trials),
            "best_score": self.best_trial["score"] if self.best_trial else None,
            "best_trial_id": self.best_trial["trial_id"] if self.best_trial else None,
            "best_params": self.best_trial["params"] if self.best_trial else None,
            "mean_score": float(pd.Series(scores).mean()),
            "std_score": float(pd.Series(scores).std()),
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "total_duration": sum(durations),
            "mean_duration": float(pd.Series(durations).mean()),
        }
        
        if self.baseline_score is not None:
            report["baseline_score"] = self.baseline_score
            report["improvement"] = (
                self.best_trial["score"] - self.baseline_score
                if self.best_trial
                else 0
            )
            report["improvement_pct"] = self.get_improvement()
        
        return report

    def save_report(self, filename: str = "optimization_report.json") -> Path:
        """
        Save optimization report to JSON file.
        
        Args:
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        report = self.generate_report()
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved optimization report to {filepath}")
        return filepath