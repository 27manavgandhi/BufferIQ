"""Manage Optuna studies with persistence and resumption."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler

from bufferiq.core.logging import get_logger

logger = get_logger(__name__)


class OptunaStudyManager:
    """
    Manage Optuna studies with persistence.
    
    Provides CRUD operations for studies, including creation,
    loading, deletion, and export.
    """

    def __init__(self, storage: str = "sqlite:///optuna_studies.db"):
        """
        Initialize study manager.
        
        Args:
            storage: Storage URL (SQLite or PostgreSQL)
        
        Example:
            >>> manager = OptunaStudyManager("sqlite:///my_studies.db")
            >>> study = manager.create_study("my_study", direction="maximize")
        """
        self.storage = storage
        logger.info(f"Study manager initialized with storage: {storage}")

    def create_study(
        self,
        study_name: str,
        direction: str = "maximize",
        sampler: Optional[BaseSampler] = None,
        pruner: Optional[BasePruner] = None,
    ) -> optuna.Study:
        """
        Create new study.
        
        Args:
            study_name: Name for the study
            direction: Optimization direction ('maximize' or 'minimize')
            sampler: Optuna sampler
            pruner: Optuna pruner
        
        Returns:
            Created study object
        
        Raises:
            ValueError: If study already exists
        
        Example:
            >>> study = manager.create_study("xgboost_optimization", direction="maximize")
        """
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=False,
            )
            logger.info(f"Created study: {study_name}")
            return study
            
        except optuna.exceptions.DuplicatedStudyError:
            raise ValueError(f"Study '{study_name}' already exists")

    def load_study(self, study_name: str) -> optuna.Study:
        """
        Load existing study.
        
        Args:
            study_name: Name of study to load
        
        Returns:
            Loaded study object
        
        Raises:
            ValueError: If study not found
        
        Example:
            >>> study = manager.load_study("xgboost_optimization")
            >>> print(f"Loaded study with {len(study.trials)} trials")
        """
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=self.storage,
            )
            logger.info(f"Loaded study: {study_name} ({len(study.trials)} trials)")
            return study
            
        except KeyError:
            raise ValueError(f"Study '{study_name}' not found")

    def list_studies(self) -> List[str]:
        """
        List all study names.
        
        Returns:
            List of study names
        
        Example:
            >>> studies = manager.list_studies()
            >>> print(f"Found {len(studies)} studies")
        """
        try:
            studies = optuna.study.get_all_study_names(self.storage)
            logger.info(f"Found {len(studies)} studies")
            return studies
        except Exception as e:
            logger.error(f"Failed to list studies: {e}")
            return []

    def delete_study(self, study_name: str) -> None:
        """
        Delete study.
        
        Args:
            study_name: Name of study to delete
        
        Example:
            >>> manager.delete_study("old_study")
        """
        try:
            optuna.delete_study(
                study_name=study_name,
                storage=self.storage,
            )
            logger.info(f"Deleted study: {study_name}")
        except KeyError:
            logger.warning(f"Study '{study_name}' not found")

    def export_study(self, study_name: str, output_path: Path) -> None:
        """
        Export study data to JSON.
        
        Args:
            study_name: Name of study to export
            output_path: Path to save JSON file
        
        Example:
            >>> manager.export_study("my_study", Path("study_export.json"))
        """
        study = self.load_study(study_name)
        
        data = {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "trials": [
                {
                    "number": t.number,
                    "params": t.params,
                    "value": t.value,
                    "state": t.state.name,
                    "datetime_start": (
                        t.datetime_start.isoformat() if t.datetime_start else None
                    ),
                    "datetime_complete": (
                        t.datetime_complete.isoformat() if t.datetime_complete else None
                    ),
                }
                for t in study.trials
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported study to {output_path}")

    def get_study_summary(self, study_name: str) -> Dict[str, Any]:
        """
        Get summary of study.
        
        Args:
            study_name: Name of study
        
        Returns:
            Dictionary with study summary
        """
        study = self.load_study(study_name)
        
        n_complete = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ])
        n_pruned = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.PRUNED
        ])
        n_failed = len([
            t for t in study.trials
            if t.state == optuna.trial.TrialState.FAIL
        ])
        
        return {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "n_complete": n_complete,
            "n_pruned": n_pruned,
            "n_failed": n_failed,
        }