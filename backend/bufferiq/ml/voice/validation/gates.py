"""
Voice quality gates.

Implements quality gates and approval workflows
for voice consistency.
"""

from typing import List, Dict, Optional
from enum import Enum
import logging

from bufferiq.ml.voice.validation.validator import ValidationResult

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Approval status for content."""
    
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"
    PENDING = "pending"


class VoiceQualityGates:
    """
    Implement quality gates for voice consistency.
    
    Defines threshold-based gates and approval workflows.
    
    Example:
```python
        gates = VoiceQualityGates(
            auto_approve_threshold=85.0,
            auto_reject_threshold=50.0
        )
        
        status = gates.evaluate(validation_result)
        
        if status == ApprovalStatus.APPROVED:
            publish_content()
        elif status == ApprovalStatus.NEEDS_REVISION:
            request_revision()
        else:
            reject_content()
```
    """
    
    def __init__(
        self,
        auto_approve_threshold: float = 85.0,
        auto_reject_threshold: float = 50.0,
        require_manual_review: bool = False,
    ):
        """
        Initialize quality gates.
        
        Args:
            auto_approve_threshold: Score above which content auto-approves
            auto_reject_threshold: Score below which content auto-rejects
            require_manual_review: If True, all content needs manual review
        """
        self.auto_approve = auto_approve_threshold
        self.auto_reject = auto_reject_threshold
        self.manual_review = require_manual_review
    
    def evaluate(self, validation: ValidationResult) -> ApprovalStatus:
        """
        Evaluate content against quality gates.
        
        Args:
            validation: Validation result
        
        Returns:
            Approval status
        """
        if self.manual_review:
            logger.info("Manual review required - marking as pending")
            return ApprovalStatus.PENDING
        
        score = validation.score
        
        if score >= self.auto_approve and validation.passed:
            logger.info(f"Auto-approved: score {score:.1f} >= {self.auto_approve}")
            return ApprovalStatus.APPROVED
        
        elif score < self.auto_reject:
            logger.info(f"Auto-rejected: score {score:.1f} < {self.auto_reject}")
            return ApprovalStatus.REJECTED
        
        else:
            logger.info(f"Needs revision: score {score:.1f} in manual review range")
            return ApprovalStatus.NEEDS_REVISION
    
    def evaluate_batch(
        self, validations: List[ValidationResult]
    ) -> Dict[ApprovalStatus, List[int]]:
        """
        Evaluate batch of content.
        
        Args:
            validations: List of validation results
        
        Returns:
            Dictionary mapping status to indices
        """
        results: Dict[ApprovalStatus, List[int]] = {
            ApprovalStatus.APPROVED: [],
            ApprovalStatus.NEEDS_REVISION: [],
            ApprovalStatus.REJECTED: [],
            ApprovalStatus.PENDING: [],
        }
        
        for idx, validation in enumerate(validations):
            status = self.evaluate(validation)
            results[status].append(idx)
        
        logger.info(
            f"Batch evaluation complete: "
            f"{len(results[ApprovalStatus.APPROVED])} approved, "
            f"{len(results[ApprovalStatus.NEEDS_REVISION])} need revision, "
            f"{len(results[ApprovalStatus.REJECTED])} rejected, "
            f"{len(results[ApprovalStatus.PENDING])} pending"
        )
        
        return results
    
    def get_gate_config(self) -> Dict[str, any]:
        """
        Get current gate configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "auto_approve_threshold": self.auto_approve,
            "auto_reject_threshold": self.auto_reject,
            "require_manual_review": self.manual_review,
        }
    
    def update_thresholds(
        self,
        auto_approve: Optional[float] = None,
        auto_reject: Optional[float] = None,
    ) -> None:
        """
        Update quality gate thresholds.
        
        Args:
            auto_approve: New auto-approve threshold
            auto_reject: New auto-reject threshold
        """
        if auto_approve is not None:
            self.auto_approve = auto_approve
            logger.info(f"Updated auto-approve threshold to {auto_approve}")
        
        if auto_reject is not None:
            self.auto_reject = auto_reject
            logger.info(f"Updated auto-reject threshold to {auto_reject}")