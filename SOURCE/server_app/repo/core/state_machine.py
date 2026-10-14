"""任务状态机 — 状态转换规则。

UPLOADED → ANALYZING → NEEDS_REVIEW → CONFIRMED → RESTORING → COMPLETED
任何阶段都可以 → FAILED | ABSTAINED
"""
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    UPLOADED = "UPLOADED"
    ANALYZING = "ANALYZING"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    CONFIRMED = "CONFIRMED"
    RESTORING = "RESTORING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABSTAINED = "ABSTAINED"


# 合法状态转换
_ALLOWED = {
    JobStatus.UPLOADED: {JobStatus.ANALYZING, JobStatus.FAILED},
    JobStatus.ANALYZING: {JobStatus.NEEDS_REVIEW, JobStatus.FAILED},
    JobStatus.NEEDS_REVIEW: {JobStatus.CONFIRMED, JobStatus.ABSTAINED, JobStatus.FAILED},
    JobStatus.CONFIRMED: {JobStatus.RESTORING, JobStatus.FAILED},
    JobStatus.RESTORING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: {JobStatus.UPLOADED},  # 允许重试
    JobStatus.ABSTAINED: set(),
}


def can_transition(current: JobStatus, target: JobStatus) -> bool:
    return target in _ALLOWED.get(current, set())


def transition(current: JobStatus, target: JobStatus) -> JobStatus:
    if not can_transition(current, target):
        raise ValueError(f"Invalid transition: {current.value} → {target.value}")
    return target
