"""Deterministic resource limits for callback-driven execution.

Exceeding a limit fails the run. No partial result is returned and no event is
silently dropped. Budgets span all settlement passes at the same timestamp.
"""

from dataclasses import dataclass


class ExecutionLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionLimits:
    max_actions_per_timestamp: int = 100_000

    def __post_init__(self):
        if self.max_actions_per_timestamp < 1:
            raise ValueError("max_actions_per_timestamp must be positive")

    def budget(self):
        return TimestampBudget(self)


class TimestampBudget:
    def __init__(self, limits: ExecutionLimits):
        self.limits = limits
        self.timestamp = None
        self.actions = 0

    def consume(self, timestamp_ns: int, count: int = 1):
        if timestamp_ns != self.timestamp:
            self.timestamp = timestamp_ns
            self.actions = 0
        self.actions += count
        if self.actions > self.limits.max_actions_per_timestamp:
            raise ExecutionLimitExceeded(
                f"execution did not quiesce at timestamp {timestamp_ns}: "
                f"{self.actions} actions exceed limit "
                f"{self.limits.max_actions_per_timestamp}"
            )
