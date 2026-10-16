"""Centralized Inference Budget Configuration for TinyText Flash V1.5.5.

Defines typed inference profiles (low, medium, high) and runtime budget enforcement.
Controls maximum generation tokens, tool calls, verification attempts, and corrective retries.
Enforces invariant global hard caps to prevent unbounded resource consumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ThinkProfile(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Invariant Global Hard Upper Bounds
# The TinyText-Flash-20M architecture has max_seq_len = 512.
# Hard caps ensure generation and orchestrator loops never exceed context limits or cause OOM.
HARD_CAP_MAX_NEW_TOKENS = 256
HARD_CAP_MAX_TOOL_CALLS = 5
HARD_CAP_MAX_VERIFICATIONS = 5
HARD_CAP_MAX_RETRIES = 3


@dataclass(frozen=True)
class InferenceBudget:
    """Immutable, typed specification of runtime inference limits."""

    profile: Optional[ThinkProfile] = None
    max_new_tokens: int = 50
    max_tool_calls: int = 1
    max_verification_attempts: int = 2
    max_retries: int = 1
    enforce_hard_caps: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.max_tool_calls < 0:
            raise ValueError(f"max_tool_calls must be non-negative, got {self.max_tool_calls}")
        if self.max_verification_attempts < 0:
            raise ValueError(f"max_verification_attempts must be non-negative, got {self.max_verification_attempts}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

        # Enforce hard caps strictly
        if self.enforce_hard_caps:
            capped_tokens = min(self.max_new_tokens, HARD_CAP_MAX_NEW_TOKENS)
            capped_tools = min(self.max_tool_calls, HARD_CAP_MAX_TOOL_CALLS)
            capped_verifs = min(self.max_verification_attempts, HARD_CAP_MAX_VERIFICATIONS)
            capped_retries = min(self.max_retries, HARD_CAP_MAX_RETRIES)
            if (
                capped_tokens != self.max_new_tokens
                or capped_tools != self.max_tool_calls
                or capped_verifs != self.max_verification_attempts
                or capped_retries != self.max_retries
            ):
                object.__setattr__(self, "max_new_tokens", capped_tokens)
                object.__setattr__(self, "max_tool_calls", capped_tools)
                object.__setattr__(self, "max_verification_attempts", capped_verifs)
                object.__setattr__(self, "max_retries", capped_retries)


# Exact Profile Configurations derived from repository defaults:
# - LOW: Latency-sensitive, fast completion, 0 retries, 1 verification attempt.
# - MEDIUM: Preserves the exact existing defaults (max_new_tokens=50, 1 tool, 2 verifications, 1 retry).
# - HIGH: Bounded expansion (max_new_tokens=128, 3 tools, 4 verifications, 2 retries), strictly < hard caps.
BUDGET_PROFILES: Dict[ThinkProfile, InferenceBudget] = {
    ThinkProfile.LOW: InferenceBudget(
        profile=ThinkProfile.LOW,
        max_new_tokens=25,
        max_tool_calls=1,
        max_verification_attempts=1,
        max_retries=0,
    ),
    ThinkProfile.MEDIUM: InferenceBudget(
        profile=ThinkProfile.MEDIUM,
        max_new_tokens=50,
        max_tool_calls=1,
        max_verification_attempts=2,
        max_retries=1,
    ),
    ThinkProfile.HIGH: InferenceBudget(
        profile=ThinkProfile.HIGH,
        max_new_tokens=128,
        max_tool_calls=3,
        max_verification_attempts=4,
        max_retries=2,
    ),
}

# Default budget when --think is omitted matches MEDIUM parameters
DEFAULT_INFERENCE_BUDGET = InferenceBudget(
    profile=None,
    max_new_tokens=50,
    max_tool_calls=1,
    max_verification_attempts=2,
    max_retries=1,
)


def get_budget_for_profile(profile: Optional[ThinkProfile | str]) -> InferenceBudget:
    """Returns the InferenceBudget corresponding to the given profile name or enum."""
    if profile is None:
        return DEFAULT_INFERENCE_BUDGET

    if isinstance(profile, str):
        profile_lower = profile.lower().strip()
        try:
            profile = ThinkProfile(profile_lower)
        except ValueError:
            valid_choices = [p.value for p in ThinkProfile]
            raise ValueError(f"Invalid think profile '{profile}'. Valid choices are: {valid_choices}")

    return BUDGET_PROFILES[profile]


@dataclass
class BudgetDiagnostics:
    """Tracks runtime consumption against allocated budgets and reports exhaustion status."""

    profile: Optional[str]
    effective_max_tokens: int
    effective_max_tool_calls: int
    effective_max_verifications: int
    effective_max_retries: int

    tokens_generated: int = 0
    tool_calls_count: int = 0
    verification_attempts: int = 0
    retries_attempted: int = 0

    token_budget_exhausted: bool = False
    tool_budget_exhausted: bool = False
    verification_budget_exhausted: bool = False
    retry_budget_exhausted: bool = False

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.effective_max_tokens - self.tokens_generated)

    @property
    def remaining_tool_calls(self) -> int:
        return max(0, self.effective_max_tool_calls - self.tool_calls_count)

    @property
    def remaining_verifications(self) -> int:
        return max(0, self.effective_max_verifications - self.verification_attempts)

    @property
    def remaining_retries(self) -> int:
        return max(0, self.effective_max_retries - self.retries_attempted)

    @property
    def any_budget_exhausted(self) -> bool:
        return (
            self.token_budget_exhausted
            or self.tool_budget_exhausted
            or self.verification_budget_exhausted
            or self.retry_budget_exhausted
        )

    def can_generate(self, num_tokens: int = 1) -> bool:
        if self.tokens_generated + num_tokens > self.effective_max_tokens:
            self.token_budget_exhausted = True
            return False
        return True

    def record_tokens(self, count: int) -> None:
        try:
            count_val = int(count)
        except (TypeError, ValueError):
            count_val = 0
        self.tokens_generated += count_val
        if self.tokens_generated >= self.effective_max_tokens:
            self.token_budget_exhausted = True

    def can_call_tool(self) -> bool:
        if self.tool_calls_count >= self.effective_max_tool_calls:
            self.tool_budget_exhausted = True
            return False
        return True

    def record_tool_call(self) -> None:
        self.tool_calls_count += 1
        if self.tool_calls_count >= self.effective_max_tool_calls:
            self.tool_budget_exhausted = True

    def can_verify(self) -> bool:
        if self.verification_attempts >= self.effective_max_verifications:
            self.verification_budget_exhausted = True
            return False
        return True

    def record_verification(self) -> None:
        self.verification_attempts += 1
        if self.verification_attempts >= self.effective_max_verifications:
            self.verification_budget_exhausted = True

    def can_retry(self) -> bool:
        if self.retries_attempted >= self.effective_max_retries:
            self.retry_budget_exhausted = True
            return False
        return True

    def record_retry(self) -> None:
        self.retries_attempted += 1
        if self.retries_attempted >= self.effective_max_retries:
            self.retry_budget_exhausted = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "effective_budgets": {
                "max_tokens": self.effective_max_tokens,
                "max_tool_calls": self.effective_max_tool_calls,
                "max_verifications": self.effective_max_verifications,
                "max_retries": self.effective_max_retries,
            },
            "consumption": {
                "tokens_generated": self.tokens_generated,
                "tool_calls_count": self.tool_calls_count,
                "verification_attempts": self.verification_attempts,
                "retries_attempted": self.retries_attempted,
            },
            "exhausted": {
                "token_budget_exhausted": self.token_budget_exhausted,
                "tool_budget_exhausted": self.tool_budget_exhausted,
                "verification_budget_exhausted": self.verification_budget_exhausted,
                "retry_budget_exhausted": self.retry_budget_exhausted,
                "any_exhausted": self.any_budget_exhausted,
            },
        }

    @classmethod
    def from_budget(cls, budget: InferenceBudget) -> BudgetDiagnostics:
        return cls(
            profile=budget.profile.value if budget.profile else None,
            effective_max_tokens=budget.max_new_tokens,
            effective_max_tool_calls=budget.max_tool_calls,
            effective_max_verifications=budget.max_verification_attempts,
            effective_max_retries=budget.max_retries,
        )
