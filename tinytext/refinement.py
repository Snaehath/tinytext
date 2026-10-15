"""Controlled Test-Time Refinement Framework for TinyText Flash.

Supports three reproducible inference strategies:
- Strategy A: One-Pass Baseline (Hardened orchestrator with zero refinement).
- Strategy B: Generic Self-Critique & Revision (Draft -> Generic Critique -> Revision).
- Strategy C: Constraint-Guided Refinement (Task checks -> Targeted feedback -> Gated revision).

Key Architectural Guarantees:
1. Zero code execution (no eval, no exec, no shell commands).
2. Deterministic task bypass: Arithmetic and Source-Grounded Extraction are never overridden by LM revision.
3. Strict Gated Selection: Strategy C only adopts a revised candidate if it measurably improves
   constraint compliance without regressing on previously satisfied checks.
4. Privacy: Intermediate critiques/drafts are never exposed in final output.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch

from tinytext.model import TinyTextForCausalLM
from tinytext.tokenizer import TinyTextTokenizer
from tinytext.generate import generate_text
from tinytext.orchestrator import (
    ReasoningOrchestrator,
    Route,
    VerificationStatus,
    OrchestrationResult,
    TaskVerifier,
    SafeArithmeticEvaluator,
    SourceGroundedExtractor,
    LocalFactualRetriever,
)


class RefinementStrategy(str, Enum):
    NONE = "none"  # Strategy A: One-pass baseline
    GENERIC_CRITIQUE = "generic_critique"  # Strategy B: Generic self-critique & revision
    CONSTRAINT_GUIDED = "constraint_guided"  # Strategy C: Constraint-guided selective revision


@dataclass
class RefinementResult:
    final_output: str
    strategy: RefinementStrategy
    route: Route
    status: VerificationStatus
    verified: bool
    task_success: bool = False
    draft_output: str = ""
    critique_or_feedback: Optional[str] = None
    revision_output: Optional[str] = None
    was_revised: bool = False
    revision_accepted: bool = False
    checks_performed: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    validation_detail: str = ""
    tokens_draft: int = 0
    tokens_critique: int = 0
    tokens_revision: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    tool_used: Optional[str] = None

    @property
    def output(self) -> str:
        return self.final_output


class RefinementPipeline:
    """Orchestrates test-time refinement strategies around TinyText language models."""

    def __init__(
        self,
        orchestrator: ReasoningOrchestrator,
        model: Optional[TinyTextForCausalLM] = None,
        tokenizer: Optional[TinyTextTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.orchestrator = orchestrator
        self.model = model or orchestrator.model
        self.tokenizer = tokenizer or orchestrator.tokenizer
        self.device = device or orchestrator.device

    def process(
        self,
        user_query: str,
        strategy: RefinementStrategy = RefinementStrategy.NONE,
        max_new_tokens: int = 60,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> RefinementResult:
        start_time = time.time()
        route = self.orchestrator.route_query(user_query)

        # Deterministic tasks (Arithmetic, Extraction, Factual Retrieval) MUST bypass refinement.
        # Refinement is never allowed to override a verified deterministic tool result.
        if route in (Route.ARITHMETIC, Route.EXTRACTION, Route.FACTUAL_RETRIEVAL):
            orch_res = self.orchestrator.process(
                user_query=user_query,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            elapsed = time.time() - start_time
            return RefinementResult(
                final_output=orch_res.output,
                strategy=strategy,
                route=route,
                status=orch_res.status,
                verified=orch_res.verified,
                task_success=orch_res.task_success,
                draft_output=orch_res.output,
                critique_or_feedback=None,
                revision_output=None,
                was_revised=False,
                revision_accepted=False,
                checks_performed=orch_res.checks_performed,
                rejection_reason=orch_res.rejection_reason,
                validation_detail=orch_res.validation_detail,
                tokens_draft=orch_res.tokens_generated,
                tokens_critique=0,
                tokens_revision=0,
                total_tokens=orch_res.tokens_generated,
                latency_seconds=elapsed,
                tool_used=orch_res.tool_used,
            )

        # Allow execution if _generate is customized/mocked or model is loaded
        is_mocked = getattr(self, "_is_mocked", False) or hasattr(self, "_generate_mocked")
        if not is_mocked and (self.model is None or self.tokenizer is None):
            elapsed = time.time() - start_time
            return RefinementResult(
                final_output="Model or tokenizer not initialized.",
                strategy=strategy,
                route=route,
                status=VerificationStatus.ERROR,
                verified=False,
                task_success=False,
                checks_performed=["model_availability_check"],
                rejection_reason="Missing model or tokenizer.",
                latency_seconds=elapsed,
            )

        # Strategy A: One-pass baseline (no refinement)
        if strategy == RefinementStrategy.NONE:
            return self._run_strategy_a(
                user_query=user_query,
                route=route,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                start_time=start_time,
            )

        # Strategy B: Generic self-critique and revision
        elif strategy == RefinementStrategy.GENERIC_CRITIQUE:
            return self._run_strategy_b(
                user_query=user_query,
                route=route,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                start_time=start_time,
            )

        # Strategy C: Constraint-guided selective refinement
        elif strategy == RefinementStrategy.CONSTRAINT_GUIDED:
            return self._run_strategy_c(
                user_query=user_query,
                route=route,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                start_time=start_time,
            )

        else:
            raise ValueError(f"Unknown refinement strategy: {strategy}")

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        return generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            device=self.device,
        )

    def _run_strategy_a(
        self,
        user_query: str,
        route: Route,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        start_time: float,
    ) -> RefinementResult:
        full_prompt = (
            "System:\n"
            "You are TinyText, a helpful and concise assistant. Follow the user's instructions, answer accurately when possible, and acknowledge uncertainty when needed.\n\n"
            f"User:\n{user_query}\n\n"
            "Assistant:\n"
        )
        gen = self._generate(full_prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        candidate = gen["continuation"].strip()
        tokens = gen["tokens_generated"]

        status, detail = self.orchestrator._verify_candidate(route, user_query, candidate)
        checks = self.orchestrator._get_checks_for_route(route)
        elapsed = time.time() - start_time

        return RefinementResult(
            final_output=candidate,
            strategy=RefinementStrategy.NONE,
            route=route,
            status=status,
            verified=(status == VerificationStatus.VERIFIED),
            task_success=(status == VerificationStatus.VERIFIED),
            draft_output=candidate,
            critique_or_feedback=None,
            revision_output=None,
            was_revised=False,
            revision_accepted=False,
            checks_performed=checks,
            rejection_reason=detail if status == VerificationStatus.REJECTED else None,
            validation_detail=detail,
            tokens_draft=tokens,
            tokens_critique=0,
            tokens_revision=0,
            total_tokens=tokens,
            latency_seconds=elapsed,
            tool_used=None,
        )

    def _run_strategy_b(
        self,
        user_query: str,
        route: Route,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        start_time: float,
    ) -> RefinementResult:
        # Step 1: Initial draft
        draft_prompt = (
            "System:\n"
            "You are TinyText, a helpful and concise assistant. Follow the user's instructions, answer accurately when possible, and acknowledge uncertainty when needed.\n\n"
            f"User:\n{user_query}\n\n"
            "Assistant:\n"
        )
        draft_gen = self._generate(draft_prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        draft_text = draft_gen["continuation"].strip()
        tokens_draft = draft_gen["tokens_generated"]

        # Step 2: Generic Self-Critique
        critique_prompt = (
            "System:\n"
            "You are a critical reviewer. Evaluate the candidate response to the user's request. Point out any errors, repetitions, missing constraints, or formatting flaws concisely.\n\n"
            f"User Request:\n{user_query}\n\n"
            f"Candidate Response:\n{draft_text}\n\n"
            "Critique:\n"
        )
        critique_gen = self._generate(critique_prompt, max_new_tokens=40, temperature=0.3, top_p=0.9)
        critique_text = critique_gen["continuation"].strip()
        tokens_critique = critique_gen["tokens_generated"]

        # Step 3: Revision based on critique
        revision_prompt = (
            "System:\n"
            "You are TinyText, a helpful and concise assistant. Follow the user's instructions.\n\n"
            f"User Request:\n{user_query}\n\n"
            f"Critique of Previous Draft:\n{critique_text}\n\n"
            "Provide the revised final response strictly addressing the critique without repeating errors or adding preamble:\n\n"
            "Assistant:\n"
        )
        revision_gen = self._generate(revision_prompt, max_new_tokens=max_new_tokens, temperature=0.5, top_p=0.9)
        revision_text = revision_gen["continuation"].strip()
        tokens_revision = revision_gen["tokens_generated"]

        total_tokens = tokens_draft + tokens_critique + tokens_revision
        elapsed = time.time() - start_time

        # Validate final revised output
        status, detail = self.orchestrator._verify_candidate(route, user_query, revision_text)
        checks = self.orchestrator._get_checks_for_route(route)

        # Do NOT expose intermediate text (critique or draft) in final_output
        return RefinementResult(
            final_output=revision_text,
            strategy=RefinementStrategy.GENERIC_CRITIQUE,
            route=route,
            status=status,
            verified=(status == VerificationStatus.VERIFIED),
            task_success=(status == VerificationStatus.VERIFIED),
            draft_output=draft_text,
            critique_or_feedback=critique_text,
            revision_output=revision_text,
            was_revised=True,
            revision_accepted=True,  # In Strategy B, revision unconditionally replaces draft
            checks_performed=checks,
            rejection_reason=detail if status == VerificationStatus.REJECTED else None,
            validation_detail=f"Strategy B Revision: {detail}",
            tokens_draft=tokens_draft,
            tokens_critique=tokens_critique,
            tokens_revision=tokens_revision,
            total_tokens=total_tokens,
            latency_seconds=elapsed,
            tool_used=None,
        )

    def _run_strategy_c(
        self,
        user_query: str,
        route: Route,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        start_time: float,
    ) -> RefinementResult:
        # Step 1: Initial draft
        draft_prompt = (
            "System:\n"
            "You are TinyText, a helpful and concise assistant. Follow the user's instructions, answer accurately when possible, and acknowledge uncertainty when needed.\n\n"
            f"User:\n{user_query}\n\n"
            "Assistant:\n"
        )
        draft_gen = self._generate(draft_prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        draft_text = draft_gen["continuation"].strip()
        tokens_draft = draft_gen["tokens_generated"]

        # Step 2: Task-specific validation on initial draft
        draft_status, draft_detail = self.orchestrator._verify_candidate(route, user_query, draft_text)
        draft_score = self._compute_task_score(route, user_query, draft_text, draft_status)
        checks = self.orchestrator._get_checks_for_route(route)

        # If draft is verified or has no concrete rejection, bypass revision to save cost/latency
        if draft_status != VerificationStatus.REJECTED and draft_score >= 10:
            elapsed = time.time() - start_time
            return RefinementResult(
                final_output=draft_text,
                strategy=RefinementStrategy.CONSTRAINT_GUIDED,
                route=route,
                status=draft_status,
                verified=(draft_status == VerificationStatus.VERIFIED),
                task_success=(draft_status == VerificationStatus.VERIFIED),
                draft_output=draft_text,
                critique_or_feedback=None,
                revision_output=None,
                was_revised=False,
                revision_accepted=False,
                checks_performed=checks,
                rejection_reason=None,
                validation_detail=f"Draft satisfied criteria without revision: {draft_detail}",
                tokens_draft=tokens_draft,
                tokens_critique=0,
                tokens_revision=0,
                total_tokens=tokens_draft,
                latency_seconds=elapsed,
                tool_used=None,
            )

        # Step 3: Formulate targeted corrective feedback
        targeted_feedback = f"Your previous response had the following issue: {draft_detail}"
        revision_prompt = (
            f"System:\n"
            f"You are TinyText, a helpful and concise assistant.\n\n"
            f"User:\n{user_query}\n\n"
            f"Note: Ensure your answer corrects the previous issue: {draft_detail}. Answer directly without preamble:\n\n"
            f"Assistant:\n"
        )

        # Step 4: Generate at most one revised candidate with lower temperature for compliance
        revision_gen = self._generate(revision_prompt, max_new_tokens=max_new_tokens, temperature=0.2, top_p=0.9)
        revision_text = revision_gen["continuation"].strip()
        tokens_revision = revision_gen["tokens_generated"]
        total_tokens = tokens_draft + tokens_revision

        # Step 5: Validate revised candidate
        rev_status, rev_detail = self.orchestrator._verify_candidate(route, user_query, revision_text)
        rev_score = self._compute_task_score(route, user_query, revision_text, rev_status)

        # Step 6: Strict Gated Selection Rule
        # Accept the revised candidate ONLY if it improves compliance without breaking existing criteria
        revision_improves = (
            (rev_status == VerificationStatus.VERIFIED and draft_status != VerificationStatus.VERIFIED)
            or (rev_status != VerificationStatus.REJECTED and draft_status == VerificationStatus.REJECTED)
            or (rev_score > draft_score and not TaskVerifier.check_repetition(revision_text))
        )

        elapsed = time.time() - start_time

        if revision_improves:
            # Gated acceptance: Revised candidate is adopted
            return RefinementResult(
                final_output=revision_text,
                strategy=RefinementStrategy.CONSTRAINT_GUIDED,
                route=route,
                status=rev_status,
                verified=(rev_status == VerificationStatus.VERIFIED),
                task_success=(rev_status == VerificationStatus.VERIFIED),
                draft_output=draft_text,
                critique_or_feedback=targeted_feedback,
                revision_output=revision_text,
                was_revised=True,
                revision_accepted=True,
                checks_performed=checks,
                rejection_reason=rev_detail if rev_status == VerificationStatus.REJECTED else None,
                validation_detail=f"Adopted revision (score: {draft_score} -> {rev_score}): {rev_detail}",
                tokens_draft=tokens_draft,
                tokens_critique=0,
                tokens_revision=tokens_revision,
                total_tokens=total_tokens,
                latency_seconds=elapsed,
                tool_used=None,
            )
        else:
            # Gated rejection: Revision failed to improve upon draft; retain draft to prevent regression
            return RefinementResult(
                final_output=draft_text,
                strategy=RefinementStrategy.CONSTRAINT_GUIDED,
                route=route,
                status=draft_status,
                verified=(draft_status == VerificationStatus.VERIFIED),
                task_success=(draft_status == VerificationStatus.VERIFIED),
                draft_output=draft_text,
                critique_or_feedback=targeted_feedback,
                revision_output=revision_text,
                was_revised=True,
                revision_accepted=False,
                checks_performed=checks,
                rejection_reason=draft_detail if draft_status == VerificationStatus.REJECTED else None,
                validation_detail=f"Retained draft (revision score {rev_score} <= draft score {draft_score}): {draft_detail}",
                tokens_draft=tokens_draft,
                tokens_critique=0,
                tokens_revision=tokens_revision,
                total_tokens=total_tokens,
                latency_seconds=elapsed,
                tool_used=None,
            )

    def _compute_task_score(self, route: Route, query: str, text: str, status: VerificationStatus) -> int:
        """Computes a heuristic quality score for gated candidate comparison."""
        score = 0
        if status == VerificationStatus.VERIFIED:
            score += 20
        elif status == VerificationStatus.UNVERIFIED:
            score += 10
        elif status == VerificationStatus.REJECTED:
            score -= 10

        # Repetition penalty
        if TaskVerifier.check_repetition(text):
            score -= 15

        # Assistant disclaimer penalty
        if any(d in text.lower() for d in TaskVerifier.DISCLAIMERS):
            score -= 10

        if route == Route.CONSTRAINED_LIST:
            matches = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s*(.+)", text.strip())
            if matches:
                score += 5
                # Exact count check
                count_match = re.search(r"(?:exactly|give|list|provide|name)\s+(\d+|two|three|four|five)", query.lower())
                word_to_num = {"two": 2, "three": 3, "four": 4, "five": 5}
                target = 3
                if count_match:
                    rc = count_match.group(1)
                    target = word_to_num.get(rc, int(rc) if rc.isdigit() else 3)
                if len(matches) == target:
                    score += 10
                # Unique items
                items = [m[1].strip().lower() for m in matches]
                if len(set(items)) == len(items):
                    score += 5
                # No placeholder
                if not any(it in TaskVerifier.PLACEHOLDER_ITEMS for it in items):
                    score += 5

        elif route == Route.REWRITING:
            parts = query.split(":", 1)
            source = parts[1] if len(parts) > 1 else query
            # Entity preservation bonus
            source_entities = re.findall(r"\b[A-Z][a-z0-9]+\b|\b\d+\b", source)
            source_entities = [e for e in source_entities if e.lower() not in {"the", "a", "an", "this", "in", "it"}]
            if source_entities:
                preserved = sum(1 for e in source_entities if e.lower() in text.lower())
                score += preserved * 3

            # Conciseness bonus if requested
            if "concise" in query.lower() or "simplify" in query.lower():
                if len(text.split()) < len(source.split()):
                    score += 8

        return score
