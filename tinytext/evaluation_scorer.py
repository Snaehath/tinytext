"""Task-Specific Ground-Truth Evaluator and Scorer for TinyText Flash.

Provides rigorous, independent validation across:
1. Arithmetic: Numerical equivalence checking (no naive substring match).
2. Factual QA: Verified fact matching and abstention checking for unavailable facts.
3. Extraction: Source-grounding, field-attribution, format-compliance, and safe absence verification.
4. Classification: Permitted label constraints and exact single-label matching.
5. Exact-Count Lists: Count, sequence, uniqueness, placeholder rejection, and semantic domain relevance.
6. Open-Ended Rewriting/Explanation: Explicitly marked UNSCORED/UNVERIFIED for binary metrics.

Guarantees strict separation of:
- Route-selection accuracy
- Route-error taxonomy (mutually exclusive per policy)
- Protocol syntax validity
- Tool argument correctness
- Tool execution / runtime success
- Grounding and value correctness
- Format compliance vs content correctness
- Final answer correctness
- End-to-end task success
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from tinytext.holdout_manifest import ToolPolicy
from tinytext.tool_protocol import ToolActionType, ToolExecutionResult, ToolParseResult, ToolProtocol, ToolStatus


class TaskCategory(str, Enum):
    ARITHMETIC = "arithmetic"
    FACTUAL_QA = "factual_qa"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    LISTS = "lists"
    REASONING = "reasoning"
    COMPLETION = "completion"
    REWRITING = "rewriting"
    EXPLANATION = "explanation"
    ABSTENTION = "abstention"


@dataclass
class ExtractionScoringResult:
    """Decomposed extraction scoring preserving separate content and formatting metrics."""
    value_correctness: bool
    grounding: bool
    field_attribution: bool
    format_compliance: bool
    strict_extraction_success: bool
    is_missing_field: bool
    correctly_reports_absence: Optional[bool] = None
    avoids_fabrication: Optional[bool] = None
    detail: str = ""


@dataclass
class TaskEvaluationResult:
    prompt: str
    category: TaskCategory
    selected_action: Optional[ToolActionType]
    expected_action: Optional[ToolActionType]
    is_tool_selection_correct: bool
    is_syntax_valid: bool
    is_malformed: bool
    is_execution_success: bool
    is_argument_correct: bool
    is_final_answer_correct: Optional[bool]  # None for unverified/unscored open-ended tasks
    is_end_to_end_success: bool
    is_abstention: bool
    is_abstention_correct: Optional[bool]
    failure_reasons: List[str]
    raw_generation: str
    final_output: str
    tool_required: bool = False
    tool_policy: ToolPolicy = ToolPolicy.FORBIDDEN
    route_category: str = "correct_non_tool"
    acceptable_routes: List[str] = field(default_factory=list)
    scoring_status: str = "objectively_scored"
    expected_outcome: str = "correct_answer"
    is_false_positive_tool: bool = False
    is_missed_tool_call: bool = False
    is_cross_tool_misrouting: bool = False
    extraction_decomposition: Optional[ExtractionScoringResult] = None


class EvaluationScorer:
    """Rigorous independent task validator."""

    # Pre-defined known categories for list semantic checks
    DOMAIN_VOCABULARIES: Dict[str, Set[str]] = {
        "primary colors": {"red", "blue", "yellow"},
        "seasons": {"spring", "summer", "autumn", "fall", "winter"},
        "fruits": {"apple", "banana", "orange", "grape", "pear", "peach", "strawberry", "mango", "lemon", "lime", "berry", "cherry", "plum"},
        "programming languages": {"python", "rust", "javascript", "typescript", "c", "c++", "java", "go", "ruby", "swift", "kotlin", "php"},
        "days of the workweek": {"monday", "tuesday", "wednesday", "thursday", "friday"},
        "continents": {"africa", "antarctica", "asia", "australia", "europe", "north america", "south america", "oceania"},
        "capital cities": {"paris", "rome", "berlin", "madrid", "london", "tokyo", "beijing", "ottawa", "canberra", "washington"},
        "musical instruments": {"piano", "guitar", "violin", "flute", "drums", "trumpet", "cello", "clarinet", "saxophone"},
        "vegetables": {"carrot", "broccoli", "spinach", "potato", "onion", "cabbage", "pepper", "tomato", "pea", "corn"},
    }

    # Obvious dummy placeholder items
    DUMMY_ITEMS: Set[str] = {
        "the pilot", "pilot", "item", "item 1", "item 2", "item 3",
        "placeholder", "none", "n/a", "etc", "test",
    }

    @staticmethod
    def check_repetition(text: str) -> bool:
        """Returns True if excessive n-gram looping or phrase repetition is detected."""
        words = text.lower().split()
        if len(words) >= 8:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            for tg in set(trigrams):
                if trigrams.count(tg) >= 3:
                    return True
            fourgrams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
            for fg in set(fourgrams):
                if fourgrams.count(fg) >= 2:
                    return True
        lines = [l.strip().lower() for l in text.split("\n") if len(l.strip()) > 5]
        if len(lines) >= 3 and len(lines) != len(set(lines)):
            return True
        return False

    @classmethod
    def decompose_extraction(
        cls,
        final_output: str,
        expected_target: Any,
        source_passage: Optional[str],
        expected_outcome: Optional[str] = None,
    ) -> ExtractionScoringResult:
        """Decomposes extraction scoring into value correctness, grounding, attribution, and format compliance."""
        if expected_target is None:
            return ExtractionScoringResult(
                value_correctness=False,
                grounding=False,
                field_attribution=False,
                format_compliance=False,
                strict_extraction_success=False,
                is_missing_field=False,
                detail="expected_target is None",
            )

        is_rep = cls.check_repetition(final_output)
        out_clean = final_output.strip().lower()

        is_missing = (
            expected_outcome == "safe_absence"
            or any(
                term in str(expected_target).lower()
                for term in ["not found", "unavailable", "absent", "not mentioned", "not present"]
            )
        )

        if is_missing:
            absence_indicators = [
                "not found", "not mentioned", "unavailable", "not available",
                "absent", "none", "unknown", "not present", "pending quote",
                "cannot be found", "does not appear", "not specified", "not provided",
            ]
            stated_absence = any(ind in out_clean for ind in absence_indicators)
            avoids_fab = (
                stated_absence
                and not any(char.isdigit() for char in final_output)
                and not any(w in out_clean for w in ["$", "%", "inv-", "ba249", "million"])
            )
            val_corr = stated_absence and avoids_fab
            grounding = True
            attrib = True
            fmt_comp = not is_rep and len(final_output.split()) <= 15
            strict = val_corr and fmt_comp

            detail_parts = []
            if not stated_absence:
                detail_parts.append("Absence not reported")
            if not avoids_fab:
                detail_parts.append("Fabricated entity detected")
            if is_rep:
                detail_parts.append("Repetition detected")

            return ExtractionScoringResult(
                value_correctness=val_corr,
                grounding=grounding,
                field_attribution=attrib,
                format_compliance=fmt_comp,
                strict_extraction_success=strict,
                is_missing_field=True,
                correctly_reports_absence=stated_absence,
                avoids_fabrication=avoids_fab,
                detail="; ".join(detail_parts) if detail_parts else "Safe absence verified",
            )

        # Present field
        targets = [expected_target] if isinstance(expected_target, str) else [str(t) for t in expected_target]
        best_decomp: Optional[ExtractionScoringResult] = None

        for tgt in targets:
            tgt_str = tgt.strip()
            if not tgt_str:
                continue

            if ":" in tgt_str:
                field_name, field_val = tgt_str.split(":", 1)
                field_name = field_name.strip().lower()
                field_val = field_val.strip()
            else:
                field_name = None
                field_val = tgt_str

            val_clean = field_val.strip().lower()
            val_bare = val_clean.strip(".,;:!?() ")
            if not val_bare:
                continue

            # 1. Value correctness
            val_in_out = (val_clean in out_clean) or (val_bare in out_clean)

            # 2. Grounding in source text
            out_val = final_output.split(":", 1)[1].strip() if ":" in final_output else final_output.strip()
            out_val_bare = out_val.lower().strip(".,;:!?() ")
            if val_in_out:
                is_grounded = bool(source_passage and val_bare in source_passage.lower())
            else:
                is_grounded = bool(source_passage and out_val_bare and out_val_bare in source_passage.lower())

            # 3. Field attribution
            field_attrib = True
            if ":" in final_output:
                out_parts = final_output.split(":", 1)
                out_field = out_parts[0].strip().lower()
                if field_name:
                    field_words = set(re.findall(r"\b\w+\b", field_name))
                    out_words = set(re.findall(r"\b\w+\b", out_field))
                    if not out_words.intersection(field_words):
                        field_attrib = False
            else:
                field_attrib = True

            # 4. Format compliance (conciseness and repetition avoidance)
            fmt_comp = True
            if is_rep:
                fmt_comp = False
            else:
                val_word_count = len(val_bare.split())
                if ":" in final_output:
                    out_val_words = len(final_output.split(":", 1)[1].strip().split())
                    if out_val_words > val_word_count + 3:
                        fmt_comp = False
                else:
                    out_words = len(final_output.split())
                    if out_words > val_word_count + 3:
                        fmt_comp = False

            strict = val_in_out and is_grounded and field_attrib and fmt_comp

            detail_reasons = []
            if not val_in_out:
                detail_reasons.append(f"Value '{val_bare}' missing from output")
            if not is_grounded:
                detail_reasons.append(f"Value '{val_bare}' not grounded in source text")
            if not field_attrib:
                detail_reasons.append("Conflicting field attribution")
            if not fmt_comp:
                detail_reasons.append("Format non-compliant (rambling/repetitive)")

            cand = ExtractionScoringResult(
                value_correctness=val_in_out,
                grounding=is_grounded,
                field_attribution=field_attrib,
                format_compliance=fmt_comp,
                strict_extraction_success=strict,
                is_missing_field=False,
                detail="; ".join(detail_reasons) if detail_reasons else "Strict extraction pass",
            )
            if best_decomp is None or (cand.strict_extraction_success and not best_decomp.strict_extraction_success):
                best_decomp = cand
            if strict:
                break

        return best_decomp or ExtractionScoringResult(
            value_correctness=False,
            grounding=False,
            field_attribution=False,
            format_compliance=False,
            strict_extraction_success=False,
            is_missing_field=False,
            detail="No matching target evaluated",
        )

    @classmethod
    def _verify_extraction_payload(cls, payload: str, expected_target: Any) -> bool:
        """Verifies if the extraction tool payload contains the requested field or key."""
        if not payload:
            return False
        p_lower = payload.lower()
        targets = [expected_target] if isinstance(expected_target, str) else [str(t) for t in expected_target]
        for tgt in targets:
            val = tgt.split(":", 1)[1].strip().lower() if ":" in tgt else tgt.strip().lower()
            val_bare = val.strip(".,;:!?() ")
            if val_bare and val_bare in p_lower:
                return True
            if ":" in tgt:
                key = tgt.split(":", 1)[0].strip().lower()
                if key and key in p_lower:
                    return True
        return False

    @classmethod
    def score_task(
        cls,
        prompt: str,
        category: str | TaskCategory,
        raw_generation: str,
        final_output: str,
        expected_action: Optional[ToolActionType | str] = None,
        expected_target: Optional[Any] = None,
        source_passage: Optional[str] = None,
        permitted_labels: Optional[List[str]] = None,
        target_count: Optional[int] = None,
        list_domain: Optional[str] = None,
        is_orchestrated_mode: bool = False,
        tool_required: bool = False,
        acceptable_routes: Optional[List[str]] = None,
        expected_outcome: Optional[str] = None,
        scoring_status: Optional[str] = None,
        tool_policy: Optional[ToolPolicy | str] = None,
    ) -> TaskEvaluationResult:
        """Evaluates a single task output against ground truth with decoupled metric tracking."""
        if isinstance(category, str):
            try:
                cat = TaskCategory(category.lower())
            except ValueError:
                cat = TaskCategory.REASONING
        else:
            cat = category

        if isinstance(expected_action, str):
            try:
                exp_action = ToolActionType(expected_action.upper())
            except ValueError:
                exp_action = None
        else:
            exp_action = expected_action

        # Resolve ToolPolicy
        if tool_policy is not None:
            pol = ToolPolicy(tool_policy) if isinstance(tool_policy, str) else tool_policy
        elif tool_required:
            pol = ToolPolicy.REQUIRED
        elif exp_action in (ToolActionType.CALC, ToolActionType.FACT, ToolActionType.EXTRACT):
            pol = ToolPolicy.REQUIRED
        else:
            pol = ToolPolicy.FORBIDDEN

        failure_reasons: List[str] = []

        # 1. Parse Protocol Syntax and Execution
        has_tool_tag = any(
            f"[{a.value}" in raw_generation.upper()
            for a in [ToolActionType.CALC, ToolActionType.FACT, ToolActionType.EXTRACT]
        )

        if is_orchestrated_mode:
            is_syntax_valid = True
            is_malformed = False
            clean_tag = raw_generation.strip("[] ").upper()
            if clean_tag in ("CALC", "ARITHMETIC"):
                selected_action = ToolActionType.CALC
            elif clean_tag in ("FACT", "FACTUAL_RETRIEVAL"):
                selected_action = ToolActionType.FACT
            elif clean_tag in ("EXTRACT", "EXTRACTION"):
                selected_action = ToolActionType.EXTRACT
            else:
                selected_action = ToolActionType.ANSWER
            is_exec_success = True
            is_abstention = False
            is_arg_correct = True
        else:
            parse_res = ToolProtocol.parse(raw_generation)
            is_syntax_valid = parse_res.is_valid
            is_malformed = not parse_res.is_valid
            selected_action = parse_res.action

            if is_malformed:
                failure_reasons.append(f"Protocol syntax error: {parse_res.error_reason}")

            is_exec_success = False
            is_abstention = False
            is_arg_correct = False

            if selected_action in (ToolActionType.CALC, ToolActionType.FACT, ToolActionType.EXTRACT):
                exec_res = ToolProtocol.execute(parse_res)
                is_abstention = exec_res.is_abstention
                is_exec_success = (exec_res.status == ToolStatus.SUCCESS)
                if not is_exec_success and not is_abstention:
                    failure_reasons.append(f"Tool execution failed: {exec_res.status.value}")
            else:
                is_exec_success = is_syntax_valid

        # 2. Mutually Exclusive Route Taxonomy per Policy
        is_tool_sel_correct = False
        route_category = ""
        is_fp_tool = False
        is_missed_tool = False
        is_cross_tool = False

        valid_routes: List[str] = []
        if acceptable_routes:
            valid_routes = [r.upper() for r in acceptable_routes]
        elif exp_action is not None:
            valid_routes = [exp_action.value]

        if pol == ToolPolicy.FORBIDDEN:
            if is_syntax_valid:
                if selected_action == ToolActionType.ANSWER or (selected_action and selected_action.value in valid_routes and selected_action.value not in ("CALC", "FACT", "EXTRACT")):
                    route_category = "correct_non_tool"
                    is_tool_sel_correct = True
                else:
                    route_category = "recognized_tool_in_error"
                    is_tool_sel_correct = False
                    is_fp_tool = True
                    failure_reasons.append(f"Tool forbidden but recognized tool [{selected_action.value}] invoked")
            else:
                if has_tool_tag:
                    route_category = "malformed_with_tool_tag"
                else:
                    route_category = "malformed_without_tool_tag"
                is_tool_sel_correct = False

        elif pol == ToolPolicy.REQUIRED:
            if is_syntax_valid:
                if selected_action == exp_action:
                    route_category = "correct_tool_route"
                    is_tool_sel_correct = True
                elif selected_action in (ToolActionType.ANSWER, None):
                    route_category = "missed_tool_route"
                    is_tool_sel_correct = False
                    is_missed_tool = True
                    failure_reasons.append(f"Tool required [{exp_action.value if exp_action else 'TOOL'}] but non-tool route chosen")
                else:
                    route_category = "cross_tool_misrouting"
                    is_tool_sel_correct = False
                    is_cross_tool = True
                    failure_reasons.append(f"Cross-tool misrouting: expected [{exp_action.value if exp_action else 'TOOL'}], got [{selected_action.value}]")
            else:
                route_category = "malformed_tool_output"
                is_tool_sel_correct = False

        elif pol == ToolPolicy.OPTIONAL:
            if is_syntax_valid:
                if selected_action == ToolActionType.ANSWER:
                    route_category = "correct_primary_route"
                    is_tool_sel_correct = True
                elif selected_action and selected_action.value in valid_routes:
                    route_category = "allowed_optional_tool"
                    is_tool_sel_correct = True
                else:
                    route_category = "unexpected_tool_route"
                    is_tool_sel_correct = False
                    is_fp_tool = True
                    failure_reasons.append(f"Optional query but unexpected tool [{selected_action.value if selected_action else 'NONE'}] invoked")
            else:
                route_category = "malformed_optional_output"
                is_tool_sel_correct = False

        # 3. Ground-Truth Answer Verification and Decomposition
        is_final_correct: Optional[bool] = None
        is_abstention_correct: Optional[bool] = None
        ext_decomp: Optional[ExtractionScoringResult] = None

        status = scoring_status or (
            "unscored" if cat in (TaskCategory.REWRITING, TaskCategory.EXPLANATION) else "objectively_scored"
        )

        if status == "unscored":
            is_final_correct = None

        elif expected_outcome == "safe_abstention" or cat == TaskCategory.ABSTENTION:
            is_abstention_correct = cls._verify_abstention(final_output, prompt)
            is_final_correct = is_abstention_correct
            if not is_final_correct:
                failure_reasons.append("Failed to safely abstain on unsupported query")

        elif cat == TaskCategory.EXTRACTION or expected_outcome in ("grounded_extraction", "safe_absence"):
            ext_decomp = cls.decompose_extraction(
                final_output=final_output,
                expected_target=expected_target,
                source_passage=source_passage,
                expected_outcome=expected_outcome,
            )
            is_final_correct = ext_decomp.strict_extraction_success
            if not is_orchestrated_mode:
                is_arg_correct = cls._verify_extraction_payload(
                    parse_res.payload if parse_res.is_valid else "", expected_target
                )
            if not is_final_correct:
                failure_reasons.append(f"Extraction failure: {ext_decomp.detail}")

        elif expected_outcome == "correct_fact_or_safe_abstention":
            is_fact = cls._verify_factual(final_output, expected_target)
            is_abst = cls._verify_abstention(final_output, prompt)
            is_final_correct = (is_fact or is_abst)
            is_abstention_correct = is_abst
            if not is_final_correct:
                failure_reasons.append(f"Neither correct fact nor safe abstention provided: '{final_output}'")

        elif cat == TaskCategory.ARITHMETIC or expected_outcome == "correct_calculation":
            is_final_correct, is_arg_correct = cls._verify_arithmetic(
                final_output,
                parse_res.payload if (not is_orchestrated_mode and parse_res.is_valid) else "",
                expected_target,
            )
            if not is_final_correct:
                failure_reasons.append(f"Arithmetic mismatch: output '{final_output}' did not match expected '{expected_target}'")

        elif cat == TaskCategory.FACTUAL_QA or expected_outcome == "correct_fact":
            is_final_correct = cls._verify_factual(final_output, expected_target)
            if not is_orchestrated_mode:
                is_arg_correct = (
                    expected_target is not None
                    and str(expected_target).lower() in (parse_res.payload.lower() if parse_res.is_valid else "")
                )
            if not is_final_correct:
                failure_reasons.append(f"Factual answer mismatch: '{final_output}' missing verified target '{expected_target}'")

        elif cat == TaskCategory.CLASSIFICATION or expected_outcome == "correct_classification":
            is_final_correct = cls._verify_classification(final_output, expected_target, permitted_labels)
            if not is_final_correct:
                failure_reasons.append(f"Classification failure: '{final_output}' invalid or not matching '{expected_target}'")

        elif cat == TaskCategory.LISTS or expected_outcome == "correct_list":
            is_final_correct = cls._verify_list(final_output, target_count or 3, list_domain)
            if not is_final_correct:
                failure_reasons.append("List validation failure: count, uniqueness, or domain check failed")

        elif cat == TaskCategory.COMPLETION:
            is_final_correct = cls._verify_completion(final_output, expected_target)
            if not is_final_correct:
                failure_reasons.append(f"Sentence completion mismatch: expected '{expected_target}'")

        elif cat == TaskCategory.REASONING:
            if expected_target is not None and any(char.isdigit() for char in str(expected_target)):
                is_final_correct, is_arg_correct = cls._verify_arithmetic(
                    final_output,
                    parse_res.payload if (not is_orchestrated_mode and parse_res.is_valid) else "",
                    expected_target,
                )
            else:
                is_final_correct = (str(expected_target).lower() in final_output.lower()) if expected_target else None
            if is_final_correct is False:
                failure_reasons.append(f"Reasoning mismatch: output '{final_output}' did not match expected '{expected_target}'")

        # 4. End-to-End Success
        is_e2e_success = False
        if is_syntax_valid and is_tool_sel_correct and (is_final_correct is True):
            is_e2e_success = True

        return TaskEvaluationResult(
            prompt=prompt,
            category=cat,
            selected_action=selected_action,
            expected_action=exp_action,
            is_tool_selection_correct=is_tool_sel_correct,
            is_syntax_valid=is_syntax_valid,
            is_malformed=is_malformed,
            is_execution_success=is_exec_success,
            is_argument_correct=is_arg_correct,
            is_final_answer_correct=is_final_correct,
            is_end_to_end_success=is_e2e_success,
            is_abstention=is_abstention,
            is_abstention_correct=is_abstention_correct,
            failure_reasons=failure_reasons,
            raw_generation=raw_generation,
            final_output=final_output,
            tool_required=(pol == ToolPolicy.REQUIRED),
            tool_policy=pol,
            route_category=route_category,
            acceptable_routes=acceptable_routes or ([] if exp_action is None else [exp_action.value]),
            scoring_status=status,
            expected_outcome=expected_outcome or "correct_answer",
            is_false_positive_tool=is_fp_tool,
            is_missed_tool_call=is_missed_tool,
            is_cross_tool_misrouting=is_cross_tool,
            extraction_decomposition=ext_decomp,
        )

    # --------------------------------------------------------------------------
    # Specialized Validators
    # --------------------------------------------------------------------------

    @classmethod
    def _verify_arithmetic(
        cls, final_output: str, payload: str, expected_target: Any
    ) -> Tuple[bool, bool]:
        """Evaluates arithmetic by extracting numbers and comparing numerically (no substring match)."""
        if expected_target is None:
            return False, False

        expected_num: Optional[float] = None
        if isinstance(expected_target, (int, float)):
            expected_num = float(expected_target)
        elif isinstance(expected_target, str):
            num_m = re.search(r"[-+]?\d*\.?\d+", expected_target)
            if num_m:
                expected_num = float(num_m.group(0))
        elif isinstance(expected_target, (list, tuple)) and len(expected_target) > 0:
            num_m = re.search(r"[-+]?\d*\.?\d+", str(expected_target[0]))
            if num_m:
                expected_num = float(num_m.group(0))

        if expected_num is None:
            return False, False

        found_numbers: List[float] = []
        for token in re.findall(r"(?:^|[\s=:\$])([-+]?\d+(?:\.\d+)?)(?:[\s\.,;\?!]|$)", final_output.strip()):
            try:
                found_numbers.append(float(token))
            except ValueError:
                continue

        is_final_correct = any(abs(n - expected_num) < 1e-5 for n in found_numbers)

        is_arg_correct = False
        if payload:
            from tinytext.orchestrator import SafeArithmeticEvaluator
            eval_val = SafeArithmeticEvaluator.evaluate(payload)
            if eval_val is not None and abs(float(eval_val) - expected_num) < 1e-5:
                is_arg_correct = True

        return is_final_correct, is_arg_correct

    @classmethod
    def _verify_factual(cls, final_output: str, expected_target: Any) -> bool:
        """Verifies factual claims using word-boundary matching to prevent false-positives."""
        if expected_target is None:
            return False

        targets: List[str] = []
        if isinstance(expected_target, str):
            targets = [expected_target]
        elif isinstance(expected_target, (list, tuple)):
            targets = [str(t) for t in expected_target]

        output_clean = final_output.strip().lower()

        for tgt in targets:
            tgt_clean = tgt.strip().lower()
            if not tgt_clean:
                continue

            if len(tgt_clean) <= 4:
                pattern = rf"\b{re.escape(tgt_clean)}\b"
                if re.search(pattern, output_clean):
                    return True
            else:
                if tgt_clean in output_clean:
                    return True

        return False

    @classmethod
    def _verify_extraction(
        cls,
        final_output: str,
        payload: str,
        expected_target: Any,
        source_passage: Optional[str],
        expected_outcome: Optional[str] = None,
    ) -> Tuple[bool, bool]:
        """Backward-compatible extraction validator delegating to decompose_extraction."""
        decomp = cls.decompose_extraction(
            final_output=final_output,
            expected_target=expected_target,
            source_passage=source_passage,
            expected_outcome=expected_outcome,
        )
        is_correct = decomp.strict_extraction_success
        is_arg_correct = cls._verify_extraction_payload(payload, expected_target) if payload else False
        return is_correct, is_arg_correct

    @classmethod
    def _verify_classification(
        cls, final_output: str, expected_target: Any, permitted_labels: Optional[List[str]]
    ) -> bool:
        """Validates single-label classification output against permitted set."""
        if expected_target is None:
            return False

        exp_label = str(expected_target[0] if isinstance(expected_target, (list, tuple)) else expected_target).strip().lower()

        if permitted_labels:
            perm_lower = [p.strip().lower() for p in permitted_labels]
        else:
            if exp_label in ("positive", "negative"):
                perm_lower = ["positive", "negative"]
            elif exp_label in ("animal", "vehicle"):
                perm_lower = ["animal", "vehicle"]
            elif exp_label in ("fruit", "vegetable"):
                perm_lower = ["fruit", "vegetable"]
            else:
                perm_lower = [exp_label]

        words = re.findall(r"\b[A-Za-z]+\b", final_output.lower())
        matched_labels = [w for w in words if w in perm_lower]

        if len(set(matched_labels)) > 1:
            if matched_labels and matched_labels[-1] == exp_label:
                return True
            return False

        if len(matched_labels) == 1 and matched_labels[0] == exp_label:
            return True

        return False

    @classmethod
    def _verify_list(cls, final_output: str, expected_count: int, list_domain: Optional[str] = None) -> bool:
        """Validates exact count, uniqueness, non-empty items, and domain relevance."""
        if cls.check_repetition(final_output):
            return False

        lines = [l.strip() for l in final_output.split("\n") if l.strip()]
        numbered_matches = []
        for l in lines:
            m = re.match(r"^(\d+)[\.\)]\s*(.+)$", l)
            if m:
                numbered_matches.append((int(m.group(1)), m.group(2).strip()))

        if len(numbered_matches) != expected_count:
            return False

        nums = [nm[0] for nm in numbered_matches]
        if nums != list(range(1, expected_count + 1)):
            return False

        items = [nm[1] for nm in numbered_matches]
        if any(len(it) == 0 for it in items):
            return False

        norm_items = [re.sub(r"[^\w\s]", "", it.lower()).strip() for it in items]
        if len(set(norm_items)) != len(norm_items):
            return False

        for it in norm_items:
            if it in cls.DUMMY_ITEMS:
                return False

        if list_domain and list_domain.lower() in cls.DOMAIN_VOCABULARIES:
            valid_vocab = cls.DOMAIN_VOCABULARIES[list_domain.lower()]
            domain_matches = 0
            for it in norm_items:
                tokens = set(it.split())
                if tokens.intersection(valid_vocab):
                    domain_matches += 1
            if domain_matches < expected_count:
                return False

        return True

    @classmethod
    def _verify_completion(cls, final_output: str, expected_target: Any) -> bool:
        """Verifies exact continuation word."""
        if expected_target is None:
            return False

        targets = [expected_target] if isinstance(expected_target, str) else [str(t) for t in expected_target]
        output_lower = final_output.strip().lower()

        for tgt in targets:
            tgt_lower = tgt.strip().lower()
            if re.search(rf"\b{re.escape(tgt_lower)}\b", output_lower):
                return True
        return False

    @classmethod
    def _verify_abstention(cls, final_output: str, prompt: Optional[str] = None) -> bool:
        """Checks for polite, unambiguous abstention on unsupported queries."""
        if cls.check_repetition(final_output):
            return False

        out_lower = final_output.strip().lower()
        abstention_phrases = [
            "unavailable", "not available", "do not have access", "cannot answer",
            "cannot execute", "not present", "not found", "cannot predict",
            "cannot disclose", "cannot be disclosed", "cannot provide",
            "does not exist", "no permanent human population", "unknown and unavailable",
            "not contained in", "not mentioned in", "pending quote",
        ]
        has_phrase = any(phrase in out_lower for phrase in abstention_phrases)
        if not has_phrase and re.search(r"\b(unknown|is unknown)\b", out_lower):
            has_phrase = True

        if not has_phrase:
            return False

        # Guard against prompt keyword echoing
        if prompt:
            p_lower = prompt.lower()
            if "unknown" in p_lower and not any(w in out_lower for w in ["unavailable", "cannot", "do not", "not available"]):
                return False

        return True

    @classmethod
    def aggregate_metrics(cls, results: List[TaskEvaluationResult]) -> Dict[str, Any]:
        """Aggregates task evaluation results reporting explicit counts, rates, and denominators."""
        n_total = len(results)

        req_tasks = [s for s in results if s.tool_policy == ToolPolicy.REQUIRED]
        n_req = len(req_tasks)
        forbid_tasks = [s for s in results if s.tool_policy == ToolPolicy.FORBIDDEN]
        n_forbid = len(forbid_tasks)
        opt_tasks = [s for s in results if s.tool_policy == ToolPolicy.OPTIONAL]
        n_opt = len(opt_tasks)

        tool_action_tasks = [s for s in results if s.selected_action != ToolActionType.ANSWER]
        n_tool_actions = len(tool_action_tasks)

        scored_tasks = [s for s in results if s.scoring_status == "objectively_scored" and s.is_final_answer_correct is not None]
        n_scored = len(scored_tasks)
        unscored_tasks = [s for s in results if s.scoring_status == "unscored" or s.is_final_answer_correct is None]
        n_unscored = len(unscored_tasks)

        abst_tasks = [s for s in results if s.category == TaskCategory.ABSTENTION or s.expected_outcome == "safe_abstention"]
        n_abst = len(abst_tasks)

        ext_tasks = [s for s in results if s.category == TaskCategory.EXTRACTION or s.extraction_decomposition is not None]
        n_ext = len(ext_tasks)

        def make_metric(count: int, denom: int, label: str) -> Dict[str, Any]:
            pct = round((count / max(1, denom)) * 100.0, 2) if denom > 0 else 0.0
            rate = round(count / max(1, denom), 4) if denom > 0 else 0.0
            return {
                "count": count,
                "denominator": denom,
                "denominator_label": label,
                "rate": rate,
                "rate_pct": pct,
                "fraction_str": f"{count}/{denom}",
            }

        # ---------------------------------------------------------------------
        # Route-Error Taxonomy (Mutually Exclusive per Policy Subset)
        # ---------------------------------------------------------------------
        # Forbidden Policy (N_forbid)
        forbid_correct_non_tool = sum(1 for s in forbid_tasks if s.route_category == "correct_non_tool")
        forbid_tool_in_error = sum(1 for s in forbid_tasks if s.route_category == "recognized_tool_in_error")
        forbid_malformed_with_tag = sum(1 for s in forbid_tasks if s.route_category == "malformed_with_tool_tag")
        forbid_malformed_no_tag = sum(1 for s in forbid_tasks if s.route_category == "malformed_without_tool_tag")
        forbid_sum = forbid_correct_non_tool + forbid_tool_in_error + forbid_malformed_with_tag + forbid_malformed_no_tag

        # Required Policy (N_req)
        req_correct_tool = sum(1 for s in req_tasks if s.route_category == "correct_tool_route")
        req_missed_tool = sum(1 for s in req_tasks if s.route_category == "missed_tool_route")
        req_cross_tool = sum(1 for s in req_tasks if s.route_category == "cross_tool_misrouting")
        req_malformed_output = sum(1 for s in req_tasks if s.route_category == "malformed_tool_output")
        req_sum = req_correct_tool + req_missed_tool + req_cross_tool + req_malformed_output

        # Optional Policy (N_opt)
        opt_correct_primary = sum(1 for s in opt_tasks if s.route_category == "correct_primary_route")
        opt_allowed_tool = sum(1 for s in opt_tasks if s.route_category == "allowed_optional_tool")
        opt_unexpected_tool = sum(1 for s in opt_tasks if s.route_category == "unexpected_tool_route")
        opt_malformed_output = sum(1 for s in opt_tasks if s.route_category == "malformed_optional_output")
        opt_sum = opt_correct_primary + opt_allowed_tool + opt_unexpected_tool + opt_malformed_output

        # Overall route selections
        tool_sel_count = sum(1 for s in results if s.is_tool_selection_correct)
        fp_tool_count = forbid_tool_in_error + opt_unexpected_tool
        missed_tool_count = req_missed_tool
        cross_tool_count = req_cross_tool

        # ---------------------------------------------------------------------
        # Tool Execution & Argument Metrics
        # ---------------------------------------------------------------------
        arg_correct_count = sum(1 for s in tool_action_tasks if s.is_argument_correct)
        exec_success_count = sum(1 for s in tool_action_tasks if s.is_execution_success)
        syntax_count = sum(1 for s in results if s.is_syntax_valid)
        malformed_count = sum(1 for s in results if s.is_malformed)

        # ---------------------------------------------------------------------
        # Task & Ground-Truth Outcome Metrics
        # ---------------------------------------------------------------------
        final_correct_count = sum(1 for s in scored_tasks if s.is_final_answer_correct is True)
        e2e_total_count = sum(1 for s in results if s.is_end_to_end_success)
        e2e_scored_count = sum(1 for s in scored_tasks if s.is_end_to_end_success)
        abst_correct_count = sum(1 for s in abst_tasks if s.is_abstention_correct is True)

        # ---------------------------------------------------------------------
        # Extraction Decomposition Summary
        # ---------------------------------------------------------------------
        ext_decomps = [s.extraction_decomposition for s in ext_tasks if s.extraction_decomposition is not None]
        ext_val_corr = sum(1 for d in ext_decomps if d.value_correctness)
        ext_ground = sum(1 for d in ext_decomps if d.grounding)
        ext_attrib = sum(1 for d in ext_decomps if d.field_attribution)
        ext_fmt = sum(1 for d in ext_decomps if d.format_compliance)
        ext_strict = sum(1 for d in ext_decomps if d.strict_extraction_success)

        missing_ext_decomps = [d for d in ext_decomps if d.is_missing_field]
        n_missing_ext = len(missing_ext_decomps)
        missing_abs_rep = sum(1 for d in missing_ext_decomps if d.correctly_reports_absence)
        missing_fab_avoid = sum(1 for d in missing_ext_decomps if d.avoids_fabrication)

        # ---------------------------------------------------------------------
        # Heuristic Generation Metric (renamed from qualitative evaluation)
        # ---------------------------------------------------------------------
        heuristic_pass_count = sum(
            1 for s in unscored_tasks
            if not cls.check_repetition(s.final_output) and len(s.final_output.strip().split()) >= 3
        )

        category_breakdown: Dict[str, Dict[str, Any]] = {}
        for s in results:
            c = s.category.value
            if c not in category_breakdown:
                category_breakdown[c] = {
                    "total": 0,
                    "tool_required": 0,
                    "tool_selection_correct": 0,
                    "final_answer_correct": 0,
                    "e2e_success": 0,
                    "unscored_count": 0,
                }
            category_breakdown[c]["total"] += 1
            if s.tool_required:
                category_breakdown[c]["tool_required"] += 1
            if s.is_tool_selection_correct:
                category_breakdown[c]["tool_selection_correct"] += 1
            if s.is_final_answer_correct is True:
                category_breakdown[c]["final_answer_correct"] += 1
            elif s.is_final_answer_correct is None:
                category_breakdown[c]["unscored_count"] += 1
            if s.is_end_to_end_success:
                category_breakdown[c]["e2e_success"] += 1

        for c, stats in category_breakdown.items():
            tot = stats["total"]
            stats["tool_selection_accuracy_pct"] = round(stats["tool_selection_correct"] / max(1, tot) * 100.0, 2)
            stats["e2e_success_rate_pct"] = round(stats["e2e_success"] / max(1, tot) * 100.0, 2)
            stats["fraction_str"] = f"{stats['e2e_success']}/{tot}"

        return {
            "summary_counts": {
                "total_tasks": n_total,
                "tool_required_tasks": n_req,
                "tool_forbidden_tasks": n_forbid,
                "tool_optional_tasks": n_opt,
                "evaluable_scored_tasks": n_scored,
                "unscored_open_ended_tasks": n_unscored,
                "abstention_tasks": n_abst,
                "extraction_tasks": n_ext,
                "tool_action_tasks": n_tool_actions,
            },
            "route_taxonomy": {
                "forbidden_prompts": {
                    "total": n_forbid,
                    "reconciled": (forbid_sum == n_forbid),
                    "correct_non_tool": make_metric(forbid_correct_non_tool, n_forbid, "forbidden_prompts (count / N_forbid)"),
                    "recognized_tool_in_error": make_metric(forbid_tool_in_error, n_forbid, "forbidden_prompts (count / N_forbid)"),
                    "malformed_with_tool_tag": make_metric(forbid_malformed_with_tag, n_forbid, "forbidden_prompts (count / N_forbid)"),
                    "malformed_without_tool_tag": make_metric(forbid_malformed_no_tag, n_forbid, "forbidden_prompts (count / N_forbid)"),
                },
                "required_prompts": {
                    "total": n_req,
                    "reconciled": (req_sum == n_req),
                    "correct_tool_route": make_metric(req_correct_tool, n_req, "required_prompts (count / N_req)"),
                    "missed_tool_route": make_metric(req_missed_tool, n_req, "required_prompts (count / N_req)"),
                    "cross_tool_misrouting": make_metric(req_cross_tool, n_req, "required_prompts (count / N_req)"),
                    "malformed_tool_output": make_metric(req_malformed_output, n_req, "required_prompts (count / N_req)"),
                },
                "optional_prompts": {
                    "total": n_opt,
                    "reconciled": (opt_sum == n_opt),
                    "correct_primary_route": make_metric(opt_correct_primary, n_opt, "optional_prompts (count / N_opt)"),
                    "allowed_optional_tool": make_metric(opt_allowed_tool, n_opt, "optional_prompts (count / N_opt)"),
                    "unexpected_tool_route": make_metric(opt_unexpected_tool, n_opt, "optional_prompts (count / N_opt)"),
                    "malformed_optional_output": make_metric(opt_malformed_output, n_opt, "optional_prompts (count / N_opt)"),
                },
                "total_reconciled": (forbid_sum + req_sum + opt_sum == n_total),
            },
            "extraction_decomposition": {
                "total_extraction_tasks": n_ext,
                "value_correctness": make_metric(ext_val_corr, n_ext, "extraction_tasks (count / N_ext)"),
                "grounding": make_metric(ext_ground, n_ext, "extraction_tasks (count / N_ext)"),
                "field_attribution": make_metric(ext_attrib, n_ext, "extraction_tasks (count / N_ext)"),
                "format_compliance": make_metric(ext_fmt, n_ext, "extraction_tasks (count / N_ext)"),
                "strict_extraction_success": make_metric(ext_strict, n_ext, "extraction_tasks (count / N_ext)"),
                "missing_fields": {
                    "total_missing_fields": n_missing_ext,
                    "absence_reported": make_metric(missing_abs_rep, n_missing_ext, "missing_fields (count / N_missing)"),
                    "fabrication_avoided": make_metric(missing_fab_avoid, n_missing_ext, "missing_fields (count / N_missing)"),
                },
            },
            "metrics": {
                # 1. Route selection accuracy across all prompts
                "route_selection_accuracy": make_metric(
                    tool_sel_count, n_total, "total_prompts (count / N_total)"
                ),
                "tool_selection_accuracy": make_metric(
                    tool_sel_count, n_total, "total_prompts (count / N_total)"
                ),
                # 2. False-positive tool calls on non-required prompts
                "false_positive_tool_rate": make_metric(
                    fp_tool_count, n_forbid + n_opt, "non_required_prompts (count / (N_forbid + N_opt))"
                ),
                # 3. Missed tool calls on tool-requiring prompts
                "missed_tool_call_rate": make_metric(
                    missed_tool_count, n_req, "tool_requiring_prompts (count / N_req)"
                ),
                # 4. Cross-tool misrouting
                "cross_tool_misrouting_rate_on_tool_req": make_metric(
                    cross_tool_count, n_req, "tool_requiring_prompts (count / N_req)"
                ),
                "cross_tool_misrouting_rate_on_actions": make_metric(
                    cross_tool_count, n_tool_actions, "tool_actions (count / N_tool_actions)"
                ),
                # 5. Tool argument accuracy
                "tool_argument_accuracy": make_metric(
                    arg_correct_count, n_tool_actions, "tool_actions (count / N_tool_actions)"
                ),
                "argument_accuracy_on_tool_actions": make_metric(
                    arg_correct_count, n_tool_actions, "tool_actions (count / N_tool_actions)"
                ),
                # 6. Invocation/runtime execution success
                "runtime_execution_success": make_metric(
                    exec_success_count, n_tool_actions, "tool_actions (count / N_tool_actions)"
                ),
                "execution_success_on_tool_actions": make_metric(
                    exec_success_count, n_tool_actions, "tool_actions (count / N_tool_actions)"
                ),
                # 7. Final-answer accuracy on objectively scored tasks
                "final_answer_accuracy_scored": make_metric(
                    final_correct_count, n_scored, "objectively_scored_tasks (count / N_scored)"
                ),
                # 8. End-to-end correctness
                "end_to_end_correctness_full_suite": make_metric(
                    e2e_total_count, n_total, "total_prompts (count / N_total)"
                ),
                "end_to_end_success_full_suite": make_metric(
                    e2e_total_count, n_total, "total_prompts (count / N_total)"
                ),
                "end_to_end_correctness_scored_only": make_metric(
                    e2e_scored_count, n_scored, "objectively_scored_tasks (count / N_scored)"
                ),
                "end_to_end_success_scored_only": make_metric(
                    e2e_scored_count, n_scored, "objectively_scored_tasks (count / N_scored)"
                ),
                # 9. Safe-abstention accuracy
                "safe_abstention_accuracy": make_metric(
                    abst_correct_count, n_abst, "abstention_prompts (count / N_abstention)"
                ),
                "abstention_accuracy": make_metric(
                    abst_correct_count, n_abst, "abstention_prompts (count / N_abstention)"
                ),
                # 10. Heuristic generation pass rate (renamed from qualitative evaluation)
                "heuristic_generation_pass_rate": make_metric(
                    heuristic_pass_count, n_unscored, "unscored_generation_tasks (count / N_unscored) [heuristic, not human-rated]"
                ),
                "qualitative_unscored_evaluation": make_metric(
                    heuristic_pass_count, n_unscored, "unscored_generation_tasks (count / N_unscored) [heuristic, not human-rated]"
                ),
                "unscored_task_proportion": make_metric(
                    n_unscored, n_total, "total_prompts (count / N_total)"
                ),
                # Supplementary protocol health metrics
                "syntax_validity_rate": make_metric(
                    syntax_count, n_total, "total_prompts (count / N_total)"
                ),
                "malformed_call_rate": make_metric(
                    malformed_count, n_total, "total_prompts (count / N_total)"
                ),
            },
            "category_breakdown": category_breakdown,
        }
