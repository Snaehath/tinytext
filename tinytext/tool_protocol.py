"""Structured Tool Protocol for TinyText Flash V1.5.

Defines a strict, safe, single-turn action grammar for tool selection and direct answers.
Guarantees zero arbitrary code execution (zero eval / exec / shell commands),
strict schema validation, and explicit abstention for unavailable facts or fields.

Supported Action Grammar:
- [CALC] <arithmetic expression>
- [FACT] <factual query key>
- [EXTRACT] key: <field> | text: <source passage>
- [ANSWER] <direct response text>
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    from tinytext.budget import BudgetDiagnostics
    from tinytext.tool_runtime import SafeToolRuntime

from tinytext.orchestrator import (
    LocalFactualRetriever,
    SafeArithmeticEvaluator,
    SourceGroundedExtractor,
)


class ToolActionType(str, Enum):
    CALC = "CALC"
    FACT = "FACT"
    EXTRACT = "EXTRACT"
    ANSWER = "ANSWER"
    READ_TEXT = "READ_TEXT"
    WRITE_TEXT = "WRITE_TEXT"


class ToolStatus(str, Enum):
    SUCCESS = "SUCCESS"
    UNSUPPORTED_TOOL = "UNSUPPORTED_TOOL"
    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    INVALID_CALC_EXPR = "INVALID_CALC_EXPR"
    UNAVAILABLE_FACT = "UNAVAILABLE_FACT"
    UNAVAILABLE_FIELD = "UNAVAILABLE_FIELD"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    PATH_TRAVERSAL_DETECTED = "PATH_TRAVERSAL_DETECTED"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    ENCODING_ERROR = "ENCODING_ERROR"
    TOOL_BUDGET_EXHAUSTED = "TOOL_BUDGET_EXHAUSTED"


@dataclass
class ToolParseResult:
    action: Optional[ToolActionType]
    raw_text: str
    payload: str
    is_valid: bool
    error_reason: Optional[str] = None


@dataclass
class ToolExecutionResult:
    action: Optional[ToolActionType]
    status: ToolStatus
    result: str
    is_abstention: bool
    error_detail: Optional[str] = None


class ToolProtocol:
    """Strict parser and executor for the TinyText V1.5 tool grammar."""

    # Disallow dangerous substrings in any payload
    DANGEROUS_PATTERNS = [
        re.compile(r"__import__", re.IGNORECASE),
        re.compile(r"\b(eval|exec|compile|globals|locals|getattr|setattr|delattr)\b", re.IGNORECASE),
        re.compile(r"\b(os|sys|subprocess|shutil|socket|requests|urllib)\b", re.IGNORECASE),
    ]

    ALLOWED_ACTIONS = {
        ToolActionType.CALC,
        ToolActionType.FACT,
        ToolActionType.EXTRACT,
        ToolActionType.ANSWER,
        ToolActionType.READ_TEXT,
        ToolActionType.WRITE_TEXT,
    }

    @classmethod
    def parse(cls, text: str) -> ToolParseResult:
        """Parses a model generation into a validated ToolParseResult."""
        raw = text.strip()
        if not raw:
            return ToolParseResult(
                action=None,
                raw_text=raw,
                payload="",
                is_valid=False,
                error_reason="Empty generation",
            )

        # Check for dangerous payload tokens defense-in-depth
        for pat in cls.DANGEROUS_PATTERNS:
            if pat.search(raw):
                return ToolParseResult(
                    action=None,
                    raw_text=raw,
                    payload="",
                    is_valid=False,
                    error_reason="Disallowed security-sensitive tokens in input",
                )

        # Match single leading action tag: [TAG] payload
        match = re.match(r"^\[([A-Za-z0-9_]+)\](?:\s*(.*))?$", raw, flags=re.DOTALL)
        if not match:
            # Check if there is an unclosed tag e.g. "[CALC 15 + 27"
            if raw.startswith("[") and "]" not in raw:
                return ToolParseResult(
                    action=None,
                    raw_text=raw,
                    payload="",
                    is_valid=False,
                    error_reason="Malformed request: Missing closing bracket ']' in action tag",
                )
            # Check if tags appear anywhere else in text
            tag_search = re.search(r"\[([A-Za-z0-9_]+)\]", raw)
            if tag_search:
                return ToolParseResult(
                    action=None,
                    raw_text=raw,
                    payload="",
                    is_valid=False,
                    error_reason="Malformed request: Action tag must appear strictly at start of response",
                )
            # Default fallback: Treat as unbracketed direct answer if no bracket
            return ToolParseResult(
                action=ToolActionType.ANSWER,
                raw_text=raw,
                payload=raw,
                is_valid=True,
                error_reason=None,
            )

        tag_name = match.group(1).upper()
        payload = (match.group(2) or "").strip()

        # Disallow multiple brackets in payload to prevent injection or chained calls
        if "[" in payload or "]" in payload:
            return ToolParseResult(
                action=None,
                raw_text=raw,
                payload=payload,
                is_valid=False,
                error_reason="Malformed request: Multiple or nested action tags are not allowed",
            )

        try:
            action_type = ToolActionType(tag_name)
        except ValueError:
            return ToolParseResult(
                action=None,
                raw_text=raw,
                payload=payload,
                is_valid=False,
                error_reason=f"Unsupported tool action: '[{tag_name}]'",
            )

        if not payload and action_type != ToolActionType.ANSWER:
            return ToolParseResult(
                action=action_type,
                raw_text=raw,
                payload="",
                is_valid=False,
                error_reason=f"Empty payload for action '[{tag_name}]'",
            )

        return ToolParseResult(
            action=action_type,
            raw_text=raw,
            payload=payload,
            is_valid=True,
            error_reason=None,
        )

    @classmethod
    def execute(
        cls,
        parse_res: ToolParseResult,
        runtime: Optional[SafeToolRuntime] = None,
        budget_diagnostics: Optional[BudgetDiagnostics] = None,
    ) -> ToolExecutionResult:
        """Executes a parsed tool action strictly using safe determinism."""
        if not parse_res.is_valid or parse_res.action is None:
            return ToolExecutionResult(
                action=parse_res.action,
                status=ToolStatus.MALFORMED_REQUEST if "Malformed" in (parse_res.error_reason or "") else ToolStatus.UNSUPPORTED_TOOL,
                result="I could not execute the requested action due to a malformed or unsupported request.",
                is_abstention=True,
                error_detail=parse_res.error_reason,
            )

        action = parse_res.action
        payload = parse_res.payload

        # Enforce tool call budget on all tool actions (except direct ANSWER)
        if action != ToolActionType.ANSWER and budget_diagnostics is not None:
            if not budget_diagnostics.can_call_tool():
                return ToolExecutionResult(
                    action=action,
                    status=ToolStatus.TOOL_BUDGET_EXHAUSTED,
                    result="Tool call budget exceeded for this request.",
                    is_abstention=True,
                    error_detail="Maximum permitted tool calls reached.",
                )
            budget_diagnostics.record_tool_call()

        # 1. Calculator
        if action == ToolActionType.CALC:
            eval_res = SafeArithmeticEvaluator.evaluate(payload)
            if eval_res is None:
                return ToolExecutionResult(
                    action=action,
                    status=ToolStatus.INVALID_CALC_EXPR,
                    result="Error: Invalid arithmetic expression. Cannot evaluate.",
                    is_abstention=True,
                    error_detail=f"Expression failed safe AST parsing: '{payload}'",
                )
            return ToolExecutionResult(
                action=action,
                status=ToolStatus.SUCCESS,
                result=str(eval_res),
                is_abstention=False,
            )

        # 2. Local Factual Lookup
        if action == ToolActionType.FACT:
            fact_res = LocalFactualRetriever.lookup(payload)
            if fact_res is None:
                return ToolExecutionResult(
                    action=action,
                    status=ToolStatus.UNAVAILABLE_FACT,
                    result="Fact unavailable: The requested fact is not present in the verified local knowledge store.",
                    is_abstention=True,
                    error_detail=f"Key '{payload}' not found in local factual store.",
                )
            return ToolExecutionResult(
                action=action,
                status=ToolStatus.SUCCESS,
                result=fact_res,
                is_abstention=False,
            )

        # 3. Source-Grounded Extraction
        if action == ToolActionType.EXTRACT:
            # Format expected: "key: <field> | text: <passage>" or standard "Output: <field> ... Extract ...: <passage>"
            extracted_val: Optional[str] = None
            if "|" in payload and "key:" in payload.lower() and "text:" in payload.lower():
                parts = payload.split("|", 1)
                key_part = parts[0].strip()
                text_part = parts[1].strip()
                field_match = re.search(r"key:\s*([A-Za-z0-9_]+)", key_part, re.IGNORECASE)
                field_name = field_match.group(1) if field_match else ""
                text_match = re.search(r"text:\s*(.+)$", text_part, re.IGNORECASE | re.DOTALL)
                passage = text_match.group(1).strip() if text_match else ""

                if field_name and passage:
                    extracted_dict = SourceGroundedExtractor._extract_field(field_name, passage)
                    if extracted_dict:
                        extracted_val = f"{field_name}: {extracted_dict}"
            else:
                # Fallback to query extraction
                dict_res = SourceGroundedExtractor.extract(payload)
                if dict_res:
                    extracted_val = SourceGroundedExtractor.format_output(dict_res)

            if extracted_val is None:
                return ToolExecutionResult(
                    action=action,
                    status=ToolStatus.UNAVAILABLE_FIELD,
                    result="Extraction unavailable: Requested field not found verbatim in source text.",
                    is_abstention=True,
                    error_detail="Field absent or ungrounded in source text.",
                )
            return ToolExecutionResult(
                action=action,
                status=ToolStatus.SUCCESS,
                result=extracted_val,
                is_abstention=False,
            )

        # 4. Workspace Text Read
        if action == ToolActionType.READ_TEXT:
            if runtime is None:
                return ToolExecutionResult(
                    action=action,
                    status=ToolStatus.PERMISSION_DENIED,
                    result="Permission denied: Filesystem tools are disabled (no workspace runtime configured).",
                    is_abstention=True,
                    error_detail="Workspace runtime not configured",
                )
            target_path = payload
            path_match = re.search(r"path:\s*([^\s|]+)", payload, re.IGNORECASE)
            if path_match:
                target_path = path_match.group(1).strip()

            success, out_msg, err_code = runtime.read_text(target_path)
            status_map = {
                "PERMISSION_DENIED": ToolStatus.PERMISSION_DENIED,
                "PATH_TRAVERSAL_DETECTED": ToolStatus.PATH_TRAVERSAL_DETECTED,
                "FILE_TOO_LARGE": ToolStatus.FILE_TOO_LARGE,
                "ENCODING_ERROR": ToolStatus.ENCODING_ERROR,
                "EXECUTION_ERROR": ToolStatus.EXECUTION_ERROR,
            }
            status = ToolStatus.SUCCESS if success else status_map.get(err_code or "", ToolStatus.EXECUTION_ERROR)
            return ToolExecutionResult(
                action=action,
                status=status,
                result=out_msg,
                is_abstention=not success,
                error_detail=None if success else out_msg,
            )

        # 5. Workspace Text Write
        if action == ToolActionType.WRITE_TEXT:
            if runtime is None:
                return ToolExecutionResult(
                    action=action,
                    status=ToolStatus.PERMISSION_DENIED,
                    result="Permission denied: Filesystem tools are disabled (no workspace runtime configured).",
                    is_abstention=True,
                    error_detail="Workspace runtime not configured",
                )
            if "|" in payload and "path:" in payload.lower() and "content:" in payload.lower():
                parts = payload.split("|", 1)
                p_match = re.search(r"path:\s*([^\s|]+)", parts[0], re.IGNORECASE)
                c_match = re.search(r"content:\s*(.+)$", parts[1], re.IGNORECASE | re.DOTALL)
                if not p_match or not c_match:
                    return ToolExecutionResult(
                        action=action,
                        status=ToolStatus.MALFORMED_REQUEST,
                        result="Malformed request: WRITE_TEXT expects 'path: <file> | content: <text>' format.",
                        is_abstention=True,
                        error_detail="Malformed argument schema",
                    )
                target_path = p_match.group(1).strip()
                content = c_match.group(1).strip()
            else:
                return ToolExecutionResult(
                    action=action,
                    status=ToolStatus.MALFORMED_REQUEST,
                    result="Malformed request: WRITE_TEXT expects 'path: <file> | content: <text>' format.",
                    is_abstention=True,
                    error_detail="Missing path or content delimiter '|'",
                )

            success, out_msg, err_code = runtime.write_text(target_path, content)
            status_map = {
                "PERMISSION_DENIED": ToolStatus.PERMISSION_DENIED,
                "PATH_TRAVERSAL_DETECTED": ToolStatus.PATH_TRAVERSAL_DETECTED,
                "PAYLOAD_TOO_LARGE": ToolStatus.PAYLOAD_TOO_LARGE,
                "EXECUTION_ERROR": ToolStatus.EXECUTION_ERROR,
            }
            status = ToolStatus.SUCCESS if success else status_map.get(err_code or "", ToolStatus.EXECUTION_ERROR)
            return ToolExecutionResult(
                action=action,
                status=status,
                result=out_msg,
                is_abstention=not success,
                error_detail=None if success else out_msg,
            )

        # 6. Direct Answer
        if action == ToolActionType.ANSWER:
            return ToolExecutionResult(
                action=action,
                status=ToolStatus.SUCCESS,
                result=payload,
                is_abstention=False,
            )

        return ToolExecutionResult(
            action=action,
            status=ToolStatus.UNSUPPORTED_TOOL,
            result="Unsupported action type.",
            is_abstention=True,
        )

    @classmethod
    def format_call(cls, action: ToolActionType, payload: str) -> str:
        """Helper to create a canonical tool call string."""
        return f"[{action.value}] {payload.strip()}"
