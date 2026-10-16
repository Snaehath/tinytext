"""Learned Reasoning Effort Conditioning Specification for TinyText Flash V1.5.6.

Defines input/output formatting, effort mode enum, prompt conditioning,
and scratchpad isolation (separating internal scratchpad from user-visible answer).

Uses existing Byte-Level BPE vocabulary (zero added special tokens, zero weight migration needed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class EffortMode(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Semantic descriptions of effort modes
EFFORT_DESCRIPTIONS: Dict[EffortMode, str] = {
    EffortMode.LOW: (
        "Direct mode: Provide a direct, concise solution without scratchpad or intermediate reasoning steps."
    ),
    EffortMode.MEDIUM: (
        "Intermediate mode: Provide key reasoning and calculation steps before stating the final answer."
    ),
    EffortMode.HIGH: (
        "Deliberate mode: Use <scratchpad>...</scratchpad> for structured problem decomposition, "
        "step execution, and explicit consistency/sanity verification. Then provide the final answer as [ANSWER] <result>."
    ),
}

# System prompt directives per mode
EFFORT_SYSTEM_DIRECTIVES: Dict[EffortMode, str] = {
    EffortMode.LOW: (
        "You are TinyText, a helpful assistant. [EFFORT: LOW] "
        "Direct mode: Provide the direct final answer immediately as [ANSWER] <result>. Do not use scratchpad or steps."
    ),
    EffortMode.MEDIUM: (
        "You are TinyText, a helpful assistant. [EFFORT: MEDIUM] "
        "Intermediate mode: Show key intermediate steps under [STEPS], then provide the final answer as [ANSWER] <result>."
    ),
    EffortMode.HIGH: (
        "You are TinyText, a helpful assistant. [EFFORT: HIGH] "
        "Deliberate mode: First use <scratchpad>...</scratchpad> to decompose the problem into sub-goals, "
        "execute each step, and verify consistency with a check. Then provide the verified final answer as [ANSWER] <result>."
    ),
}


@dataclass(frozen=True)
class ParsedEffortResponse:
    raw_text: str
    mode: Optional[EffortMode]
    has_scratchpad: bool
    scratchpad_content: str
    steps_content: str
    visible_answer: str
    is_valid_format: bool
    verification_detected: bool


def format_effort_prompt(user_prompt: str, mode: EffortMode | str) -> str:
    """Formats a full conversation prompt conditioned on the requested effort mode."""
    if isinstance(mode, str):
        mode = EffortMode(mode.lower().strip())
    sys_dir = EFFORT_SYSTEM_DIRECTIVES[mode]
    return f"System:\n{sys_dir}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def parse_effort_response(raw_text: str, expected_mode: Optional[EffortMode] = None) -> ParsedEffortResponse:
    """Parses a generation into internal scratchpad/steps vs user-visible answer."""
    text = raw_text.strip()
    scratchpad_content = ""
    steps_content = ""
    visible_answer = ""
    has_scratchpad = False
    verification_detected = False

    # Extract scratchpad if present
    scratch_match = re.search(r"<scratchpad>(.*?)</scratchpad>", text, flags=re.DOTALL | re.IGNORECASE)
    if scratch_match:
        has_scratchpad = True
        scratchpad_content = scratch_match.group(1).strip()
        # Look for verification in scratchpad
        if re.search(r"\b(check|verify|verification|sanity|consistent|parity|valid)\b", scratchpad_content, re.IGNORECASE):
            verification_detected = True

    # Extract steps if present
    steps_match = re.search(r"\[STEPS\](.*?)(?=\[ANSWER\]|$)", text, flags=re.DOTALL | re.IGNORECASE)
    if steps_match:
        steps_content = steps_match.group(1).strip()

    # Extract [ANSWER]
    if "[ANSWER]" in text:
        visible_answer = text.split("[ANSWER]", 1)[1].strip()
    elif has_scratchpad:
        # Take text after </scratchpad>
        after_scratch = text.split("</scratchpad>", 1)[1].strip()
        visible_answer = after_scratch
    else:
        # Fallback to text directly if neither marker exists
        visible_answer = text

    # Clean any trailing punctuation or tags from visible answer
    visible_answer = re.sub(r"<\|endoftext\|>", "", visible_answer).strip()

    # Format validity check based on expected mode
    is_valid = True
    if expected_mode == EffortMode.LOW:
        # Low mode should have [ANSWER] and NO scratchpad
        if has_scratchpad:
            is_valid = False
    elif expected_mode == EffortMode.MEDIUM:
        # Medium should have steps or multiple lines and [ANSWER]
        if has_scratchpad:
            is_valid = False
        if not steps_content and "\n" not in text:
            is_valid = False
    elif expected_mode == EffortMode.HIGH:
        # High mode requires scratchpad and visible answer
        if not has_scratchpad or not visible_answer:
            is_valid = False

    return ParsedEffortResponse(
        raw_text=raw_text,
        mode=expected_mode,
        has_scratchpad=has_scratchpad,
        scratchpad_content=scratchpad_content,
        steps_content=steps_content,
        visible_answer=visible_answer,
        is_valid_format=is_valid,
        verification_detected=verification_detected,
    )


def extract_visible_response(raw_text: str, show_scratchpad: bool = False) -> str:
    """Returns the user-visible response, keeping internal scratchpad hidden by default."""
    if show_scratchpad:
        return raw_text.strip()
    parsed = parse_effort_response(raw_text)
    return parsed.visible_answer
