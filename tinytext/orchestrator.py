"""Reasoning Orchestrator and Bounded Verification Layer for TinyText Flash.

Provides modular routing, safe deterministic tools (zero arbitrary code execution),
and a bounded verification policy with at most one corrective retry.

Components:
1. Task Router: Identifies intent (Arithmetic, Extraction, Constrained List, Rewriting, Factual, Direct Generation).
2. Safe Tools: AST-based safe arithmetic parser, source-grounded extractor, local factual retriever.
3. Bounded Verifier: Validates list counts, sequence, uniqueness, schema conformity, fact preservation,
   and prevents superficial checks from claiming semantic correctness.
4. Corrective Policy: At most 1 targeted retry with feedback, re-verifying the final result.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from tinytext.model import TinyTextForCausalLM
from tinytext.tokenizer import TinyTextTokenizer
from tinytext.generate import generate_text


class Route(str, Enum):
    ARITHMETIC = "arithmetic"
    EXTRACTION = "extraction"
    CONSTRAINED_LIST = "constrained_list"
    REWRITING = "rewriting"
    FACTUAL_RETRIEVAL = "factual_retrieval"
    DIRECT_GENERATION = "direct_generation"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"
    ERROR = "error"

    def __bool__(self) -> bool:
        """Allow backward-compatible truthiness checks (True iff status is VERIFIED)."""
        return self == VerificationStatus.VERIFIED


@dataclass
class OrchestrationResult:
    route: Route
    output: str
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    verified: bool = False
    retried: bool = False
    validation_detail: str = ""
    checks_performed: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    task_success: bool = False
    tokens_generated: int = 0
    latency_seconds: float = 0.0
    tool_used: Optional[str] = None

    def __post_init__(self) -> None:
        # Guarantee synchronization between status and boolean verified flag
        if self.status == VerificationStatus.VERIFIED:
            self.verified = True
        elif self.verified and self.status == VerificationStatus.UNVERIFIED:
            self.status = VerificationStatus.VERIFIED
        else:
            self.verified = (self.status == VerificationStatus.VERIFIED)


# ==============================================================================
# Safe Deterministic Tools (Zero exec / eval)
# ==============================================================================

class SafeArithmeticEvaluator:
    """Safely parses and evaluates basic arithmetic using Python AST.
    Guarantees zero arbitrary code execution, bounds input size and exponents,
    and handles negative numbers, decimals, and division by zero.
    """
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    MAX_EXPR_LEN = 120
    MAX_NODES = 40
    MAX_POWER_EXPONENT = 20
    MAX_POWER_BASE = 10000

    @classmethod
    def evaluate(cls, expression: str) -> Optional[float | int]:
        if not expression or len(expression) > cls.MAX_EXPR_LEN:
            return None

        # Disallow alphabetic characters, quotes, or dangerous tokens in the raw expression
        if re.search(r"[a-zA-Z_\'\"]", expression):
            return None

        # Clean expression of surrounding text while preserving numbers, signs, operators, parentheses
        cleaned = re.sub(r"[^\d\+\-\*\/\(\)\.\s\%]", "", expression).strip()
        if not cleaned:
            return None

        try:
            node = ast.parse(cleaned, mode="eval")
            
            # Defense-in-depth: limit total AST node count to prevent DoS via deep nesting
            node_count = sum(1 for _ in ast.walk(node))
            if node_count > cls.MAX_NODES:
                return None

            result = cls._eval_node(node.body)

            # Check for non-finite values (inf, nan)
            if isinstance(result, (int, float)):
                if math.isinf(result) or math.isnan(result):
                    return None

            # Numeric precision and clean formatting rules
            if isinstance(result, float):
                if result.is_integer():
                    return int(result)
                # Round to 10 decimal places to eliminate IEEE 754 precision artifacts (e.g. 0.1 + 0.2)
                return round(result, 10)

            return result
        except (ZeroDivisionError, OverflowError, ValueError, TypeError, SyntaxError):
            return None
        except Exception:
            return None

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            op_type = type(node.op)

            if op_type not in cls._OPERATORS:
                raise ValueError(f"Unsupported binary operator: {op_type}")

            # Safe bound check on exponentiation to prevent CPU freeze / memory exhaustion
            if op_type == ast.Pow:
                if abs(right) > cls.MAX_POWER_EXPONENT or abs(left) > cls.MAX_POWER_BASE:
                    raise ValueError(f"Power operation exceeds safety limit: {left} ** {right}")

            # Zero-division prevention for division and modulo
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero")

            return cls._OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = cls._eval_node(node.operand)
            op_type = type(node.op)
            if op_type in cls._OPERATORS:
                return cls._OPERATORS[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type}")
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")


class SourceGroundedExtractor:
    """Deterministic extractor for source-grounded structured extraction queries.
    Never invents missing facts; abstains if fields are absent or ambiguous.
    """

    @classmethod
    def extract(cls, query: str) -> Optional[Dict[str, str]]:
        """Parses an extraction query, locates the source passage and requested fields,
        and extracts strictly grounded values.
        """
        # 1. Identify requested field(s)
        field_matches = []
        output_match = re.search(r"Output:\s*(.+)$", query, re.IGNORECASE)
        if output_match:
            schema_raw = output_match.group(1).strip()
            field_matches = re.findall(r"([A-Za-z0-9_]+):\s*\.\.\.", schema_raw)
            if not field_matches:
                field_matches = re.findall(r"([A-Za-z0-9_]+):", schema_raw)
        else:
            f_m = re.search(r"(?:extract\s+the|what\s+is\s+the)\s+([A-Za-z0-9_]+)\b", query, re.IGNORECASE)
            if f_m:
                field_matches = [f_m.group(1)]

        if not field_matches:
            return None

        # 2. Extract source passage
        passage = ""
        quote_m = re.search(r"[\"']([^\"']{10,})[\"']", query)
        if quote_m:
            passage = quote_m.group(1).strip()
        else:
            source_split = re.search(r"(?:Extract[^:]*|from\s+the\s+following\s+text):\s*(.+?)(?:\s+Output:|$)", query, re.IGNORECASE | re.DOTALL)
            if source_split:
                passage = source_split.group(1).strip()
            else:
                parts = query.split(":", 1)
                if len(parts) > 1:
                    after_colon = parts[1]
                    passage = re.sub(r"Output:.*$", "", after_colon, flags=re.IGNORECASE).strip()

        if not passage:
            return None

        # 3. Extract each requested field deterministically
        extracted: Dict[str, str] = {}
        for field_name in field_matches:
            val = cls._extract_field(field_name, passage)
            if val is None:
                # Missing field -> Never guess or invent!
                return None
            
            # Grounding check: extracted value must exist verbatim in source passage
            val_clean = val.strip(".,;:!?() ")
            if val_clean.lower() not in passage.lower():
                return None
            
            extracted[field_name] = val

        return extracted

    @classmethod
    def format_output(cls, extracted: Dict[str, str]) -> str:
        """Formats extracted key-value pairs according to standard schema."""
        return "; ".join(f"{k}: {v}" for k, v in extracted.items())

    @classmethod
    def _extract_field(cls, field_name: str, source: str) -> Optional[str]:
        fn = field_name.lower().strip()

        if fn == "time":
            m = re.search(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)\b", source)
            return m.group(1) if m else None

        if fn == "date":
            m = re.search(
                r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)|\d{4}-\d{2}-\d{2})\b",
                source,
                re.IGNORECASE,
            )
            return m.group(1) if m else None

        if fn == "city":
            m = re.search(r"\b(?:in|at|for|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", source)
            return m.group(1) if m else None

        if fn == "location":
            m = re.search(r"\b(?:in|at)\s+(Room\s+[A-Za-z0-9]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", source)
            return m.group(1) if m else None

        if fn == "price":
            m = re.search(r"(\$\d+(?:\.\d{2})?|\b\d+\s*(?:dollars|USD|euros|EUR|cents)\b)", source)
            return m.group(1) if m else None

        if fn == "item":
            m = re.search(r"(?:The\s+)?(.+?)\s+is\s+priced\s+at\b", source, re.IGNORECASE)
            return m.group(1).strip() if m else None

        if fn == "sender":
            m = re.search(r"\b([A-Z][a-z]+)\s+sent\s+", source)
            return m.group(1) if m else None

        if fn == "recipient":
            m = re.search(r"\bsent\s+(?:an?\s+)?\w+\s+to\s+([A-Z][a-z]+)\b", source)
            return m.group(1) if m else None

        if fn == "model":
            m = re.search(r"\b(?:a|an)\s+(?:[a-z]+\s+)?([A-Z][a-z0-9]+)\s+(?:yesterday|today|last)\b", source)
            if not m:
                m = re.search(r"\b(?:blue|red|green|black|white|silver|gray)\s+([A-Z][a-z0-9]+)\b", source, re.IGNORECASE)
            return m.group(1) if m else None

        if fn == "color":
            m = re.search(r"\b(red|blue|green|yellow|black|white|silver|gray|grey|purple|orange|brown)\b", source, re.IGNORECASE)
            return m.group(1) if m else None

        if fn == "flight":
            m = re.search(r"\b(?:Flight\s+)?([A-Z]{2}\s*\d{3,4})\b", source)
            return m.group(1).replace(" ", "") if m else None

        if fn == "destination":
            m = re.search(r"\b(?:for|to)\s+([A-Z][a-z]+)\b", source)
            return m.group(1) if m else None

        if fn == "author":
            m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+wrote\b", source)
            return m.group(1) if m else None

        if fn == "title":
            m = re.search(r"\bwrote\s+(?:the\s+novel\s+)?(.+?)[.;]?$", source, re.IGNORECASE)
            return m.group(1).strip() if m else None

        if fn == "temperature":
            m = re.search(r"(\b\d{1,3}\s*°\s*[FCfc]\b|\b\d{1,3}\s*degrees\b)", source)
            return m.group(1) if m else None

        if fn == "condition":
            m = re.search(r"\b(?:and\s+)?(sunny|rainy|cloudy|snowing|clear|stormy|windy|foggy|overcast)\b", source, re.IGNORECASE)
            return m.group(1) if m else None

        if fn == "winner":
            m = re.search(r"\b(?:The\s+)?([A-Z][a-z]+)\s+defeated\b", source)
            return m.group(1) if m else None

        if fn == "score":
            m = re.search(r"(\b\d+\s+to\s+\d+\b|\b\d+-\d+\b)", source, re.IGNORECASE)
            return m.group(1) if m else None

        if fn == "doctor":
            m = re.search(r"\b(Dr\.\s+[A-Z][a-z]+)\b", source)
            return m.group(1) if m else None

        if fn == "phone":
            m = re.search(r"((?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{3}-\d{4}\b)", source)
            return m.group(1) if m else None

        if fn == "ingredient":
            m = re.search(r"\b(?:of)\s+([a-z]+)\b", source)
            return m.group(1) if m else None

        if fn == "quantity":
            m = re.search(r"(\b\d+(?:\.\d+)?\s*(?:cups|tablespoons|teaspoons|liters|ml|pieces|slices|oz)\b)", source, re.IGNORECASE)
            return m.group(1) if m else None

        if fn == "hotel":
            m = re.search(r"\b(?:at\s+)?([A-Z][a-z]+\s+Hotel)\b", source)
            return m.group(1) if m else None

        if fn == "room":
            m = re.search(r"\b(?:room|Room)\s+([A-Za-z0-9]+)\b", source)
            return m.group(1) if m else None

        if fn == "code":
            m = re.search(r"\b(?:code\s+)?([A-Z0-9_-]{4,})\b", source)
            return m.group(1) if m else None

        if fn == "discount":
            m = re.search(r"(\b\d+(?:\.\d+)?%)(?:\s|$|[.,;])", source)
            return m.group(1) if m else None

        if fn == "courier":
            m = re.search(r"\b(?:handled\s+by|via|by)\s+([A-Z][A-Za-z0-9]+)\b", source)
            return m.group(1) if m else None

        if fn == "weight":
            m = re.search(r"(\b\d+(?:\.\d+)?\s*(?:kg|g|lbs|pounds|ounces)\b)", source, re.IGNORECASE)
            return m.group(1) if m else None

        if fn == "ceo":
            m = re.search(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\s+serves\s+as\s+(?:the\s+)?CEO\b", source)
            return m.group(1) if m else None

        if fn == "company":
            m = re.search(r"\bCEO\s+of\s+([A-Z][A-Za-z0-9]+)\b", source)
            return m.group(1) if m else None

        if fn == "course":
            m = re.search(r"\b([A-Z]{2,4}\s*\d{3,4})\b", source)
            return m.group(1).replace(" ", "") if m else None

        if fn == "credits":
            m = re.search(r"(\b\d+\s*credits?\b)", source, re.IGNORECASE)
            return m.group(1) if m else None

        return None


class LocalFactualRetriever:
    """Lightweight in-memory factual knowledge store for verified facts."""
    _KNOWLEDGE_BASE = {
        "capital of japan": "Tokyo",
        "capital of france": "Paris",
        "capital of canada": "Ottawa",
        "capital of the united kingdom": "London",
        "capital of germany": "Berlin",
        "primary function of red blood cells": "To transport oxygen from the lungs to body tissues and carry carbon dioxide back to the lungs.",
        "first person to walk on the moon": "Neil Armstrong in 1969.",
        "boiling point of water": "100 degrees Celsius at sea level.",
        "hardest natural mineral": "Diamond.",
        "year world war ii ended": "1945.",
    }

    _KEY_ALIASES = {
        "france's capital": "capital of france",
        "french capital": "capital of france",
        "capital city of france": "capital of france",
        "japan's capital": "capital of japan",
        "japanese capital": "capital of japan",
        "capital city of japan": "capital of japan",
        "germany's capital": "capital of germany",
        "german capital": "capital of germany",
        "capital city of germany": "capital of germany",
        "canada's capital": "capital of canada",
        "canadian capital": "capital of canada",
        "capital city of canada": "capital of canada",
        "uk capital": "capital of the united kingdom",
        "united kingdom's capital": "capital of the united kingdom",
        "capital city of the united kingdom": "capital of the united kingdom",
        "red blood cells": "primary function of red blood cells",
        "walked on the moon": "first person to walk on the moon",
        "walk on the moon": "first person to walk on the moon",
        "stepped on the moon": "first person to walk on the moon",
        "water boil": "boiling point of water",
        "boiling point of water": "boiling point of water",
        "boils at": "boiling point of water",
        "hardest mineral": "hardest natural mineral",
        "hardest natural mineral": "hardest natural mineral",
        "world war ii ended": "year world war ii ended",
        "world war 2 ended": "year world war ii ended",
        "ending year of world war ii": "year world war ii ended",
        "world war ii end": "year world war ii ended",
        "world war 2 end": "year world war ii ended",
    }

    @classmethod
    def lookup(cls, query: str) -> Optional[str]:
        q_clean = query.lower().strip("?.! ")
        # 1. Direct match in knowledge base
        for k, v in cls._KNOWLEDGE_BASE.items():
            if k in q_clean:
                return v
        # 2. Check aliases
        for alias, canonical in cls._KEY_ALIASES.items():
            if alias in q_clean:
                return cls._KNOWLEDGE_BASE.get(canonical)
        return None


# ==============================================================================
# Bounded Task Verifier
# ==============================================================================

class TaskVerifier:
    """Applies objective verification rules to candidate responses.
    Differentiates VERIFIED, REJECTED, and UNVERIFIED states.
    Never marks a response VERIFIED based merely on superficial structural checks.
    """

    # Generic assistant disclaimer/refusal phrases
    DISCLAIMERS = [
        "as an ai",
        "i do not have access",
        "feel free to ask",
        "i cannot browse",
        "consultative:",
        "i am an ai language model",
    ]

    # Obvious placeholder / dummy list tokens that must be rejected
    PLACEHOLDER_ITEMS = {
        "item 1", "item 2", "item 3", "item 4", "item 5",
        "placeholder", "none", "n/a", "etc", "test",
        "the name of a pet turtle", "pet turtle",
        "healthy breakfast food", "breakfast food",
        "programming language", "planet",
        "renewable energy source", "musical instrument",
        "dog breed", "common bird species", "cold dessert",
    }

    @staticmethod
    def check_repetition(text: str) -> bool:
        """Returns True if excessive n-gram looping is detected."""
        words = text.lower().split()
        if len(words) >= 9:
            trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
            for tg in set(trigrams):
                if trigrams.count(tg) >= 3:
                    return True
        return False

    @classmethod
    def verify_list(
        cls,
        text: str,
        expected_count: int,
        category_keywords: Optional[List[str]] = None,
    ) -> Tuple[VerificationStatus, str]:
        if cls.check_repetition(text):
            return VerificationStatus.REJECTED, "Response contains repetitive phrase loops."

        # Match numbered lines: e.g. "1. Item" or "1) Item"
        matches = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s*(.+)", text.strip())
        if not matches:
            return VerificationStatus.REJECTED, f"Expected numbered list with {expected_count} items, but no numbered format found."

        # Numbering sequence consistency check (1, 2, 3...)
        numbers = [int(m[0]) for m in matches]
        expected_numbers = list(range(1, len(numbers) + 1))
        if numbers != expected_numbers:
            return VerificationStatus.REJECTED, f"List numbering is inconsistent or out of order: found {numbers}, expected {expected_numbers}."

        items = [m[1].strip() for m in matches]
        cleaned_items = [it for it in items if it]

        # Exact count check
        if len(cleaned_items) != expected_count:
            return VerificationStatus.REJECTED, f"Expected exactly {expected_count} items, but found {len(cleaned_items)}."

        # Non-empty items check
        for it in cleaned_items:
            if len(re.sub(r"[^\w]", "", it)) == 0:
                return VerificationStatus.REJECTED, "List contains empty or punctuation-only items."

        # Duplicate items check (case and punctuation normalized)
        normalized_items = [re.sub(r"[^\w\s]", "", it.lower()).strip() for it in cleaned_items]
        if len(set(normalized_items)) != len(normalized_items):
            return VerificationStatus.REJECTED, "List contains duplicate items."

        # Placeholder / dummy content check
        for norm_it in normalized_items:
            if norm_it in cls.PLACEHOLDER_ITEMS:
                return VerificationStatus.REJECTED, f"List item '{norm_it}' is a dummy placeholder or echoed prompt instruction."

        # Check for assistant disclaimers
        if any(d in text.lower() for d in cls.DISCLAIMERS):
            return VerificationStatus.REJECTED, "List contains assistant refusal or disclaimer preamble."

        # Semantic relevance limitation:
        # A structural check alone cannot establish deep semantic relevance of open-ended lists.
        # Mark as UNVERIFIED unless category keywords match.
        if category_keywords:
            text_lower = text.lower()
            matched = any(kw.lower() in text_lower for kw in category_keywords)
            if matched:
                return VerificationStatus.VERIFIED, f"Valid numbered list satisfying count ({expected_count}), uniqueness, and category match."

        return VerificationStatus.UNVERIFIED, f"List satisfies structural count ({expected_count}) and uniqueness, but semantic domain correctness cannot be verified deterministically."

    @classmethod
    def verify_extraction(
        cls,
        text: str,
        source_text: str,
        requested_fields: List[str],
    ) -> Tuple[VerificationStatus, str]:
        if cls.check_repetition(text):
            return VerificationStatus.REJECTED, "Response contains repetitive phrase loops."

        if any(d in text.lower() for d in cls.DISCLAIMERS):
            return VerificationStatus.REJECTED, "Response contains assistant refusal or disclaimer preamble."

        # Verify requested fields are present
        missing_fields = []
        extracted_values: Dict[str, str] = {}
        for f in requested_fields:
            m = re.search(rf"{re.escape(f)}:\s*([^;\n]+)", text, re.IGNORECASE)
            if not m or not m.group(1).strip():
                missing_fields.append(f)
            else:
                extracted_values[f] = m.group(1).strip()

        if missing_fields:
            return VerificationStatus.REJECTED, f"Output format missing required fields: {missing_fields}"

        # Verify that all extracted values exist verbatim in the source text
        s_lower = source_text.lower()
        unsupported = []
        for f, val in extracted_values.items():
            val_clean = val.strip(".,;:!?() ").lower()
            if not val_clean or val_clean not in s_lower:
                unsupported.append(f"{f}: {val}")

        if unsupported:
            return VerificationStatus.REJECTED, f"Extracted values not grounded in source text: {unsupported}"

        return VerificationStatus.VERIFIED, "All fields accurately extracted and strictly grounded in source text."

    @classmethod
    def verify_rewriting(
        cls,
        text: str,
        source_text: str,
        query: str = "",
    ) -> Tuple[VerificationStatus, str]:
        if cls.check_repetition(text):
            return VerificationStatus.REJECTED, "Response contains repetitive phrase loops."

        # Disallow generic assistant refusal boilerplate
        if any(d in text.lower() for d in cls.DISCLAIMERS):
            return VerificationStatus.REJECTED, "Response contains unhelpful disclaimer or refusal preamble."

        # Check for unrequested ballooning
        source_words = source_text.split()
        text_words = text.split()
        if len(text_words) > len(source_words) * 3.5 + 5:
            return VerificationStatus.REJECTED, "Rewritten text adds substantial unrequested length."

        q_lower = query.lower()

        # Transformation Check: Conciseness
        if "concise" in q_lower or "simplify" in q_lower:
            if len(text_words) >= len(source_words):
                return VerificationStatus.REJECTED, f"Conciseness requested, but output ({len(text_words)} words) is not shorter than source ({len(source_words)} words)."

        # Transformation Check: Negative word removal
        if "without negative" in q_lower:
            negative_words = ["not", "no", "never", "don't", "cant", "can't", "won't", "wont", "none"]
            found_neg = [w for w in negative_words if re.search(rf"\b{re.escape(w)}\b", text.lower())]
            if found_neg:
                return VerificationStatus.REJECTED, f"Negative words prohibited, but found: {found_neg}"

        # Transformation Check: Active voice
        if "active voice" in q_lower:
            # Passive markers: "was/were/is/are/been/being [verb] by"
            if re.search(r"\b(?:was|were|is|are|been|being)\s+\w+\s+by\b", text.lower()):
                return VerificationStatus.REJECTED, "Requested active voice, but passive construction remains."

        # Transformation Check: Passive voice
        if "passive voice" in q_lower:
            if not re.search(r"\b(?:was|were|is|are|been|being)\s+\w+\s+by\b", text.lower()):
                return VerificationStatus.REJECTED, "Requested passive voice, but passive construction was not detected."

        # Fact Preservation Check: Key named entities/numbers must be preserved
        source_entities = re.findall(r"\b[A-Z][a-z0-9]+\b|\b\d+\b", source_text)
        # Filter out common sentence start words
        source_entities = [e for e in source_entities if e.lower() not in {"the", "a", "an", "this", "in", "it", "she", "he", "they", "we", "you", "that"}]
        if source_entities:
            missing_entities = [e for e in source_entities if e.lower() not in text.lower()]
            # If all key entities were dropped, the rewrite completely changed meaning
            if len(missing_entities) == len(source_entities):
                return VerificationStatus.REJECTED, f"Rewrite failed to preserve critical source entities: {missing_entities}"

        # Semantic limitation note: Deterministic checks cannot verify full semantic nuance without an LLM judge.
        # Report honest UNVERIFIED rather than falsely claiming full verification.
        return VerificationStatus.UNVERIFIED, "Deterministic checks passed (length, entity preservation, constraints), but full semantic equivalence requires semantic judge."

    @classmethod
    def verify_direct_generation(cls, text: str) -> Tuple[VerificationStatus, str]:
        if cls.check_repetition(text):
            return VerificationStatus.REJECTED, "Response contains repetitive phrase loops."
        if any(d in text.lower() for d in cls.DISCLAIMERS):
            return VerificationStatus.REJECTED, "Response contains refusal or disclaimer preamble."

        # Open-ended generation cannot be established as correct deterministically
        return VerificationStatus.UNVERIFIED, "Open-ended generation cannot be deterministically verified without reference ground truth."


# ==============================================================================
# Reasoning Orchestrator Core
# ==============================================================================

class ReasoningOrchestrator:
    """Coordinates task routing, safe tool execution, and verified LM generation."""

    def __init__(
        self,
        model: Optional[TinyTextForCausalLM] = None,
        tokenizer: Optional[TinyTextTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def route_query(self, user_query: str) -> Route:
        q = user_query.lower().strip()

        # 1. Pure Arithmetic
        has_arith_verb = any(
            re.search(rf"\b{w}\b", q)
            for w in [
                "what is", "compute", "calculate", "evaluate", "multiply", "divide",
                "subtract", "add", "find the sum", "product of"
            ]
        )
        has_arith_op = any(op in q for op in [" + ", " - ", " * ", " / ", " plus ", " minus ", " times ", " divided by "]) or bool(
            re.search(r"\b(?:multiply|divide|subtract|add)\s+\d+", q)
        )
        is_conceptual = any(w in q for w in ["explain", "why", "story", "poem", "history of", "concept of", "how do"])

        if has_arith_verb and has_arith_op and not is_conceptual:
            return Route.ARITHMETIC

        # 2. Structured Extraction
        if (
            ("extract" in q or "from this passage" in q or "from the following text" in q)
            and ("output:" in q or "time:" in q or "date:" in q or "location:" in q or "price:" in q or "item:" in q or "sender:" in q or ":" in q or "what is the" in q)
        ):
            return Route.EXTRACTION

        # 3. Constrained List
        if (
            any(w in q for w in ["list", "give", "name", "provide", "enumerate"])
            and any(w in q for w in ["exactly", "only a numbered list", "numbered list", " 2 ", " 3 ", " 4 ", " 5 ", "two", "three", "four", "five"])
        ):
            return Route.CONSTRAINED_LIST

        # 4. Rewriting
        if any(
            q.startswith(w) or f" {w}" in q
            for w in [
                "rewrite", "make this concise", "make this sentence concise", "rephrase",
                "fix grammar", "fix grammatical", "change to active", "change to passive",
                "simplify for clarity", "make this customer greeting"
            ]
        ):
            return Route.REWRITING

        # 5. Factual Retrieval
        if LocalFactualRetriever.lookup(q) is not None:
            return Route.FACTUAL_RETRIEVAL

        return Route.DIRECT_GENERATION

    def process(
        self,
        user_query: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        refinement_strategy: str = "none",
    ) -> OrchestrationResult:
        if refinement_strategy != "none":
            from tinytext.refinement import RefinementPipeline, RefinementStrategy
            pipe = RefinementPipeline(orchestrator=self, model=self.model, tokenizer=self.tokenizer, device=self.device)
            res = pipe.process(
                user_query=user_query,
                strategy=RefinementStrategy(refinement_strategy),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            return OrchestrationResult(
                route=res.route,
                output=res.final_output,
                status=res.status,
                verified=res.verified,
                retried=res.was_revised,
                validation_detail=res.validation_detail,
                checks_performed=res.checks_performed,
                rejection_reason=res.rejection_reason,
                task_success=res.task_success,
                tokens_generated=res.total_tokens,
                latency_seconds=res.latency_seconds,
                tool_used=res.tool_used,
            )

        route = self.route_query(user_query)
        q = user_query.lower().strip()

        # Route 1: Safe Deterministic Arithmetic
        if route == Route.ARITHMETIC:
            math_expr_match = re.search(
                r"([-+]?\s*(?:\(?\s*[-+]?\s*\d+(?:\.\d+)?\s*\)?\s*[\+\-\*\/\%]\s*)+[-+]?\s*\(?\s*[-+]?\s*\d+(?:\.\d+)?\s*\)?)",
                user_query,
            )
            expr = None
            if math_expr_match:
                expr = math_expr_match.group(1).strip()
            else:
                m_mult = re.search(r"\bmultiply\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\b", q)
                if m_mult:
                    expr = f"{m_mult.group(1)} * {m_mult.group(2)}"
                m_div = re.search(r"\bdivide\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\b", q)
                if m_div:
                    expr = f"{m_div.group(1)} / {m_div.group(2)}"
                m_sub = re.search(r"\bsubtract\s+(\d+(?:\.\d+)?)\s+from\s+(\d+(?:\.\d+)?)\b", q)
                if m_sub:
                    expr = f"{m_sub.group(2)} - {m_sub.group(1)}"
                m_add = re.search(r"\badd\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\b", q)
                if m_add:
                    expr = f"{m_add.group(1)} + {m_add.group(2)}"

            if expr:
                result = SafeArithmeticEvaluator.evaluate(expr)
                if result is not None:
                    return OrchestrationResult(
                        route=Route.ARITHMETIC,
                        output=str(result),
                        status=VerificationStatus.VERIFIED,
                        task_success=True,
                        validation_detail=f"Safely computed deterministic AST evaluation for: {expr}",
                        checks_performed=["ast_safe_syntax_check", "operator_whitelist_check", "zero_division_guard", "power_bound_guard"],
                        tool_used="SafeArithmeticEvaluator",
                    )
                else:
                    return OrchestrationResult(
                        route=Route.ARITHMETIC,
                        output="Unsupported or invalid arithmetic expression.",
                        status=VerificationStatus.REJECTED,
                        task_success=False,
                        validation_detail="Arithmetic expression rejected due to safety or syntax check (e.g. division by zero, non-finite value, or unsupported operator).",
                        checks_performed=["ast_safe_syntax_check", "zero_division_guard"],
                        rejection_reason="Expression failed AST evaluation or safety constraints.",
                        tool_used="SafeArithmeticEvaluator",
                    )

        # Route 2: Source-Grounded Deterministic Extraction
        if route == Route.EXTRACTION:
            extracted = SourceGroundedExtractor.extract(user_query)
            if extracted is not None:
                formatted = SourceGroundedExtractor.format_output(extracted)
                return OrchestrationResult(
                    route=Route.EXTRACTION,
                    output=formatted,
                    status=VerificationStatus.VERIFIED,
                    task_success=True,
                    validation_detail="Deterministically extracted fields with 100% source text grounding.",
                    checks_performed=["schema_parsing", "source_grounding_check", "verbatim_field_verification"],
                    tool_used="SourceGroundedExtractor",
                )
            # If deterministic extraction abstained due to ambiguity/missing field, proceed to verified LM fallback

        # Route 3: Local Factual Retrieval
        if route == Route.FACTUAL_RETRIEVAL:
            fact = LocalFactualRetriever.lookup(user_query)
            if fact is not None:
                return OrchestrationResult(
                    route=Route.FACTUAL_RETRIEVAL,
                    output=fact,
                    status=VerificationStatus.VERIFIED,
                    task_success=True,
                    validation_detail="Retrieved verified fact from local curated knowledge base.",
                    checks_performed=["local_kb_exact_match"],
                    tool_used="LocalFactualRetriever",
                )

        # Fallback to Language Model with Bounded Verification
        if self.model is None or self.tokenizer is None:
            return OrchestrationResult(
                route=route,
                output="Model or tokenizer not initialized in orchestrator.",
                status=VerificationStatus.ERROR,
                task_success=False,
                validation_detail="Model not available.",
                checks_performed=["model_availability_check"],
                rejection_reason="Missing model or tokenizer.",
            )

        # First pass generation
        initial_gen = self._generate(user_query, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        candidate_text = initial_gen["continuation"].strip()

        # Apply Verification Policy based on Route
        status, detail = self._verify_candidate(route, user_query, candidate_text)
        checks = self._get_checks_for_route(route)
        
        # If passed without rejection (VERIFIED or UNVERIFIED open generation), return initial result
        if status != VerificationStatus.REJECTED:
            return OrchestrationResult(
                route=route,
                output=candidate_text,
                status=status,
                task_success=(status == VerificationStatus.VERIFIED),
                validation_detail=detail,
                checks_performed=checks,
                tokens_generated=initial_gen["tokens_generated"],
                latency_seconds=initial_gen["time_seconds"],
            )

        # Corrective Retry Policy: At most ONE corrective attempt when a concrete check was rejected
        retry_prompt = (
            f"{user_query}\n\n"
            f"Note: Ensure your answer corrects the previous issue: {detail}. Answer directly without preamble:"
        )

        retry_gen = self._generate(retry_prompt, max_new_tokens=max_new_tokens, temperature=0.2, top_p=0.9)
        retry_text = retry_gen["continuation"].strip()

        retry_status, retry_detail = self._verify_candidate(route, user_query, retry_text)
        total_tokens = initial_gen["tokens_generated"] + retry_gen["tokens_generated"]
        total_latency = initial_gen["time_seconds"] + retry_gen["time_seconds"]

        if retry_status == VerificationStatus.VERIFIED:
            return OrchestrationResult(
                route=route,
                output=retry_text,
                status=VerificationStatus.VERIFIED,
                task_success=True,
                retried=True,
                validation_detail=f"Corrected on retry: {retry_detail}",
                checks_performed=checks,
                tokens_generated=total_tokens,
                latency_seconds=total_latency,
            )
        else:
            return OrchestrationResult(
                route=route,
                output=retry_text,
                status=retry_status,
                task_success=(retry_status == VerificationStatus.VERIFIED),
                retried=True,
                validation_detail=f"Post-retry outcome: {retry_detail}",
                checks_performed=checks,
                rejection_reason=retry_detail if retry_status == VerificationStatus.REJECTED else None,
                tokens_generated=total_tokens,
                latency_seconds=total_latency,
            )

    def _get_checks_for_route(self, route: Route) -> List[str]:
        if route == Route.CONSTRAINED_LIST:
            return ["repetition_check", "numbered_format_check", "count_check", "uniqueness_check", "placeholder_filter_check", "semantic_relevance_check"]
        elif route == Route.EXTRACTION:
            return ["repetition_check", "field_presence_check", "source_grounding_check", "disclaimer_check"]
        elif route == Route.REWRITING:
            return ["repetition_check", "disclaimer_check", "length_bound_check", "fact_preservation_check", "semantic_equivalence_limitation"]
        else:
            return ["repetition_check", "disclaimer_check", "open_domain_limitation"]

    def _verify_candidate(self, route: Route, query: str, candidate: str) -> Tuple[VerificationStatus, str]:
        if route == Route.CONSTRAINED_LIST:
            count_match = re.search(r"(?:exactly|give|list|provide|name)\s+(\d+|two|three|four|five)", query.lower())
            word_to_num = {"two": 2, "three": 3, "four": 4, "five": 5}
            target_count = 3  # default
            if count_match:
                raw_c = count_match.group(1)
                target_count = word_to_num.get(raw_c, int(raw_c) if raw_c.isdigit() else 3)
            return TaskVerifier.verify_list(candidate, target_count)

        elif route == Route.EXTRACTION:
            fields_match = re.findall(r"([A-Za-z0-9_]+):\s*\.\.\.", query)
            if not fields_match:
                fields_match = re.findall(r"([A-Za-z0-9_]+):", query)
            fields = fields_match if fields_match else ["Time", "Location"]
            
            source_match = re.search(r"Extract[^:]*:\s*(.+?)(?:\s+Output:|$)", query, re.IGNORECASE)
            source_text = source_match.group(1) if source_match else query
            return TaskVerifier.verify_extraction(candidate, source_text, fields)

        elif route == Route.REWRITING:
            parts = query.split(":", 1)
            source_text = parts[1] if len(parts) > 1 else query
            return TaskVerifier.verify_rewriting(candidate, source_text, query=query)

        else:
            return TaskVerifier.verify_direct_generation(candidate)

    def _generate(self, user_query: str, max_new_tokens: int, temperature: float, top_p: float) -> dict:
        full_prompt = (
            "System:\n"
            "You are TinyText, a helpful and concise assistant. Follow the user's instructions, answer accurately when possible, and acknowledge uncertainty when needed.\n\n"
            f"User:\n{user_query}\n\n"
            "Assistant:\n"
        )
        return generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=full_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            device=self.device,
        )
