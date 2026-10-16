"""Restricted Read/Write Tool Runtime for TinyText Flash V1.5.5.

Provides a dedicated workspace boundary with strict path sanitization, permission policies,
and atomic file operations. Guarantees zero arbitrary code execution, zero shell access,
and defense-in-depth protection against path traversal, symlink escapes, and sensitive data access.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, Tuple


@dataclass
class WorkspacePolicy:
    """Enforces boundary rules and resource constraints on workspace filesystem operations."""

    workspace_root: Path
    max_file_size_bytes: int = 65536  # 64 KB
    max_payload_size_bytes: int = 65536  # 64 KB
    allowed_encoding: str = "utf-8"
    allow_read: bool = True
    allow_write: bool = True

    # Sensitive targets that must never be accessed even if within workspace
    denied_filenames: Set[str] = field(
        default_factory=lambda: {
            ".env",
            ".env.local",
            ".env.production",
            "id_rsa",
            "id_ed25519",
            "credentials",
            "secrets.json",
            "passwd",
            "shadow",
        }
    )
    denied_extensions: Set[str] = field(
        default_factory=lambda: {
            ".pt",
            ".pth",
            ".ckpt",
            ".bin",
            ".safetensors",
            ".onnx",
            ".key",
            ".pem",
            ".p12",
            ".pfx",
        }
    )
    denied_directories: Set[str] = field(
        default_factory=lambda: {
            "checkpoints",
            "data",
            ".git",
            ".ssh",
            "node_modules",
            "venv",
            ".venv",
        }
    )

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True, exist_ok=True)

    def validate_and_resolve_path(
        self,
        requested_path: str,
        for_write: bool = False,
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """Resolves requested relative path against workspace_root, returning (is_valid, resolved_path, error_reason).

        Rejects:
        - Non-string or empty paths.
        - Null bytes (defense-in-depth against C-string truncation).
        - Windows drive-qualified paths (e.g. C:, D:).
        - Absolute paths (starting with / or \\).
        - Traversal sequences ('..').
        - Symlink escapes pointing outside workspace_root.
        - Denied filenames, extensions, and directory names.
        """
        if not isinstance(requested_path, str):
            return False, None, "Invalid path type: path must be a string."

        raw = requested_path.strip()
        if not raw:
            return False, None, "Invalid path: Path cannot be empty."

        # Reject null bytes
        if "\0" in raw:
            return False, None, "Security violation: Null bytes prohibited in path."

        # Reject Windows drive qualification (e.g. C:path or D:/path)
        drive, _ = os.path.splitdrive(raw)
        if drive or re.match(r"^[a-zA-Z]:", raw):
            return False, None, "Security violation: Drive-qualified paths are prohibited."

        # Reject absolute paths (POSIX and Windows)
        if os.path.isabs(raw) or raw.startswith("/") or raw.startswith("\\"):
            return False, None, "Security violation: Absolute paths are prohibited."

        # Reject traversal elements ('..')
        components = re.split(r"[/\\]+", raw)
        if any(c == ".." for c in components):
            return False, None, "Security violation: Path traversal ('..') is prohibited."

        # Check denied directory or file names in components
        for c in components:
            c_clean = c.strip()
            if c_clean in self.denied_directories:
                return False, None, f"Security violation: Access to directory '{c_clean}' is prohibited."
            if c_clean in self.denied_filenames:
                return False, None, f"Security violation: Access to file '{c_clean}' is prohibited."

        # Check denied file extensions
        for ext in self.denied_extensions:
            if raw.lower().endswith(ext.lower()):
                return False, None, f"Security violation: Access to '{ext}' files is prohibited."

        # Resolve candidate path
        try:
            candidate = self.workspace_root / raw
            # If path does not exist yet (for write), resolve parent
            if for_write and not candidate.exists():
                parent_resolved = candidate.parent.resolve()
                if not parent_resolved.is_relative_to(self.workspace_root):
                    return False, None, "Security violation: Parent directory escapes workspace root."
                resolved = parent_resolved / candidate.name
            else:
                resolved = candidate.resolve()
        except Exception as e:
            return False, None, f"Path resolution error: {e}"

        # Strict containment validation
        try:
            if not resolved.is_relative_to(self.workspace_root):
                return False, None, "Security violation: Resolved path escapes workspace root."
        except AttributeError:
            # Fallback for Python versions without Path.is_relative_to
            try:
                resolved.relative_to(self.workspace_root)
            except ValueError:
                return False, None, "Security violation: Resolved path escapes workspace root."

        # Symlink check on existing targets: ensure the target doesn't resolve outside
        if candidate.is_symlink():
            real_target = candidate.resolve()
            if not real_target.is_relative_to(self.workspace_root):
                return False, None, "Security violation: Symlink escapes workspace root."

        return True, resolved, None


class SafeToolRuntime:
    """Isolated, permission-restricted execution runtime for file read/write tools."""

    def __init__(self, policy: WorkspacePolicy):
        self.policy = policy

    def read_text(self, requested_path: str) -> Tuple[bool, str, Optional[str]]:
        """Reads a UTF-8 text file within the workspace boundary.

        Returns (success, content_or_error_message, error_code).
        """
        if not self.policy.allow_read:
            return False, "Permission denied: Workspace read operations are disabled.", "PERMISSION_DENIED"

        is_valid, resolved_path, err = self.policy.validate_and_resolve_path(requested_path, for_write=False)
        if not is_valid or resolved_path is None:
            return False, f"Permission denied: {err}", "PATH_TRAVERSAL_DETECTED"

        if not resolved_path.exists():
            return False, f"File not found: '{resolved_path.name}' does not exist in workspace.", "EXECUTION_ERROR"

        if not resolved_path.is_file():
            return False, f"Invalid target: '{resolved_path.name}' is a directory or special file.", "EXECUTION_ERROR"

        file_size = resolved_path.stat().st_size
        if file_size > self.policy.max_file_size_bytes:
            return (
                False,
                f"File too large: File size ({file_size} bytes) exceeds limit ({self.policy.max_file_size_bytes} bytes).",
                "FILE_TOO_LARGE",
            )

        try:
            content = resolved_path.read_text(encoding=self.policy.allowed_encoding, errors="strict")
            return True, content, None
        except UnicodeDecodeError:
            return False, f"Encoding error: File is not valid {self.policy.allowed_encoding} text.", "ENCODING_ERROR"
        except Exception as e:
            return False, f"Read error: {e}", "EXECUTION_ERROR"

    def write_text(self, requested_path: str, content: str) -> Tuple[bool, str, Optional[str]]:
        """Atomically creates or overwrites a UTF-8 text file within the workspace boundary.

        Returns (success, result_message, error_code).
        """
        if not self.policy.allow_write:
            return False, "Permission denied: Workspace write operations are disabled.", "PERMISSION_DENIED"

        if not isinstance(content, str):
            return False, "Invalid payload: Content must be a string.", "EXECUTION_ERROR"

        is_valid, resolved_path, err = self.policy.validate_and_resolve_path(requested_path, for_write=True)
        if not is_valid or resolved_path is None:
            return False, f"Permission denied: {err}", "PATH_TRAVERSAL_DETECTED"

        # Check payload size in bytes
        payload_bytes = content.encode(self.policy.allowed_encoding)
        if len(payload_bytes) > self.policy.max_payload_size_bytes:
            return (
                False,
                f"Payload too large: Content size ({len(payload_bytes)} bytes) exceeds limit ({self.policy.max_payload_size_bytes} bytes).",
                "PAYLOAD_TOO_LARGE",
            )

        # Ensure parent directory exists within workspace root
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        # Safe atomic write: write to temp file in same directory and atomically replace
        temp_file = None
        try:
            temp_file = tempfile.NamedTemporaryFile(
                dir=resolved_path.parent,
                delete=False,
                mode="w",
                encoding=self.policy.allowed_encoding,
            )
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()

            # Atomic replace
            os.replace(temp_file.name, resolved_path)
            return True, f"Successfully wrote {len(content)} characters to '{resolved_path.name}'.", None
        except Exception as e:
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except OSError:
                    pass
            return False, f"Write error: {e}", "EXECUTION_ERROR"
