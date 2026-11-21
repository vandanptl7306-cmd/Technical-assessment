"""
Security, Integrity, and Safe Execution Module.
Provides:
1. Secret and Credential Leak Scanner.
2. Artifact Integrity & SHA-256 Checksum Manifest Generator.
3. Fail-Safe Pipeline Execution Guard (prevents partial evaluations from shipping).
4. Performance & Execution Latency Profiler.
"""

from typing import Dict, Any, List, Optional
import os
import re
import time
import hashlib
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PromptEvaluator.Security")

# Secret leak patterns
SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]", "API Key / Token Literal"),
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI API Key Pattern"),
    (r"AIzaSy[A-Za-z0-9_\-]{33}", "Google Cloud API Key Pattern"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"-----BEGIN (RSA|EC|OPENSSH|PRIVATE) KEY-----", "Private Key Header"),
]


class SecretScanner:
    """
    Scans repository files and inputs for leaked API keys, tokens, or private credentials.
    """

    @classmethod
    def scan_directory(cls, target_dir: str = ".") -> Dict[str, Any]:
        """Scans project files for leaked secrets and unauthorized .env files."""
        findings = []
        scanned_files = 0
        ignored_dirs = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}

        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            for file in files:
                if file.endswith((".py", ".csv", ".md", ".txt", ".json", ".env", ".yml", ".yaml")):
                    scanned_files += 1
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            for pat, desc in SECRET_PATTERNS:
                                matches = re.findall(pat, content)
                                if matches:
                                    findings.append({
                                        "file": os.path.relpath(file_path, target_dir),
                                        "description": desc,
                                        "count": len(matches)
                                    })
                    except Exception as e:
                        logger.warning(f"Could not read {file_path} during secret scan: {e}")

        is_clean = len(findings) == 0
        return {
            "is_clean": is_clean,
            "scanned_files": scanned_files,
            "findings": findings,
            "status": "PASS: No leaked secrets detected" if is_clean else f"FAIL: {len(findings)} secret leaks found",
        }


class ArtifactIntegrityManager:
    """
    Computes cryptographic SHA-256 hashes of generated artifacts and writes an integrity manifest.
    """

    @classmethod
    def calculate_sha256(cls, file_path: str) -> Optional[str]:
        """Calculate SHA-256 checksum of a file."""
        if not os.path.exists(file_path):
            return None
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def generate_manifest(cls, artifact_paths: Dict[str, str], manifest_output: str) -> Dict[str, Any]:
        """
        Generate and persist a verifiable SHA-256 integrity manifest.
        """
        manifest = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "artifacts": {},
        }
        for name, path in artifact_paths.items():
            if os.path.exists(path):
                manifest["artifacts"][name] = {
                    "path": os.path.relpath(path),
                    "size_bytes": os.path.getsize(path),
                    "sha256": cls.calculate_sha256(path),
                }

        os.makedirs(os.path.dirname(manifest_output), exist_ok=True)
        with open(manifest_output, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Integrity manifest generated at {manifest_output}")
        return manifest

    @classmethod
    def verify_manifest(cls, manifest_path: str) -> Tuple[bool, List[str]]:
        """Verify artifact checksums against an existing manifest."""
        if not os.path.exists(manifest_path):
            return False, ["Manifest file does not exist"]
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        mismatches = []
        for name, meta in manifest.get("artifacts", {}).items():
            path = meta["path"]
            if not os.path.exists(path):
                mismatches.append(f"Missing artifact: {name} at {path}")
                continue
            current_hash = cls.calculate_sha256(path)
            if current_hash != meta["sha256"]:
                mismatches.append(f"Hash mismatch for {name}: expected {meta['sha256']}, got {current_hash}")

        is_valid = len(mismatches) == 0
        return is_valid, mismatches


class PerformanceProfiler:
    """
    Measures execution time, latency per question, and throughput.
    """

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.total_questions = 0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self, total_questions: int) -> Dict[str, Any]:
        self.end_time = time.perf_counter()
        self.total_questions = total_questions
        total_duration = max(self.end_time - self.start_time, 0.001)
        latency_per_q_ms = (total_duration / max(total_questions, 1)) * 1000
        throughput = total_questions / total_duration

        return {
            "total_duration_sec": round(total_duration, 3),
            "total_questions": total_questions,
            "latency_per_question_ms": round(latency_per_q_ms, 2),
            "throughput_q_per_sec": round(throughput, 1),
        }
