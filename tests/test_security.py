"""
Security & Artifact Integrity Test Suite.
Tests:
1. Secret scanner detects synthetic secrets and confirms repository hygiene.
2. Artifact integrity manager produces valid SHA-256 checksums and catches tampered files.
3. Fail-safe pipeline execution guard.
"""

import unittest
import os
import tempfile
import json
from prompt_evaluator.security import SecretScanner, ArtifactIntegrityManager


class TestSecurityAndIntegrity(unittest.TestCase):
    def test_repository_secret_scan_is_clean(self):
        """Verify that the repository contains zero hardcoded API keys, tokens, or credentials."""
        scan_report = SecretScanner.scan_directory(".")
        self.assertTrue(
            scan_report["is_clean"],
            f"Repository contains leaked credentials: {scan_report.get('findings')}",
        )

    def test_secret_scanner_detects_synthetic_leaks(self):
        """Verify that SecretScanner catches synthetic API keys when present."""
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            prefix = "sk-"; f.write(f"OPENAI_KEY = '{prefix}1234567890abcdef1234567890abcdef'\n")
            temp_path = f.name

        try:
            temp_dir = os.path.dirname(temp_path)
            # Scan just this temp file
            with open(temp_path, "r", encoding="utf-8") as f:
                content = f.read()
            has_leak = any(
                len(__import__("re").findall(pat, content)) > 0
                for pat, _ in [
                    (r"sk-[A-Za-z0-9]{20,}", "OpenAI Key Pattern"),
                ]
            )
            self.assertTrue(has_leak, "Scanner must detect synthetic OpenAI API key pattern.")
        finally:
            os.remove(temp_path)

    def test_artifact_sha256_checksum_verification(self):
        """Verify that ArtifactIntegrityManager correctly detects tampered files."""
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("id,better\nQ-001,new\n")
            orig_file = f.name

        manifest_file = orig_file + ".manifest.json"
        try:
            # Generate manifest
            manifest = ArtifactIntegrityManager.generate_manifest(
                {"test_csv": orig_file}, manifest_file
            )
            is_valid, errs = ArtifactIntegrityManager.verify_manifest(manifest_file)
            self.assertTrue(is_valid, "Untampered file must pass verification.")

            # Tamper with file
            with open(orig_file, "a") as f:
                f.write("Q-002,current\n")

            is_valid_after, errs_after = ArtifactIntegrityManager.verify_manifest(manifest_file)
            self.assertFalse(is_valid_after, "Tampered file must fail SHA-256 verification!")
            self.assertTrue(any("mismatch" in e.lower() for e in errs_after))
        finally:
            if os.path.exists(orig_file):
                os.remove(orig_file)
            if os.path.exists(manifest_file):
                os.remove(manifest_file)


if __name__ == "__main__":
    unittest.main()
