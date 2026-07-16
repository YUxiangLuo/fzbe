from __future__ import annotations

import argparse
import unittest

from py.deep_tests.support.profiles import profile
from py.src.error_protocol import KNOWN_ERROR_TAGS, format_known_error
from py.src.log_safety import summarize_cli_args


class CliArgumentRedactionTests(unittest.TestCase):
    @profile("fast")
    def test_sensitive_paths_are_redacted_and_payloads_are_summarized(self):
        namespace = argparse.Namespace(
            model="ma",
            data_file="/srv/students/secret-dataset.json",
            save_model="/srv/artifacts/out.pkl",
            load_model=None,
            output_state_file="/srv/state.pkl",
            data='[{"y": 1}, {"y": 2}]',
            window=3,
        )
        summarized = summarize_cli_args(namespace)
        # Real filesystem locations never reach the logs.
        self.assertEqual("<redacted>", summarized["data_file"])
        self.assertEqual("<redacted>", summarized["save_model"])
        self.assertEqual("<redacted>", summarized["output_state_file"])
        # A None-valued sensitive key is left untouched (nothing to leak).
        self.assertIsNone(summarized["load_model"])
        # The inline dataset payload is reduced to a length, not echoed verbatim.
        self.assertEqual(f"<json payload chars={len(namespace.data)}>", summarized["data"])
        # Harmless hyperparameters still pass through for debuggability.
        self.assertEqual(3, summarized["window"])
        self.assertEqual("ma", summarized["model"])
        # The helper copies rather than mutating the parsed namespace.
        self.assertEqual("/srv/artifacts/out.pkl", namespace.save_model)


class ErrorTagNormalizationTests(unittest.TestCase):
    @profile("fast")
    def test_known_tags_are_normalized_with_single_spacing(self):
        for tag in KNOWN_ERROR_TAGS:
            with self.subTest(tag=tag):
                self.assertEqual(f"{tag} detail", format_known_error(ValueError(f"{tag}   detail  ")))
                self.assertEqual(tag, format_known_error(ValueError(tag)))

    @profile("fast")
    def test_timeout_errors_are_tagged_without_duplication(self):
        self.assertEqual("[TIMEOUT] deadline exceeded", format_known_error(TimeoutError("deadline exceeded")))
        self.assertEqual("[TIMEOUT] deadline exceeded", format_known_error(TimeoutError("[TIMEOUT] deadline exceeded")))
        self.assertEqual("[TIMEOUT]", format_known_error(TimeoutError()))

    @profile("fast")
    def test_untagged_errors_are_not_classified(self):
        # Only the stable protocol tags cross the subprocess boundary as typed errors;
        # everything else stays an opaque non-zero exit for the caller to surface.
        self.assertIsNone(format_known_error(RuntimeError("database connection refused")))
        self.assertIsNone(format_known_error(ValueError("plain message without a tag")))


if __name__ == "__main__":
    unittest.main()
