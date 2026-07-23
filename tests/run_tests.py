"""
Small pytest runner for Lyr's test groups.

Examples:
    python tests/run_tests.py --unit
    python tests/run_tests.py --integration
    python tests/run_tests.py --sensor
    python tests/run_tests.py --threshold
    python tests/run_tests.py --all
"""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path.cwd()

TEST_GROUPS = {
    "unit": ["tests/unit"],
    "integration": ["tests/integration"],
    "sensor": [
        "tests/unit/test_sensor_api.py",
        "tests/unit/test_sensor_contracts.py",
        "tests/unit/test_registry.py",
        "tests/unit/test_sensor_history.py",
    ],
    "threshold": [
        "tests/unit/test_breach_log.py",
        "tests/unit/test_ollama_advisory.py",
        "tests/unit/test_threshold_contracts.py",
        "tests/unit/test_threshold_gate.py",
        "tests/unit/test_threshold_rules.py",
        "tests/unit/test_threshold_state.py",
    ],
}


def _build_args(args: argparse.Namespace) -> list[str]:
    selected = []

    if args.all:
        selected.extend(TEST_GROUPS["unit"])
        selected.extend(TEST_GROUPS["integration"])
    else:
        if args.unit:
            selected.extend(TEST_GROUPS["unit"])
        if args.integration:
            selected.extend(TEST_GROUPS["integration"])
        if args.sensor:
            selected.extend(TEST_GROUPS["sensor"])
        if args.threshold:
            selected.extend(TEST_GROUPS["threshold"])

    if not selected:
        selected.extend(TEST_GROUPS["sensor"])

    pytest_args = [
        sys.executable,
        "-m",
        "pytest",
        "--rootdir=.",
        *selected,
    ]
    if args.quiet:
        pytest_args.append("-q")
    if args.verbose:
        pytest_args.append("-v")
    if args.extra:
        pytest_args.extend(args.extra)
    return pytest_args


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lyr test groups.")
    parser.add_argument("--all", action="store_true", help="Run unit and integration tests.")
    parser.add_argument("--unit", action="store_true", help="Run all unit tests.")
    parser.add_argument("--integration", action="store_true", help="Run integration tests.")
    parser.add_argument("--sensor", action="store_true", help="Run sensor registry/history tests.")
    parser.add_argument("--threshold", action="store_true", help="Run Tier 2 threshold tests.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Pass -q to pytest.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Pass -v to pytest.")
    parser.add_argument("extra", nargs="*", help="Extra arguments passed through to pytest.")
    args = parser.parse_args()

    completed = subprocess.run(_build_args(args), cwd=PROJECT_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
