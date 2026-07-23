"""
Tier 2 rule loading.

Phase 2.2 loads declarative threshold rules from JSON files only. YAML can be
added later if dependency cost is justified.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

from .threshold_contracts import ComparisonOperator, Rule


DEFAULT_RULES_DIR = Path(__file__).parent / "rules"


class RuleLoadError(ValueError):
    """Raised when a rule file cannot be loaded or validated."""


def load_rules(
    path: Union[str, Path] = DEFAULT_RULES_DIR,
    include_disabled: bool = False,
) -> List[Rule]:
    """
    Load threshold rules from one JSON file or a directory of JSON files.

    Files may contain either:
    - a list of rule objects
    - an object with a top-level "rules" list
    - a single rule object
    """
    rule_data = []
    for source in _iter_rule_files(Path(path)):
        rule_data.extend(_load_rule_data(source))

    rules = [_build_rule(item) for item in rule_data]
    _validate_unique_rule_ids(rules)

    if not include_disabled:
        rules = [rule for rule in rules if rule.enabled]
    return rules


def _iter_rule_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() != ".json":
            raise RuleLoadError(f"Unsupported rule file type: {path.suffix}")
        yield path
        return

    if not path.exists():
        raise RuleLoadError(f"Rules path does not exist: {path}")
    if not path.is_dir():
        raise RuleLoadError(f"Rules path is not a file or directory: {path}")

    for file_path in sorted(path.glob("*.json")):
        yield file_path


def _load_rule_data(path: Path) -> List[Dict[str, Any]]:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuleLoadError(f"Malformed JSON in {path}: {exc.msg}") from exc

    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, dict) and "rules" in raw:
        data = raw["rules"]
    elif isinstance(raw, dict):
        data = [raw]
    else:
        raise RuleLoadError(f"Rule file {path} must contain an object or list")

    if not isinstance(data, list):
        raise RuleLoadError(f"Rule file {path} field 'rules' must be a list")
    if not all(isinstance(item, dict) for item in data):
        raise RuleLoadError(f"Rule file {path} contains a non-object rule")
    return data


def _build_rule(data: Dict[str, Any]) -> Rule:
    try:
        rule = Rule.from_dict(data)
        _reject_unimplemented_operators(rule)
        return rule
    except (TypeError, ValueError) as exc:
        rule_id = data.get("rule_id", "<unknown>")
        raise RuleLoadError(f"Invalid rule {rule_id}: {exc}") from exc


def _reject_unimplemented_operators(rule: Rule) -> None:
    unsupported = [
        condition.operator.value
        for condition in [*rule.conditions, *rule.context_gates]
        if condition.operator in (ComparisonOperator.RISING, ComparisonOperator.DROPPING)
    ]
    if unsupported:
        operators = ", ".join(sorted(set(unsupported)))
        raise ValueError(
            f"Trend operators are not implemented for rule loading yet: {operators}"
        )


def _validate_unique_rule_ids(rules: List[Rule]) -> None:
    seen = set()
    duplicates = []
    for rule in rules:
        if rule.rule_id in seen:
            duplicates.append(rule.rule_id)
        seen.add(rule.rule_id)

    if duplicates:
        names = ", ".join(sorted(set(duplicates)))
        raise RuleLoadError(f"Duplicate rule_id values: {names}")


__all__ = [
    "DEFAULT_RULES_DIR",
    "RuleLoadError",
    "load_rules",
]
