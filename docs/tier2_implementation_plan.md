# Tier 2 Implementation Plan

This plan splits Threshold Gate implementation into small phases. Each phase has
tests that must pass before the next phase begins.

## Phase 2.1: Rule Contracts and Validation

Goal: Define the rule schema without building the evaluator.

Build:

- `recognition/threshold_contracts.py`
- `Rule`
- `RuleCondition`
- `RuleSeverity`
- `RuleMode`
- `StalenessPolicy`
- `BreachStatus`
- `BreachState`
- `BreachLogEntry`

Decisions enforced:

- Declarative rules only.
- No `eval`, Python expressions, or arbitrary code.
- Use enter and clear thresholds, no deadband.
- Use wall-clock sustain and clear-delay timers, no consecutive reading counts.
- Cross-sensor rules must declare stale-data behavior.

Tests:

- Reject rule with `clear_threshold >= enter_threshold` for high-threshold rules.
- Reject negative `sustained_for_seconds`.
- Reject negative `clear_delay_seconds`.
- Reject unsupported comparison operators.
- Accept valid single-sensor temperature rule.
- Accept valid cross-sensor rule with `alert_stale`.
- Ensure rule serialization round-trips.

Exit criteria:

- Contract tests pass.
- No evaluator logic exists yet.

## Phase 2.2: Rule Loading From Files

Goal: Load rules from versioned config files.

Build:

- `recognition/threshold_rules.py`
- `recognition/rules/`
- JSON rule loader first. YAML can be added later if dependency cost is worth it.

Tests:

- Load valid rule file.
- Reject malformed JSON.
- Reject duplicate `rule_id`.
- Reject invalid rule via contract validation.
- Load only enabled rules by default.
- Preserve disabled rules when explicitly requested.

Exit criteria:

- Rules can be loaded from files.
- Bad rule files fail loudly before runtime.

## Phase 2.3: SQLite Breach State Store

Goal: Persist active/pending/clearing breach state without relying on memory.

Build:

- `recognition/threshold_state.py`
- SQLite DB: `recognition/database/threshold_state.db`
- `BreachStateStore`

Tests:

- Create schema on first use.
- Upsert `pending_sustain` state.
- Upsert `active` state.
- Transition to `clearing`.
- Transition to `cleared`.
- Retrieve by `rule_id` and `sensor_id`.
- Delete or compact cleared states.
- Survive store restart.
- Concurrent upserts for different sensors do not corrupt state.

Exit criteria:

- State is persistent.
- Restart does not lose active breach state.

## Phase 2.4: SQLite Breach Log Store

Goal: Persist audit-ready breach events.

Build:

- `recognition/breach_log.py`
- SQLite DB: `recognition/database/breach_log.db`
- `BreachLogStore`

Tests:

- Create schema on first use.
- Append breach log entry.
- Retrieve by `breach_id`.
- Query by `rule_id`.
- Query by `sensor_id`.
- Query by time range.
- Mark `human_reviewed`.
- Attach Tier 3 decision.
- Attach Tier 4 action.
- Survive store restart.

Exit criteria:

- Log-only mode has a real durable target.
- Breach logs are queryable for audit.

## Phase 2.5: Single-Sensor State Machine

Goal: Implement deterministic state transitions for one reading and one rule.

Build:

- `recognition/threshold_gate.py`
- `ThresholdGate`
- `_evaluate_single_sensor_rule`
- `_update_state`

State transitions:

- `idle -> pending_sustain`
- `pending_sustain -> active`
- `pending_sustain -> idle`
- `active -> clearing`
- `clearing -> cleared`
- `clearing -> active`

Tests:

- Reading below enter threshold stays idle.
- Reading above enter threshold creates `pending_sustain`.
- Condition sustained long enough becomes `active`.
- Condition drops before sustain time resets to idle.
- Active breach entering clear threshold becomes `clearing`.
- Clear condition sustained long enough becomes `cleared`.
- Clear condition interrupted returns to `active`.
- `clear_threshold` prevents boundary flapping.
- `sustained_for_seconds = 0` triggers immediately.
- `clear_delay_seconds = 0` clears immediately.

Exit criteria:

- Temperature rule boundary table passes.
- No cross-sensor behavior yet.

## Phase 2.6: Context-Aware Single-Sensor Evaluation

Goal: Reduce false positives with deterministic context gates.

Build:

- Context gate support on rules.
- History lookup from `HistoryStore`.
- Environmental sensor support, e.g. outdoor temperature or heat index.

Example:

- Indoor temperature crosses threshold.
- Outdoor temperature/heat index is also high.
- No smoke/humidity/power corroboration.
- Result is lower severity or log-only warning, not critical escalation.

Tests:

- Hot-day context downgrades temperature breach severity.
- Smoke corroboration upgrades temperature breach severity.
- Humidity drop corroboration upgrades temperature breach severity.
- Missing optional context does not crash evaluation.
- Missing required context follows staleness policy.

Exit criteria:

- Temperature thresholds can distinguish likely ambient heat from likely danger.
- Every context decision is explainable in breach reason text.

## Phase 2.7: Cross-Sensor Rules and Staleness

Goal: Evaluate rules that require multiple sensors or sensor types.

Build:

- `_evaluate_cross_sensor_rule`
- Related sensor history fetch.
- `alert_stale`, `fail_open`, and `fail_closed` policies.

Tests:

- Temperature high plus humidity dropping triggers cross-sensor breach.
- Temperature high without humidity data uses `alert_stale` by default.
- `fail_closed` does not trigger primary breach when required sensor is stale.
- `fail_open` evaluates with available data.
- Stale sensor breach is lower severity and explains missing sensor.
- Cross-sensor history window is configurable per rule.

Exit criteria:

- Missing data behavior is explicit.
- No silent pass or silent escalation on stale related sensors.

## Phase 2.8: Breach Context Builder and Clustering

Goal: Package related breaches into one context for Tier 3.

Build:

- `BreachContext`
- `build_context`
- `cluster_breaches`

Tests:

- Breaches within 500ms cluster together.
- Breaches outside window remain separate.
- Same-location breaches cluster.
- Unrelated locations do not cluster.
- Context includes triggered rules.
- Context includes recent primary sensor history.
- Context includes related sensor history.
- One cluster produces one Tier 3-ready context.

Exit criteria:

- Alert storms are reduced deterministically before Tier 3.

## Phase 2.9: Log-Only Mode

Goal: Validate rules without escalating.

Build:

- Global log-only toggle.
- Per-rule log-only mode.
- Breach logging on active transitions.

Tests:

- Global log-only logs breach but returns no escalation context.
- Per-rule log-only logs breach but returns no escalation context.
- Non-log-only rule returns breach context.
- Log entries include context snapshot.
- Log-only does not skip state transitions.

Exit criteria:

- New rules can run safely in production-like data before escalation is enabled.

## Phase 2.10: Read-Only Tier 2 API

Goal: Expose observability, not rule mutation.

Build:

- `GET /api/v1/threshold/rules`
- `GET /api/v1/threshold/state`
- `GET /api/v1/threshold/breaches`
- `GET /api/v1/threshold/breaches/{breach_id}`

Tests:

- All endpoints require API key.
- Rules endpoint redacts any sensitive context fields.
- State endpoint returns active/pending/clearing states.
- Breach query supports `rule_id`, `sensor_id`, and time range.
- Unknown breach returns `404`.
- Limit is capped.

Exit criteria:

- Operators can inspect Tier 2 behavior without mutating rules.

## Phase 2.11: Replay and Regression Harness

Goal: Prove rule changes before deployment.

Build:

- Historical replay runner.
- Rule output diff tool.

Tests:

- Replay known normal sequence produces no breaches.
- Replay known threshold crossing produces expected breach.
- Replay hot-day temperature sequence produces warning, not critical.
- Replay fire-like sequence produces clustered high-severity context.
- Old vs new rule diff reports added, removed, and changed breaches.

Exit criteria:

- Rule changes can be tested before deployment.

## Phase 2.12: Integration With Ingest Path

Goal: Wire Tier 1 sensor ingest into Tier 2 evaluation.

Build:

- After successful ingest, call `ThresholdGate.evaluate(reading)`.
- Preserve existing ingest behavior if Tier 2 is disabled.
- Log breaches immediately.
- Return optional threshold result in ingest response.

Tests:

- Normal reading ingests with no breach.
- Breaching reading ingests and logs breach.
- Log-only breach does not escalate.
- Tier 2 disabled leaves ingest behavior unchanged.
- History write failure still maps to existing `503`.
- Threshold failure maps to explicit error and does not corrupt history.

Exit criteria:

- Tier 1 and Tier 2 are connected behind tests.

## First Implementation Slice

Start with Phase 2.1 only:

- contracts
- validation
- serialization
- tests

Do not build the evaluator until Phase 2.1 tests pass.

