# Tier 2 Threshold Gate Design

Tier 2 is Lyr's deterministic breach-detection layer. It evaluates normalized
`SensorReading` objects from Tier 1 against declarative rules and emits breach
events only when a rule is satisfied. It does not call AI, infer intent, or take
actions.

## Decisions

- Rules are declarative only. No `eval()`, embedded Python, or expression strings.
- Rules are file-backed for Phase 2. Git is the versioning and rollback system.
- Evaluation is event-driven: each incoming reading is evaluated immediately.
- Rule correctness is order-independent. Rules may later be sorted by evaluation
  cost, but outcomes must not depend on configuration order.
- Cross-sensor rules live inside the same rule engine as single-sensor rules.
- Rule confirmation is time-based, not count-based. A breach must be sustained
  for `sustained_for_seconds`, because sensor cadences differ.
- Hysteresis is represented by one asymmetric pair: `enter_threshold` and
  `clear_threshold`. There is no separate deadband knob in Phase 2.
- Rules may include deterministic context gates. Environmental data, nearby
  sensors, or known operating conditions can adjust the breach decision before
  escalation, but Tier 2 still does not use AI.
- Missing or stale related sensor data is fail-closed-but-bounded: it produces an
  explicit lower-severity breach instead of silently passing or escalating as if
  the primary rule fired.
- Simultaneous breaches should be clustered into one `BreachContext` for Tier 3,
  not sent as independent AI calls.
- Log-only mode is first-class. Rules can record breaches without escalating.
- Breach state and breach logs are SQLite-backed from Phase 2. This keeps memory
  bounded, survives restarts, and matches the Tier 1 `HistoryStore` pattern.

## Context-Aware Thresholds

A threshold crossing is not always an anomaly. Temperature is the clearest
example: `35C` indoors on a normal day may be serious, while a high ambient
temperature during a very hot day may require a different interpretation.

Tier 2 should reduce false positives by aggregating deterministic context before
declaring a breach official:

- current reading from the primary sensor
- recent history from the same sensor
- nearby or related sensors such as humidity, smoke, light, motion, or power
- environmental feeds such as outdoor weather, heat index, or known ambient
  baseline
- sensor health and staleness

This context can change the severity or suppress escalation, but it must remain
explainable. The breach reason should say exactly which rule and context gate
were used, for example:

```text
temperature_high crossed 36C for 10s, but outdoor_heat_index is high and
nearby smoke/humidity rules did not trigger; logged as warning.
```

Dire situations increase viability. A temperature threshold plus smoke, humidity
drop, power anomaly, or motion anomaly should escalate faster or at higher
severity than temperature alone.

## Trigger Timing

Tier 2 is event-driven. Evaluation starts as soon as a reading arrives.

How quickly a breach triggers depends on rule configuration:

- `sustained_for_seconds = 0`: trigger immediately on threshold crossing
- `sustained_for_seconds = 5`: trigger only after the condition remains true for
  at least 5 seconds
- cross-sensor context may add one history lookup, but it should not wait for AI

Fast-danger rules should use low sustain times. Example: smoke detector breach
may trigger immediately, while temperature alone may require 10-30 seconds plus
context checks to reduce false positives.

## State

Tier 2 is stateful. It must remember:

- active breach state per `rule_id` and `sensor_id`
- first trigger timestamp
- last matched timestamp
- clear timestamp
- whether the rule is in log-only mode

This state should live in SQLite from the start, not only in memory. The state
table should be small, overwritten as state changes, and cleaned up when breaches
clear. This avoids trampling system memory while preserving enough continuity
after restart to avoid losing active or clearing breaches.

Suggested location:

```text
recognition/database/threshold_state.db
```

Suggested columns:

- `state_key` (`rule_id:sensor_id`)
- `rule_id`
- `sensor_ids_json`
- `status` (`idle`, `pending_sustain`, `active`, `clearing`, `cleared`)
- `first_triggered_at`
- `last_triggered_at`
- `clear_started_at`
- `cleared_at`
- `rule_snapshot_json`
- `updated_at`

## Breach Log

Log-only mode requires a breach log, not just transient context. Phase 2 should
include a SQLite-backed append-only breach store with:

- `breach_id`
- `rule_id`
- primary sensor ID
- related sensor IDs
- `triggered_at`
- `cleared_at`
- `severity`
- context snapshot JSON
- `escalated_to_tier3`
- `tier3_decision_json`
- `action_taken_json`
- `human_reviewed`
- `human_notes`

Suggested location:

```text
recognition/database/breach_log.db
```

The log may be capped or compacted over time, but it should not be purely
in-memory. It is part of the audit trail.

## Boundary Example

For a temperature rule:

- `enter_threshold = 36.0`
- `clear_threshold = 34.0`
- `sustained_for_seconds = 10`
- `clear_delay_seconds = 3`

Expected behavior:

- values below or equal to 36.0 do not enter breach
- values above 36.0 start candidate breach timing
- dips between 34.0 and 36.0 do not clear an active breach
- values below 34.0 start the clear timer
- values below 34.0 for at least 3 seconds clear the breach
- a value above 36.0 for less than 10 seconds remains a candidate, not active

## Open Phase 2 Choices

- Rule format: YAML is preferred for readability; JSON remains acceptable if the
  repo avoids adding YAML dependencies.
- Rule location: `recognition/rules/`.
- Rule deployment: Git commit plus restart for Phase 2.
- Initial rules: start with temperature, humidity, battery, and face-presence
  projection only after the vision refactor is safely projected into Tier 1.
