# Tier 2 Multi-Evidence Evaluation

Tier 2.7 evaluates one triggering reading against a rule that can require
other evidence sources. Evidence can corroborate, contradict, or contextualize
the primary threshold event.

## Rule Shape

The primary condition is still defined by:

- `sensor_type`
- `enter_threshold`
- `clear_threshold`

Additional evidence is declared in `conditions`:

```json
{
  "sensor_type": "rainfall",
  "operator": ">=",
  "threshold": 30.0,
  "history_window_seconds": 600,
  "required": true,
  "role": "corroborates",
  "weight": 0.25
}
```

Roles:

- `corroborates`: matched evidence raises confidence.
- `contradicts`: matched evidence lowers confidence.
- `context`: matched evidence is recorded but does not directly alter confidence.

## Evidence Snapshot

For each condition, Tier 2 records:

- `matched`
- `not_matched`
- `missing`
- `stale`

Snapshots include the reading value, timestamp, age, threshold, operator,
required flag, role, and weight. This is the explanation surface for humans,
Tier 3, and the optional AI advisory layer.

## Staleness Policy

`alert_stale`:
Required missing/stale evidence creates a lower-severity log-only stale alert.

`fail_closed`:
Required missing/stale evidence suppresses evaluation.

`fail_open`:
Required missing/stale evidence continues with available data.

Optional missing/stale evidence never suppresses evaluation.

## Confidence

Confidence is deterministic:

```text
base_confidence
+ matched corroborating weights
- matched contradicting weights
- missing/stale required penalties
clamped to max_confidence
```

Every contribution is stored in `cross_sensor_evaluation.confidence`.

## AI Advisory Boundary

AI advisory runs only after deterministic Tier 2 produces a breach state.
It may annotate:

- summary
- risk level
- recommended action
- raw response
- model metadata

It may not:

- create a breach
- suppress a breach
- clear a breach
- mutate deterministic confidence

Live Ollama tests are opt-in through `LYR_RUN_OLLAMA_TESTS=1`.

## Replay Fixtures

Replay fixtures live under `tests/fixtures/replay` and cover:

- fire
- flash flood
- heatwave context
- false smoke
- missing sensor
- breach clearing
