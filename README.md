# Lyr

Lyr is a cyber-physical perception and decision engine—an "Alpha Watcher" that ingests heterogeneous sensor data, reasons about anomalies with contextual awareness, and takes calibrated action: from silent logging, to automated countermeasures, to hard alerts for a human.

The system is built on a simple observation: most monitoring systems escalate on every fluctuation and train their operators to ignore them. Lyr takes the opposite approach. Its default state is watching—it only spends compute on deep reasoning when a reading actually breaches a threshold, and it only acts autonomously when it's confident the action is safe.

## Architecture

| Tier | Name | State | Job |
|------|------|-------|-----|
| 1 | Data nodes | Ingesting | Normalize raw readings into a `SensorReading` contract |
| 2 | Threshold gate | Watching | Deterministic rule check against baselines |
| 3 | AI reasoner | Analyzing | Pulls historical + cross-sensor context to explain *why* a threshold was breached |
| 4 | Action router | Escalating | Executes a safe countermeasure, or fires a hard alert with the AI's hypothesis attached |

Any sensor—vision, environmental, motion, audio—can plug in as a Tier 1 node. The system is sensor-agnostic by design.

## Capabilities

- Multi-modal sensor ingestion with self-validating data contracts
- Real-time threshold evaluation with hysteresis and deadband logic
- Cross-sensor correlation for contextual anomaly detection
- AI-powered reasoning with explainable confidence scoring
- Autonomous action execution with explicit safe-action allowlist
- Human-in-the-loop escalation for high-risk decisions
- Full audit trail with decision-chain reconstruction

## Deployment

```bash
python recognition/run_api.py
The system serves a WebSocket-enabled live dashboard and REST API for integration with external tools.

Design Principles
Fail toward hard alert. When confidence is uncertain, the system defaults to human attention, not autonomous action.

Confidence is explicit. Every reading carries a confidence score (0–1). Autonomous action requires ≥0.95.

Sensors are nodes, not special cases. Vision, temperature, motion, audio—all emit the same SensorReading contract and are treated identically.

The audit log is the source of truth. Every decision chain can be reconstructed from the logs.

Why Lyr
Lyr is an exploration of what happens when you give a system persistent, ambient awareness of its environment. It's not reactive—it watches continuously, builds a model of normal over time, and only acts when it's sure something has changed.

It solves a fundamental problem: how do you know when to pay attention? Most systems assume you already know. Lyr assumes you don't, and watches until it does.
