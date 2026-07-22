# Lyr

**Lyr** is a cyber-physical perception and decision engine — an "Alpha Watcher" that
ingests heterogeneous sensor data (environmental, structural, visual, and beyond),
reasons about anomalies with contextual awareness, and takes calibrated action:
from silent logging, to automated countermeasures, to hard alerts for a human.

Lyr is built to be unhurried. Most systems escalate on every fluctuation and train
their operators to ignore them. Lyr's default state is watching — it only spends
compute on deep reasoning when a reading actually breaches a threshold, and it
only acts autonomously when it's confident the action is safe.

## Architecture — the four tiers


| Tier | Name | State | Compute | Job |
|------|------|-------|---------|-----|
| 1 | Data nodes | Ingesting | very low | Normalize raw readings (temp, water, vision, etc.) into a `SensorReading` contract |
| 2 | Threshold gate | Watching | low | Deterministic rule check against baselines. Normal → loop back, no AI involved |
| 3 | AI reasoner | Analyzing | high | Pulls historical + cross-sensor context to explain *why* a threshold was breached |
| 4 | Action router | Escalating | low | Executes a safe countermeasure, or fires a hard alert with the AI's hypothesis attached |

The vision/face-recognition module is one Tier 1 plugin among many possible sensor
types — not a special case. Any node that can emit a `SensorReading` can plug in.

## Current status

**Built (Tier 1, vision plugin only):**
- Face detection → liveness check → embedding → identification pipeline
- Quality-weighted multi-frame enrollment, EMA-based re-learning
- Producer/consumer camera loop, FastAPI + WebSocket live dashboard
- Self-validating data contracts (`contracts.py`) — the pattern the generic
  `SensorReading` contract will be built on

**Not yet built:**
- Generic `SensorReading` contract + sensor type registry (Tier 1, generalized)
- Cross-sensor correlation / threshold rules engine (Tier 2, generalized —
  today's `correlation_gate` only correlates frames, not arbitrary sensors)
- AI reasoning layer (Tier 3)
- Action router / countermeasure execution + hard alert dispatch (Tier 4)
- Time-series storage for rolling sensor history
- API authentication (currently **none** — open endpoints, must fix before any
  real deployment)

## Running it today
python recognition/run_api.py

text

Serves the face-recognition API + dashboard. (No `requirements.txt` was found in
review — pin dependencies before this leaves prototype stage.)

## Roadmap

1. Generalize `contracts.py` → `SensorReading` + sensor registry
2. Rebuild the threshold gate as a generic rules engine, multi-sensor aware
3. Stand up a rolling time-series store (readings history for Tier 3 context)
4. Build the AI reasoning layer (model choice: local SLM vs. cloud LLM — open question)
5. Build the action router with an explicit safe-action allowlist + hard-alert fallback
6. Add API authentication
7. Wire the face module in as a Tier 1 plugin under the new contract
