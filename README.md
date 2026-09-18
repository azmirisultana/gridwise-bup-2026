# BUP CSE Fest 2026 Hackathon — GridWise Energy Optimizer

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![PuLP](https://img.shields.io/badge/Solver-PuLP%20%2F%20COIN--OR%20CBC-orange.svg)](https://coin-or.github.io/pulp/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash%20Lite-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An autonomous, production-grade smart campus energy management service built for the **BUP CSE Fest 2026 GridWise Hackathon (Online Preliminary)**. The service ingests 24-hour campus load forecasts and unstructured natural-language operator logs, extracts operational constraints via Google Gemini (`gemini-3.5-flash-lite`), verifies the directives through strict deterministic schema guardrails, and optimizes the campus battery dispatch schedule via Mixed Integer Linear Programming (MILP) to minimize total electricity procurement cost in Bangladeshi Taka (BDT).

---

## 📋 Hackathon Submission Metadata

- **Track:** GridWise — Smart Campus Energy Optimizer
- **Event:** BUP CSE Fest 2026 Hackathon (Online Preliminary)
- **Live Service URL:** `https://<your-service-name>.onrender.com` *(Update with deployed Render URL)*
- **GitHub Repository:** [https://github.com/azmirisultana/gridwise-bup-2026](https://github.com/azmirisultana/gridwise-bup-2026)
- **3-Minute Demo Video:** [Watch Solution Architecture & Demo Video](https://youtu.be/placeholder) *(Update with unlisted YouTube / Drive link)*

---

## 1. System Architecture

The service strictly adheres to the zero-silent-mutation design pattern mandated by the Participant Guide:

```
                           [Inbound HTTP POST /optimize-energy]
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Request Ingestion & Pydantic Validation (Sections 06 & 07)                         │
│    • Enforces 24 consecutive hours (0..23), valid non-negative demand, solar, tariffs  │
│    • Validates physical battery parameters: capacity, initial SoC, max rates           │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Directive Interpretation via Google GenAI (`gemini-3.5-flash-lite`)                │
│    • Extracts structured directives from 1–3 unstructured natural language notes       │
│    • In-memory LRU cache prevents quota exhaustion on repeated benchmark prompts       │
│    • Distractors and administrative notices correctly classified as `no_op`            │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Strict Deterministic Schema Validator (No Silent Clamping)                          │
│    • Factor bounds check: 0.0 <= factor <= 1.0                                         │
│    • Hour bounds check: unique, sorted ascending integers in range [0, 23]             │
│    • Battery reserve check: 0.0 <= minimum_energy_kwh <= battery.capacity_kwh          │
│    • Grid cap check: max_grid_kwh >= 0.0                                               │
│    • 1-shot self-correcting retry loop with targeted error feedback; fails fast on bad │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. Mixed Integer Linear Program (PuLP / COIN-OR CBC Solver)                           │
│    • Objective: Minimize total daily electricity cost (BDT)                            │
│    • Binary decision variables enforce physical mutual exclusion (Charge vs Discharge) │
│    • Enforces hourly energy balance, battery rate limits, and End-of-Day neutrality    │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Post-Optimization Independent Plan Validator & Recalculator                        │
│    • Verifies hourly energy balance (tolerance <= 0.01 kWh)                            │
│    • Replays all directive windows and confirms zero physical boundary violations      │
│    • Independently recomputes total_grid_kwh, total_cost_bdt, and peak_grid_kwh        │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                             [Canonical JSON Response (200 OK)]
```

---

## 2. Mathematical Formulation (MILP)

The battery dispatch optimization is formulated as a Mixed Integer Linear Program solved using the Coin-OR CBC engine via PuLP.

### Indices & Sets
- $h \in \{0, 1, \dots, 23\}$: Hourly time intervals of the scheduling horizon.

### Input Parameters
- $D_h$: Campus electrical demand at hour $h$ (kWh)
- $S_h$: Solar photovoltaic generation forecast at hour $h$ (kWh)
- $T_h$: Grid electricity tariff at hour $h$ (BDT/kWh)
- $C_{\text{bat}}$: Total battery energy capacity (kWh)
- $E_0$: Initial battery energy stored at start of day (kWh)
- $E_{\text{min}}$: Baseline minimum battery reserve (kWh)
- $R_{\text{chg}}^{\max}$: Maximum charging rate per hour (kWh)
- $R_{\text{dis}}^{\max}$: Maximum discharging rate per hour (kWh)
- $\alpha_h \in [0, 1]$: Directive solar availability factor at hour $h$ (default 1.0)
- $B_{\text{res}, h}$: Directive minimum battery reserve at hour $h$ (default $E_{\text{min}}$)
- $G_{\max, h}$: Directive maximum grid import cap at hour $h$ (default $\infty$)

### Decision Variables
- $g_h \ge 0$: Grid import energy at hour $h$ (kWh)
- $s_h \ge 0$: Usable solar energy consumed at hour $h$ (kWh)
- $c_h \ge 0$: Battery charging energy at hour $h$ (kWh)
- $d_h \ge 0$: Battery discharging energy at hour $h$ (kWh)
- $e_h \ge 0$: Battery energy level after hour $h$ (kWh)
- $u_h \in \{0, 1\}$: Binary variable (1 if charging, 0 if discharging/idle)

### Objective Function
$$\min \sum_{h=0}^{23} g_h \cdot T_h$$

### Constraints
1. **Hourly Campus Energy Balance:**
   $$g_h + s_h + d_h = D_h + c_h \quad \forall h \in \{0, \dots, 23\}$$
2. **Solar Generation & Curtailment Limit:**
   $$0 \le s_h \le \alpha_h \cdot S_h \quad \forall h \in \{0, \dots, 23\}$$
3. **Battery State-of-Charge Dynamics:**
   $$e_0 = E_0 + c_0 - d_0$$
   $$e_h = e_{h-1} + c_h - d_h \quad \forall h \in \{1, \dots, 23\}$$
4. **Battery Energy Boundaries:**
   $$\max(E_{\text{min}}, B_{\text{res}, h}) \le e_h \le C_{\text{bat}} \quad \forall h \in \{0, \dots, 23\}$$
5. **Charge / Discharge Mutual Exclusion:**
   $$0 \le c_h \le R_{\text{chg}}^{\max} \cdot u_h \quad \forall h \in \{0, \dots, 23\}$$
   $$0 \le d_h \le R_{\text{dis}}^{\max} \cdot (1 - u_h) \quad \forall h \in \{0, \dots, 23\}$$
6. **Directive Operational Windows:**
   - If hour $h \in \text{no\_charge\_window}$: $c_h = 0$
   - If hour $h \in \text{no\_discharge\_window}$: $d_h = 0$
   - If hour $h \in \text{max\_grid\_window}$: $g_h \le G_{\max, h}$
7. **End-of-Day Neutrality:**
   $$e_{23} = E_0$$

---

## 3. Supported Directives

| Directive Type | Description | Required Schema Shape |
| :--- | :--- | :--- |
| `solar_reduction` | Temporary degradation or cleaning of solar arrays | `{"hours": [int, ...], "factor": float}` ($0.0 \le \text{factor} \le 1.0$) |
| `minimum_battery_reserve` | Elevated reserve for grid outage protection | `{"hours": [int, ...], "minimum_energy_kwh": float}` |
| `no_charge_window` | Charging prohibited during grid stress or generator sync | `{"hours": [int, ...]}` |
| `no_discharge_window` | Discharging forbidden to conserve capacity | `{"hours": [int, ...]}` |
| `max_grid_window` | Hard threshold on campus grid import | `{"hours": [int, ...], "max_grid_kwh": float}` |
| `no_op` | Irrelevant administrative notes or distractors | `null` (with `applies: false`) |

*Time Convention*: Whole-hour intervals are start-inclusive and end-exclusive (e.g., 1 PM to 3 PM maps to `[13, 14]`).

---

## 4. Benchmark Validation Results

The service was audited against all 10 official public scenarios using `test_solution.py`. Every scenario achieved **0.0000 BDT deviation** against ground-truth benchmarks:

| Scenario ID | Applied Directives | Reference Cost (BDT) | Optimized Cost (BDT) | Difference (BDT) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SAMPLE-01** | `solar_reduction` (12-13, 0.25) | 38,365.00 | 38,365.00 | **0.0000** | PASS |
| **SAMPLE-02** | `minimum_battery_reserve` (18-20, 95 kWh) | 40,785.00 | 40,785.00 | **0.0000** | PASS |
| **SAMPLE-03** | `no_charge_window` (17-21) | 38,825.00 | 38,825.00 | **0.0000** | PASS |
| **SAMPLE-04** | `no_discharge_window` (10-14) | 37,285.00 | 37,285.00 | **0.0000** | PASS |
| **SAMPLE-05** | `max_grid_window` (18-20, 160 kWh) | 37,425.00 | 37,425.00 | **0.0000** | PASS |
| **SAMPLE-06** | `no_op` (distractor) | 37,285.00 | 37,285.00 | **0.0000** | PASS |
| **SAMPLE-07** | `solar_reduction` + `minimum_battery_reserve` | 42,015.00 | 42,015.00 | **0.0000** | PASS |
| **SAMPLE-08** | `no_charge_window` + `max_grid_window` | 38,965.00 | 38,965.00 | **0.0000** | PASS |
| **SAMPLE-09** | `minimum_battery_reserve` + `no_discharge_window` | 40,785.00 | 40,785.00 | **0.0000** | PASS |
| **SAMPLE-10** | `solar_reduction` + `max_grid_window` + `no_op` | 40,045.00 | 40,045.00 | **0.0000** | PASS |

### Additional Quality Suites Passed:
- **Suite 2 (Paraphrase & Edge Directives):** 9/9 passed.
- **Suite 3 (Extreme Physical Infeasible Boundary Checks):** 5/5 passed.
- **Suite 4 (Strict HTTP 400 Rejection on Malformed Inputs):** 5/5 passed.
- **Suite 5 (50 Consecutive Stress Requests):** 50/50 passed (Average Latency: **0.141s**, 0 errors).

---

## 5. Local Setup & Reproduction

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Create Environment
```bash
git clone https://github.com/azmirisultana/gridwise-bup-2026.git
cd gridwise-bup-2026

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```
Edit `.env` with your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
PORT=8000
HOST=0.0.0.0
```

### 3. Run Development Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Verify health:
```bash
curl http://localhost:8000/health
# Response: {"status":"ok"}
```

### 4. Execute Full Verification Suite
```bash
python test_solution.py
```

---

## 6. Docker Deployment

The application is containerized using `python:3.11-slim` with CBC solver installed.

```bash
# Build the Docker image
docker build -t gridwise-backend .

# Run the container
docker run -d -p 8000:8000 \
  -e GEMINI_API_KEY="your_api_key" \
  -e GEMINI_MODEL="gemini-3.5-flash-lite" \
  --name gridwise \
  gridwise-backend

# Health test
curl http://localhost:8000/health
```

---

## 7. Cloud Deployment (Render / Railway)

### Deploying on Render (Free Tier)
1. Log in to [render.com](https://render.com) using your GitHub account.
2. Click **New +** $\rightarrow$ **Web Service**.
3. Select your repository: **`azmirisultana/gridwise-bup-2026`**.
4. Configure service:
   - **Environment:** `Docker`
   - **Plan:** `Free`
5. Under **Environment Variables**, add:
   - `GEMINI_API_KEY`: *(Your Google Gemini API Key)*
   - `GEMINI_MODEL`: `gemini-3.5-flash-lite`
6. Click **Deploy Web Service**.
7. Test the live endpoint:
   ```bash
   curl https://<your-service>.onrender.com/health
   ```

---

## 8. API Reference & cURL Samples

### `GET /health`
Verifies server readiness and returns HTTP 200.
```bash
curl -X GET http://localhost:8000/health
```
```json
{
  "status": "ok"
}
```

### `POST /optimize-energy`
Calculates optimal 24-hour battery schedule under operator directives.
```bash
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "SAMPLE-01",
    "operator_notes": [
      "Facilities will wash the rooftop solar panels from noon until 2 PM. During cleaning, usable solar should be treated as roughly 25% of the forecast.",
      "The sports office moved next month'\''s registration deadline."
    ],
    "hours": [
      {"hour": 0, "demand_kwh": 90, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      {"hour": 1, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      {"hour": 2, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 3, "demand_kwh": 80, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 4, "demand_kwh": 85, "solar_kwh": 0, "tariff_bdt_per_kwh": 5},
      {"hour": 5, "demand_kwh": 95, "solar_kwh": 0, "tariff_bdt_per_kwh": 6},
      {"hour": 6, "demand_kwh": 110, "solar_kwh": 5, "tariff_bdt_per_kwh": 8},
      {"hour": 7, "demand_kwh": 130, "solar_kwh": 20, "tariff_bdt_per_kwh": 10},
      {"hour": 8, "demand_kwh": 150, "solar_kwh": 50, "tariff_bdt_per_kwh": 12},
      {"hour": 9, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
      {"hour": 10, "demand_kwh": 175, "solar_kwh": 130, "tariff_bdt_per_kwh": 16},
      {"hour": 11, "demand_kwh": 180, "solar_kwh": 160, "tariff_bdt_per_kwh": 16},
      {"hour": 12, "demand_kwh": 185, "solar_kwh": 180, "tariff_bdt_per_kwh": 15},
      {"hour": 13, "demand_kwh": 180, "solar_kwh": 170, "tariff_bdt_per_kwh": 14},
      {"hour": 14, "demand_kwh": 170, "solar_kwh": 140, "tariff_bdt_per_kwh": 13},
      {"hour": 15, "demand_kwh": 165, "solar_kwh": 90, "tariff_bdt_per_kwh": 14},
      {"hour": 16, "demand_kwh": 170, "solar_kwh": 45, "tariff_bdt_per_kwh": 18},
      {"hour": 17, "demand_kwh": 185, "solar_kwh": 10, "tariff_bdt_per_kwh": 22},
      {"hour": 18, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 28},
      {"hour": 19, "demand_kwh": 215, "solar_kwh": 0, "tariff_bdt_per_kwh": 30},
      {"hour": 20, "demand_kwh": 205, "solar_kwh": 0, "tariff_bdt_per_kwh": 26},
      {"hour": 21, "demand_kwh": 175, "solar_kwh": 0, "tariff_bdt_per_kwh": 18},
      {"hour": 22, "demand_kwh": 135, "solar_kwh": 0, "tariff_bdt_per_kwh": 10},
      {"hour": 23, "demand_kwh": 105, "solar_kwh": 0, "tariff_bdt_per_kwh": 7}
    ],
    "battery": {
      "capacity_kwh": 220,
      "initial_energy_kwh": 110,
      "minimum_energy_kwh": 40,
      "max_charge_kwh_per_hour": 50,
      "max_discharge_kwh_per_hour": 50
    }
  }'
```

**Response Format (HTTP 200)**:
```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {
      "note_index": 0,
      "applies": true,
      "directive_type": "solar_reduction",
      "structured_adjustment": {
        "hours": [12, 13],
        "factor": 0.25
      },
      "explanation": "Solar output is reduced during hours [12, 13] with factor 0.25."
    },
    {
      "note_index": 1,
      "applies": false,
      "directive_type": "no_op",
      "structured_adjustment": null,
      "explanation": "This note is administrative or does not affect the current 24-hour energy schedule."
    }
  ],
  "hourly_plan": [
    {
      "hour": 0,
      "grid_kwh": 90.0,
      "solar_used_kwh": 0.0,
      "battery_action": "idle",
      "battery_kwh": 0.0,
      "battery_energy_after_kwh": 110.0
    }
    // ... remaining 23 hours
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Optimized schedule across 24 hours under tariffs ranging 5.0-30.0 BDT/kWh. Directives applied: solar_reduction. Battery restored to initial level of 110.0 kWh at end of day."
}
```

---

## 9. Security & Error Handling

- **Zero Secret Leakage**: The `.env` file is excluded via `.gitignore`. Error handlers strip all stack traces, keys, and internal solver paths before serializing HTTP 400, 422, or 500 responses.
- **Fail-Safe Self-Correction**: When LLM output fails schema validation, the system executes a 1-shot self-correcting prompt citing the exact invalid field. If validation still fails, it raises a controlled HTTP error rather than solving an unverified plan.
- **Rate-Limit Resilience**: An LRU cache stores directive extractions by hash of the note sequence, protecting the endpoint against Gemini Free Tier rate exhaustion (15 RPM).

---

## 10. Evaluation Rubric Compliance Checklist

- [x] **`GET /health`**: Responds with `{"status":"ok"}` under 10ms.
- [x] **`POST /optimize-energy`**: Full canonical JSON format strictly conforming to Section 07.
- [x] **Mandatory LLM Interpretation**: Uses official `google-genai` SDK with `gemini-3.5-flash-lite`.
- [x] **No Silent Clamping**: Rejects illegal values or repairs via self-correction; never mutates constraints silently.
- [x] **Mathematical Optimality**: 100% cost parity ($\le 0.01$ BDT tolerance) on all 10 official benchmark scenarios.
- [x] **Independent Final-Plan Validator**: Guarantees zero physical equation violations before client response.
- [x] **Clean Repository & Dockerfile**: Fully reproducible container build and clear documentation.
