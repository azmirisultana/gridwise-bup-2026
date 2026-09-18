# BUP CSE Fest 2026 Hackathon — GridWise Energy Optimizer

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![PuLP](https://img.shields.io/badge/Solver-PuLP%20%2F%20CBC-orange.svg)](https://coin-or.github.io/pulp/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini%203.5%20Flash%20Lite-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)

An autonomous, production-grade backend service built for the **BUP CSE Fest 2026 GridWise Hackathon (Online Preliminary)**. The service receives 24-hour smart campus energy scenarios alongside natural-language operator notes, interprets temporary operational constraints using Google GenAI (Gemini), enforces strict deterministic validation, and optimizes the 24-hour campus energy schedule via Mixed Integer Linear Programming (MILP) to minimize grid electricity costs in Bangladeshi Taka (BDT).

---

## 🎥 3-Minute Architecture & Demo Video

> **Demo Video Link**: [Watch Solution Architecture & Testing Video](https://youtu.be/placeholder)  
> *(Please replace with your unlisted YouTube or Google Drive video URL before submitting)*

---

## 1. System Architecture

The service strictly implements the multi-stage pipeline mandated by the problem specification:

```
                  [Campus Scenario & Operator Notes]
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Request Ingestion & Pydantic Validation (Section 06 & 07)           │
│    • Strict types, 24 consecutive hours (0..23), boundary checks       │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. LLM Directive Interpretation (google-genai / Gemini 3.5 Flash Lite) │
│    • Interprets 1-3 natural-language notes into structured JSON        │
│    • Identifies distractors and administrative notes as no_op          │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. Strict Deterministic Validation & Retry Loop                        │
│    • Enforces exact schema conformity without silent mutation          │
│    • Validates factor in [0.0, 1.0], unique ascending hours in [0, 23] │
│    • Validates reserve in [0, capacity], max_grid >= 0                 │
│    • 1-shot retry with error feedback; rejects invalid semantics       │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. PuLP Linear Programming Optimizer (Sections 05 & 09)                │
│    • Minimizes: total_cost_bdt = SUM(grid_kwh[h] * tariff_bdt[h])      │
│    • Enforces: Energy balance, solar curtailment, battery dynamics,    │
│      charge/discharge limits, directive windows, and EOD neutrality    │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. Independent Final-Plan Validator & Recalculator                     │
│    • Validates full physical feasibility and directive replay          │
│    • Confirms mutual exclusion: charge and discharge never co-occur    │
│    • Independently recalculates total_grid, total_cost, and peak_grid  │
└────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                     [Validated JSON Response]
```

---

## 2. Supported Directives

| Directive Type | Description | Required `structured_adjustment` Shape |
| :--- | :--- | :--- |
| `solar_reduction` | Usable solar output reduced during specific hours. | `{"hours": [int, ...], "factor": float}` (usable fraction $0 \le factor \le 1$) |
| `minimum_battery_reserve` | Keep battery energy at or above a required level. | `{"hours": [int, ...], "minimum_energy_kwh": float}` |
| `no_charge_window` | Battery charging is forbidden during specific hours. | `{"hours": [int, ...]}` |
| `no_discharge_window` | Battery discharging is forbidden during specific hours. | `{"hours": [int, ...]}` |
| `max_grid_window` | Grid import may not exceed stated kWh during specific hours. | `{"hours": [int, ...], "max_grid_kwh": float}` |
| `no_op` | Distractor or irrelevant note with no impact on schedule. | `null` (with `applies: false`) |

*Time Convention*: Whole-hour intervals are start-inclusive and end-exclusive (e.g. 1 PM to 3 PM maps to `[13, 14]`).

---

## 3. Environment Variables

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | String | *None* | Google Gemini API key for `google-genai` SDK. |
| `GEMINI_MODEL` | String | `gemini-3.5-flash-lite` | Gemini model identifier. |
| `PORT` | Integer | `8000` | HTTP service listening port. |
| `HOST` | String | `0.0.0.0` | Bind address. |
| `TEST_BASE_URL` | String | *None* | Optional live URL for `test_solution.py` (e.g. `http://localhost:8000`). |

> [!CAUTION]
> **Secret Handling:** Never commit your `GEMINI_API_KEY` or `.env` files to the repository. The application reads keys strictly from process environment variables and suppresses credentials and stack traces from API logs and error responses.

---

## 4. Local Quickstart

### Prerequisites
- Python 3.11 or higher
- Git

### 1. Clone & Set Up Environment
```bash
git clone <your-repository-url>
cd bup-hackathon-solution

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
```bash
# Windows PowerShell:
$env:GEMINI_API_KEY="your_api_key_here"

# Linux / macOS:
export GEMINI_API_KEY="your_api_key_here"
```
*(Note: If `GEMINI_API_KEY` is not provided, the built-in deterministic fallback engine automatically handles operator notes, ensuring tests run seamlessly in offline or non-credentialed environments.)*

### 3. Start the API Service
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
The service is now ready on `http://localhost:8000`.

---

## 5. Running the Test Suite

The solution comes with an automated test suite (`test_solution.py`) that loads `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json` and evaluates all 10 official sample cases:

```bash
python test_solution.py
```

### Verification Checks Performed:
1. **`GET /health`**: Confirms readiness and returns `{"status": "ok"}`.
2. **`directive_interpretation`**: Verifies exact `directive_type`, `note_index` ordering, boolean `applies`, and `structured_adjustment` (hours, factor, reserve, grid cap).
3. **`hourly_plan` Constraint Correctness**:
   - Hourly energy balance: $grid\_kwh + solar\_used\_kwh + discharge = demand\_kwh + charge$.
   - Solar limit: $solar\_used\_kwh \le effective\_solar$.
   - Battery bounds: $active\_min\_reserve \le battery\_energy\_after\_kwh \le capacity$.
   - End-of-day battery neutrality: $battery\_energy\_after\_kwh[23] == initial\_energy\_kwh$.
4. **Cost Tolerance**: Verifies recalculated `total_cost_bdt` matches the reference optimal cost within **0.01 BDT tolerance**.

---

## 6. Docker Instructions

A multi-stage container configuration based on `python:3.11-slim` is provided.

### Build Docker Image
```bash
docker build -t gridwise-backend .
```

### Run Docker Container
```bash
docker run -d -p 8000:8000 --name gridwise -e GEMINI_API_KEY="your_api_key" gridwise-backend
```

### Test Container Health
```bash
curl http://localhost:8000/health
```

---

## 7. cURL API Examples

### Health Endpoint
```bash
curl -X GET http://localhost:8000/health
```
**Response (HTTP 200)**:
```json
{
  "status": "ok"
}
```

### Optimize Energy Endpoint
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
    // ... 23 more hourly entries
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Optimized schedule across 24 hours under tariffs ranging 5.0-30.0 BDT/kWh. Directives applied: solar_reduction. Battery restored to initial level of 110.0 kWh at end of day."
}
```

---

## 8. HTTP Response Codes

| Status Code | Meaning | Handling |
| :--- | :--- | :--- |
| `200 OK` | Successful health or optimization response. | Returns canonical JSON schemas. |
| `400 Bad Request` | Malformed JSON or structurally invalid request. | Trapped via custom exception handler. |
| `422 Unprocessable Entity` | Pydantic validation error (e.g. missing fields, invalid ranges). | Auto-handled with field details. |
| `500 Internal Server Error` | Controlled internal error. | Returns generic error message without exposing secrets or stack traces. |

---

## 9. Technology Stack & Credits

- **FastAPI** (`0.115+`): High-performance modern web framework.
- **Uvicorn** (`0.34+`): ASGI production server.
- **Pydantic** (`2.10+`): Strict data validation and schema enforcement.
- **PuLP & COIN-OR CBC**: Mixed Integer Linear Programming formulation and solving.
- **Google GenAI SDK (`google-genai`)**: Official SDK for Google Gemini models.
- **Requests**: HTTP client for testing and verification.

---

## 10. Submission Verification Checklist

- [x] `GET /health` responds with `{"status": "ok"}` within milliseconds.
- [x] `POST /optimize-energy` accepts 1-3 operator notes and returns valid structured output.
- [x] Interpretation coverage: Exactly one entry per note in `note_index` order (0..N-1).
- [x] Strict validation active: Validates hours (unique, ascending 0..23), rejects out-of-bounds factors, strictly checks battery reserves.
- [x] Independent plan validator: Replays all physical and directive constraints against returned schedule.
- [x] LP Optimizer respects all physical constraints: hourly balance, battery limits, rates, EOD neutrality.
- [x] Public sample test suite: All 10 sample scenarios pass within 0.01 BDT tolerance.
- [x] Dockerfile containerizes application cleanly without baked-in secrets.
- [x] Clean documentation covering architecture, setup, environment variables, and curl examples.
