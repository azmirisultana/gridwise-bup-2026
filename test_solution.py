import os
import sys
import json
import time
import logging
import requests
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("judge")

# =====================================================================
# Independent Miniature Judge (Zero shared logic with main.py)
# =====================================================================

class IndependentJudge:
    """
    Independent audit validator for energy optimization responses.
    Verifies Section 05, 06, 07, 08, 09, 10, 11 constraints directly.
    """

    @staticmethod
    def audit_response(payload: Dict[str, Any], response_json: Dict[str, Any]) -> tuple[bool, str]:
        # 1. Top-level contract check
        required_keys = [
            "scenario_id", "directive_interpretation", "hourly_plan",
            "total_grid_kwh", "total_cost_bdt", "peak_grid_kwh", "plan_summary"
        ]
        for k in required_keys:
            if k not in response_json:
                return False, f"Missing required top-level key '{k}' in response"

        if response_json["scenario_id"] != payload["scenario_id"]:
            return False, f"scenario_id mismatch: expected {payload['scenario_id']}, got {response_json['scenario_id']}"

        # 2. Directive interpretation check
        directives = response_json["directive_interpretation"]
        notes = payload.get("operator_notes", [])
        if len(directives) != len(notes):
            return False, f"Directive count {len(directives)} != operator_notes count {len(notes)}"

        valid_types = {
            "solar_reduction", "minimum_battery_reserve", "no_charge_window",
            "no_discharge_window", "max_grid_window", "no_op"
        }

        seen_indices = set()
        for idx, d in enumerate(directives):
            n_idx = d.get("note_index")
            if n_idx is None or n_idx in seen_indices or n_idx < 0 or n_idx >= len(notes):
                return False, f"Invalid or duplicate note_index {n_idx} at directive {idx}"
            seen_indices.add(n_idx)

            dtype = d.get("directive_type")
            if dtype not in valid_types:
                return False, f"Invalid directive_type '{dtype}'"

            applies = d.get("applies")
            adj = d.get("structured_adjustment")

            if dtype == "no_op":
                if applies is not False:
                    return False, f"no_op directive must have applies=False"
                if adj is not None:
                    return False, f"no_op directive must have structured_adjustment=null"
            else:
                if applies is not True:
                    return False, f"Directive '{dtype}' must have applies=True"
                if not adj or not isinstance(adj, dict):
                    return False, f"Directive '{dtype}' missing structured_adjustment dictionary"

                hours = adj.get("hours")
                if not hours or not isinstance(hours, list):
                    return False, f"Directive '{dtype}' missing or invalid hours list"
                if any(not isinstance(h, int) or h < 0 or h > 23 for h in hours):
                    return False, f"Directive '{dtype}' contains invalid hour outside [0, 23]"
                if len(hours) != len(set(hours)):
                    return False, f"Directive '{dtype}' contains duplicate hours: {hours}"
                if hours != sorted(hours):
                    return False, f"Directive '{dtype}' hours must be strictly sorted: {hours}"

                if dtype == "solar_reduction":
                    factor = adj.get("factor")
                    if factor is None or not (0.0 <= factor <= 1.0):
                        return False, f"solar_reduction factor {factor} outside [0.0, 1.0]"
                elif dtype == "minimum_battery_reserve":
                    min_e = adj.get("minimum_energy_kwh")
                    if min_e is None or min_e < 0.0 or min_e > payload["battery"]["capacity_kwh"]:
                        return False, f"minimum_energy_kwh {min_e} outside [0.0, capacity]"
                elif dtype == "max_grid_window":
                    grid_cap = adj.get("max_grid_kwh")
                    if grid_cap is None or grid_cap < 0.0:
                        return False, f"max_grid_kwh {grid_cap} cannot be negative"

        # 3. Hourly plan physical audit
        plan = response_json["hourly_plan"]
        if len(plan) != 24:
            return False, f"Hourly plan length {len(plan)} != 24"

        hours_input = {h["hour"]: h for h in payload["hours"]}
        battery = payload["battery"]

        # Replay directives
        eff_solar = {h: hours_input[h]["solar_kwh"] for h in range(24)}
        active_min_reserve = {h: battery["minimum_energy_kwh"] for h in range(24)}
        no_charge_hours = set()
        no_discharge_hours = set()
        max_grid_caps = {h: float("inf") for h in range(24)}

        for d in directives:
            if not d["applies"] or not d.get("structured_adjustment"):
                continue
            adj = d["structured_adjustment"]
            d_hours = adj["hours"]
            if d["directive_type"] == "solar_reduction":
                for h in d_hours:
                    eff_solar[h] *= adj["factor"]
            elif d["directive_type"] == "minimum_battery_reserve":
                for h in d_hours:
                    active_min_reserve[h] = max(active_min_reserve[h], adj["minimum_energy_kwh"])
            elif d["directive_type"] == "no_charge_window":
                no_charge_hours.update(d_hours)
            elif d["directive_type"] == "no_discharge_window":
                no_discharge_hours.update(d_hours)
            elif d["directive_type"] == "max_grid_window":
                for h in d_hours:
                    max_grid_caps[h] = min(max_grid_caps[h], adj["max_grid_kwh"])

        prev_energy = battery["initial_energy_kwh"]
        for exp_h in range(24):
            entry = plan[exp_h]
            h = entry.get("hour")
            if h != exp_h:
                return False, f"Hour index mismatch: expected {exp_h}, got {h}"

            grid = entry.get("grid_kwh", 0.0)
            solar_used = entry.get("solar_used_kwh", 0.0)
            action = entry.get("battery_action")
            b_kwh = entry.get("battery_kwh", 0.0)
            e_after = entry.get("battery_energy_after_kwh", 0.0)
            demand = hours_input[h]["demand_kwh"]

            if action not in {"charge", "discharge", "idle"}:
                return False, f"Hour {h}: invalid battery_action '{action}'"

            chg = b_kwh if action == "charge" else 0.0
            disch = b_kwh if action == "discharge" else 0.0

            if action == "idle" and b_kwh > 1e-4:
                return False, f"Hour {h}: action is idle but battery_kwh is {b_kwh}"

            if chg > 1e-4 and disch > 1e-4:
                return False, f"Hour {h}: simultaneous charge and discharge"

            # Rate limits
            if chg > battery["max_charge_kwh_per_hour"] + 0.02:
                return False, f"Hour {h}: charge {chg} exceeds max_charge {battery['max_charge_kwh_per_hour']}"
            if disch > battery["max_discharge_kwh_per_hour"] + 0.02:
                return False, f"Hour {h}: discharge {disch} exceeds max_discharge {battery['max_discharge_kwh_per_hour']}"

            # Directive limits
            if h in no_charge_hours and chg > 1e-3:
                return False, f"Hour {h}: charged {chg} during no_charge_window"
            if h in no_discharge_hours and disch > 1e-3:
                return False, f"Hour {h}: discharged {disch} during no_discharge_window"
            if grid > max_grid_caps[h] + 0.02:
                return False, f"Hour {h}: grid {grid} exceeds max_grid cap {max_grid_caps[h]}"
            if solar_used > eff_solar[h] + 0.02:
                return False, f"Hour {h}: solar_used {solar_used} exceeds available {eff_solar[h]}"

            # Energy balance: grid + solar + discharge == demand + charge
            bal_err = abs((grid + solar_used + disch) - (demand + chg))
            if bal_err > 0.02:
                return False, f"Hour {h}: energy balance error {bal_err:.4f} exceeds 0.02 tolerance"

            # Battery state transition
            expected_e = prev_energy + chg - disch
            if abs(e_after - expected_e) > 0.02:
                return False, f"Hour {h}: battery transition error: got {e_after}, expected {expected_e}"

            # Reserve and capacity limits
            if e_after < active_min_reserve[h] - 0.02:
                return False, f"Hour {h}: battery level {e_after} below active reserve {active_min_reserve[h]}"
            if e_after > battery["capacity_kwh"] + 0.02:
                return False, f"Hour {h}: battery level {e_after} exceeds capacity {battery['capacity_kwh']}"

            prev_energy = e_after

        # End of day neutrality
        if abs(plan[23]["battery_energy_after_kwh"] - battery["initial_energy_kwh"]) > 0.02:
            return False, (
                f"End of day neutrality violated: battery ended at {plan[23]['battery_energy_after_kwh']}, "
                f"initial was {battery['initial_energy_kwh']}"
            )

        # 4. Independent recalculation of metrics
        calc_grid = round(sum(p["grid_kwh"] for p in plan), 4)
        calc_cost = round(sum(p["grid_kwh"] * hours_input[p["hour"]]["tariff_bdt_per_kwh"] for p in plan), 4)
        calc_peak = round(max(p["grid_kwh"] for p in plan), 4)

        if abs(calc_grid - response_json["total_grid_kwh"]) > 0.05:
            return False, f"total_grid_kwh mismatch: reported {response_json['total_grid_kwh']}, calculated {calc_grid}"
        if abs(calc_cost - response_json["total_cost_bdt"]) > 0.05:
            return False, f"total_cost_bdt mismatch: reported {response_json['total_cost_bdt']}, calculated {calc_cost}"
        if abs(calc_peak - response_json["peak_grid_kwh"]) > 0.05:
            return False, f"peak_grid_kwh mismatch: reported {response_json['peak_grid_kwh']}, calculated {calc_peak}"

        return True, "Audit passed successfully"


# =====================================================================
# Test Client Setup
# =====================================================================

def get_base_url() -> str:
    url = os.environ.get("TEST_BASE_URL", "http://127.0.0.1:8000")
    try:
        r = requests.get(f"{url}/health", timeout=2.0)
        if r.status_code == 200:
            logger.info(f"Targeting active server at {url}")
            return url
    except Exception as e:
        logger.error(f"Cannot reach server at {url}: {e}")
    return url


# =====================================================================
# Helper: Base 24-Hour Schedule Generator
# =====================================================================

def generate_base_scenario(scenario_id: str, operator_notes: List[str]) -> Dict[str, Any]:
    """Generates a standard 24-hour campus scenario."""
    hours = []
    for h in range(24):
        # Base solar curve peaks at midday
        solar = 50.0 * max(0.0, 1.0 - abs(h - 12) / 6.0) if 6 <= h <= 18 else 0.0
        # Typical campus demand
        demand = 40.0 if 8 <= h <= 18 else 15.0
        # Tariff: cheap off-peak (5 BDT), peak evenings (12 BDT), normal day (8 BDT)
        tariff = 12.0 if 17 <= h <= 21 else (5.0 if h < 6 or h > 22 else 8.0)
        hours.append({
            "hour": h,
            "demand_kwh": round(demand, 2),
            "solar_kwh": round(solar, 2),
            "tariff_bdt_per_kwh": round(tariff, 2)
        })

    return {
        "scenario_id": scenario_id,
        "operator_notes": operator_notes,
        "hours": hours,
        "battery": {
            "capacity_kwh": 100.0,
            "initial_energy_kwh": 50.0,
            "minimum_energy_kwh": 20.0,
            "max_charge_kwh_per_hour": 25.0,
            "max_discharge_kwh_per_hour": 25.0
        }
    }


# =====================================================================
# Test Suites
# =====================================================================

def run_suite_public_samples(base_url: str) -> int:
    """Suite 1: All 10 Public Sample Cases against Independent Judge."""
    logger.info("=== RUNNING SUITE 1: 10 Public Sample Cases ===")
    sample_file = "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
    if not os.path.exists(sample_file):
        sample_file = os.path.join(os.path.dirname(__file__), sample_file)

    with open(sample_file, "r", encoding="utf-8") as f:
        cases = json.load(f).get("cases", [])

    failures = 0
    for idx, case in enumerate(cases):
        cid = case["id"]
        payload = case["input"]
        expected = case["expected_output"]

        resp = requests.post(f"{base_url}/optimize-energy", json=payload, timeout=25.0)
        if resp.status_code != 200:
            logger.error(f"Case {cid} FAILED: HTTP {resp.status_code} - {resp.text}")
            failures += 1
            continue

        resp_json = resp.json()
        passed, msg = IndependentJudge.audit_response(payload, resp_json)
        if not passed:
            logger.error(f"Case {cid} Audit Violation: {msg}")
            failures += 1
            continue

        cost_diff = abs(resp_json["total_cost_bdt"] - expected["total_cost_bdt"])
        if cost_diff > 0.05:
            logger.error(f"Case {cid} Cost Mismatch: actual={resp_json['total_cost_bdt']}, expected={expected['total_cost_bdt']}")
            failures += 1
            continue

        logger.info(f"✓ Case {idx+1}/10 ({cid}) PASSED [Cost diff: {cost_diff:.4f} BDT]")

    return failures


def run_suite_paraphrase_and_adversarial(base_url: str) -> int:
    """Suite 2: Paraphrased Phrasing & Adversarial LLM Tests."""
    logger.info("=== RUNNING SUITE 2: Paraphrase & Adversarial Directives ===")

    test_cases = [
        (
            "paraphrase_solar_1",
            ["Solar output will be about 80% of normal today from 1 PM to 3 PM."],
            "solar_reduction",
            {"hours": [13, 14], "factor": 0.8}
        ),
        (
            "paraphrase_solar_2",
            ["Please assume a 20% reduction in PV generation between 13:00 and 15:00."],
            "solar_reduction",
            {"hours": [13, 14], "factor": 0.8}
        ),
        (
            "paraphrase_solar_3",
            ["Cloud cover means solar availability is down by 20% from 1 PM until 3 PM."],
            "solar_reduction",
            {"hours": [13, 14], "factor": 0.8}
        ),
        (
            "paraphrase_solar_4",
            ["Only 80% of expected solar generation is available between 13:00 and 15:00."],
            "solar_reduction",
            {"hours": [13, 14], "factor": 0.8}
        ),
        (
            "paraphrase_reserve_1",
            ["keep 30% battery reserve between 6 PM and 10 PM"],
            "minimum_battery_reserve",
            {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 30.0}
        ),
        (
            "paraphrase_reserve_2",
            ["never let the battery fall below 30% from 18:00 to 22:00"],
            "minimum_battery_reserve",
            {"hours": [18, 19, 20, 21], "minimum_energy_kwh": 30.0}
        ),
        (
            "paraphrase_no_charge_1",
            ["avoid charging between 14:00 and 17:00"],
            "no_charge_window",
            {"hours": [14, 15, 16]}
        ),
        (
            "paraphrase_no_charge_2",
            ["charging is prohibited during 14:00 to 17:00"],
            "no_charge_window",
            {"hours": [14, 15, 16]}
        ),
        (
            "paraphrase_distractor",
            ["The university cafeteria is serving chicken biryani today at noon."],
            "no_op",
            None
        ),
    ]

    failures = 0
    for cid, notes, exp_dtype, exp_adj in test_cases:
        payload = generate_base_scenario(cid, notes)
        resp = requests.post(f"{base_url}/optimize-energy", json=payload, timeout=25.0)

        if resp.status_code != 200:
            logger.error(f"Test {cid} FAILED: HTTP {resp.status_code} - {resp.text}")
            failures += 1
            continue

        resp_json = resp.json()
        passed, msg = IndependentJudge.audit_response(payload, resp_json)
        if not passed:
            logger.error(f"Test {cid} Audit Violation: {msg}")
            failures += 1
            continue

        d = resp_json["directive_interpretation"][0]
        if d["directive_type"] != exp_dtype:
            logger.error(f"Test {cid}: expected directive_type '{exp_dtype}', got '{d['directive_type']}'")
            failures += 1
            continue

        if exp_adj is not None:
            act_adj = d.get("structured_adjustment", {})
            if act_adj.get("hours") != exp_adj["hours"]:
                logger.error(f"Test {cid}: hours mismatch: exp {exp_adj['hours']}, got {act_adj.get('hours')}")
                failures += 1
                continue
            if "factor" in exp_adj and abs(act_adj.get("factor", 0.0) - exp_adj["factor"]) > 0.05:
                logger.error(f"Test {cid}: factor mismatch: exp {exp_adj['factor']}, got {act_adj.get('factor')}")
                failures += 1
                continue
            if "minimum_energy_kwh" in exp_adj and abs(act_adj.get("minimum_energy_kwh", 0.0) - exp_adj["minimum_energy_kwh"]) > 0.05:
                logger.error(f"Test {cid}: reserve mismatch: exp {exp_adj['minimum_energy_kwh']}, got {act_adj.get('minimum_energy_kwh')}")
                failures += 1
                continue

        logger.info(f"✓ Test {cid} PASSED ({exp_dtype})")

    return failures


def run_suite_extreme_edge_cases(base_url: str) -> int:
    """Suite 3: Extreme Physical Scenarios (Zero Solar, Zero Demand, Saturated Rates)."""
    logger.info("=== RUNNING SUITE 3: Physical Edge Cases ===")

    edge_cases = []

    # 1. Zero Demand
    sc_zero_dem = generate_base_scenario("edge_zero_demand", ["sports office meeting from 2 PM to 4 PM"])
    for h in sc_zero_dem["hours"]:
        h["demand_kwh"] = 0.0
    edge_cases.append(("Zero Demand", sc_zero_dem))

    # 2. Zero Solar
    sc_zero_sol = generate_base_scenario("edge_zero_solar", ["library maintenance from 1 PM to 3 PM"])
    for h in sc_zero_sol["hours"]:
        h["solar_kwh"] = 0.0
    edge_cases.append(("Zero Solar", sc_zero_sol))

    # 3. High Demand
    sc_high_dem = generate_base_scenario("edge_high_demand", ["library quiet hours from 2 PM to 4 PM"])
    for h in sc_high_dem["hours"]:
        h["demand_kwh"] = 250.0
    edge_cases.append(("High Demand", sc_high_dem))

    # 4. Battery at Capacity Initially
    sc_bat_cap = generate_base_scenario("edge_battery_full", ["cafeteria notice for 12 PM to 2 PM"])
    sc_bat_cap["battery"]["initial_energy_kwh"] = 100.0
    edge_cases.append(("Battery at Full Capacity", sc_bat_cap))

    # 5. Battery at Reserve Initially
    sc_bat_res = generate_base_scenario("edge_battery_reserve", ["cafeteria notice for 12 PM to 2 PM"])
    sc_bat_res["battery"]["initial_energy_kwh"] = 20.0
    edge_cases.append(("Battery at Minimum Reserve", sc_bat_res))

    failures = 0
    for name, payload in edge_cases:
        resp = requests.post(f"{base_url}/optimize-energy", json=payload, timeout=25.0)
        if resp.status_code != 200:
            logger.error(f"Edge case '{name}' FAILED with HTTP {resp.status_code}: {resp.text}")
            failures += 1
            continue

        resp_json = resp.json()
        passed, msg = IndependentJudge.audit_response(payload, resp_json)
        if not passed:
            logger.error(f"Edge case '{name}' Audit Violation: {msg}")
            failures += 1
            continue

        logger.info(f"✓ Edge case '{name}' PASSED")

    return failures


def run_suite_malformed_input_rejection(base_url: str) -> int:
    """Suite 4: Malformed Request Schema Rejections (Strict HTTP 400)."""
    logger.info("=== RUNNING SUITE 4: Malformed Input Rejection ===")

    malformed_payloads = [
        ("Missing hours entry (23 instead of 24)", {
            "scenario_id": "bad_hours_count",
            "operator_notes": ["routine check"],
            "hours": generate_base_scenario("s", ["n"])["hours"][:23],
            "battery": generate_base_scenario("s", ["n"])["battery"]
        }),
        ("Duplicate hour index", {
            "scenario_id": "dup_hour",
            "operator_notes": ["routine check"],
            "hours": generate_base_scenario("s", ["n"])["hours"][:23] + [generate_base_scenario("s", ["n"])["hours"][0]],
            "battery": generate_base_scenario("s", ["n"])["battery"]
        }),
        ("Empty operator notes list", {
            "scenario_id": "empty_notes",
            "operator_notes": [],
            "hours": generate_base_scenario("s", ["n"])["hours"],
            "battery": generate_base_scenario("s", ["n"])["battery"]
        }),
        ("Four operator notes (exceeds max 3)", {
            "scenario_id": "four_notes",
            "operator_notes": ["note 1", "note 2", "note 3", "note 4"],
            "hours": generate_base_scenario("s", ["n"])["hours"],
            "battery": generate_base_scenario("s", ["n"])["battery"]
        }),
        ("Negative battery capacity", {
            "scenario_id": "neg_cap",
            "operator_notes": ["note 1"],
            "hours": generate_base_scenario("s", ["n"])["hours"],
            "battery": {
                "capacity_kwh": -10.0,
                "initial_energy_kwh": 5.0,
                "minimum_energy_kwh": 0.0,
                "max_charge_kwh_per_hour": 10.0,
                "max_discharge_kwh_per_hour": 10.0
            }
        })
    ]

    failures = 0
    for desc, payload in malformed_payloads:
        resp = requests.post(f"{base_url}/optimize-energy", json=payload, timeout=5.0)
        if resp.status_code == 400:
            logger.info(f"✓ Rejection check passed for: {desc} (HTTP 400)")
        else:
            logger.error(f"Expected HTTP 400 for '{desc}', got HTTP {resp.status_code}")
            failures += 1

    return failures


def run_suite_stability_repeated_requests(base_url: str, iterations: int = 50) -> int:
    """Suite 5: 50 Repeated Requests to verify performance, determinism, and zero state accumulation."""
    logger.info(f"=== RUNNING SUITE 5: {iterations} Repeated Requests Stability & Performance ===")

    payload = generate_base_scenario(
        "stress_scenario",
        ["reduce solar generation by 20% between 13:00 and 15:00"]
    )

    times = []
    first_cost = None
    failures = 0

    for i in range(iterations):
        t0 = time.time()
        resp = requests.post(f"{base_url}/optimize-energy", json=payload, timeout=25.0)
        dt = time.time() - t0
        times.append(dt)

        if resp.status_code != 200:
            logger.error(f"Iteration {i+1} FAILED: HTTP {resp.status_code}")
            failures += 1
            continue

        resp_json = resp.json()
        passed, msg = IndependentJudge.audit_response(payload, resp_json)
        if not passed:
            logger.error(f"Iteration {i+1} Audit Violation: {msg}")
            failures += 1
            continue

        cost = resp_json["total_cost_bdt"]
        if first_cost is None:
            first_cost = cost
        elif abs(cost - first_cost) > 0.001:
            logger.error(f"Non-deterministic result at iteration {i+1}: got {cost}, initial was {first_cost}")
            failures += 1

    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0

    logger.info(f"Repeated requests completed. Failures: {failures}/{iterations}")
    logger.info(f"Latency stats: avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s")

    return failures


# =====================================================================
# Main Test Runner
# =====================================================================

def main():
    base_url = get_base_url()
    logger.info("Checking server health...")
    r = requests.get(f"{base_url}/health")
    assert r.status_code == 200 and r.json().get("status") == "ok", "Server health check failed!"
    logger.info("✓ GET /health is OK")

    total_failures = 0

    # Run Suite 1
    total_failures += run_suite_public_samples(base_url)

    # Run Suite 2
    total_failures += run_suite_paraphrase_and_adversarial(base_url)

    # Run Suite 3
    total_failures += run_suite_extreme_edge_cases(base_url)

    # Run Suite 4
    total_failures += run_suite_malformed_input_rejection(base_url)

    # Run Suite 5
    total_failures += run_suite_stability_repeated_requests(base_url, iterations=50)

    print("\n" + "="*70)
    if total_failures == 0:
        logger.info("ALL TEST SUITES PASSED! Solution is robust, validated, and judge-ready.")
        sys.exit(0)
    else:
        logger.error(f"TEST RUN COMPLETED WITH {total_failures} TOTAL FAILURES.")
        sys.exit(1)


if __name__ == "__main__":
    main()
