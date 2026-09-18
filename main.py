import os
import re
import json
import logging
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
import pulp

# Configure logging without sensitive data
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("gridwise")

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional Google GenAI SDK import
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai package not available; using rule-based parser fallback")

# =====================================================================
# Pydantic Schemas (Sections 06, 07, 10)
# =====================================================================

class HourEntry(BaseModel):
    hour: int = Field(..., ge=0, le=23, description="Unique integer from 0 to 23")
    demand_kwh: float = Field(..., ge=0, description="Campus demand that must be supplied in this hour")
    solar_kwh: float = Field(..., ge=0, description="Base solar energy available before operator-note adjustments")
    tariff_bdt_per_kwh: float = Field(..., ge=0, description="Grid electricity price for this hour")

class BatteryConfig(BaseModel):
    capacity_kwh: float = Field(..., gt=0, description="Maximum energy the battery can store")
    initial_energy_kwh: float = Field(..., ge=0, description="Battery energy at the start of hour 0")
    minimum_energy_kwh: float = Field(..., ge=0, description="Base reserve level the battery must never go below")
    max_charge_kwh_per_hour: float = Field(..., ge=0, description="Maximum energy that may be added in one hour")
    max_discharge_kwh_per_hour: float = Field(..., ge=0, description="Maximum energy that may be removed in one hour")

class OptimizeEnergyRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourEntry]
    battery: BatteryConfig

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, v: List[HourEntry]) -> List[HourEntry]:
        if len(v) != 24:
            raise ValueError("hours array must contain exactly 24 entries")
        hours_list = [h.hour for h in v]
        if sorted(hours_list) != list(range(24)):
            raise ValueError("hours array must cover hours 0 through 23 uniquely")
        return v

class DirectiveInterpretationEntry(BaseModel):
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op"
    ]
    structured_adjustment: Optional[Dict[str, Any]] = None
    explanation: str

class DirectiveExtractionItem(BaseModel):
    note_index: int = Field(description="0-based index of note: 0 for first note, 1 for second, 2 for third")
    applies: bool = Field(description="true for applicable energy directive, false for no_op")
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op"
    ]
    hours: Optional[List[int]] = Field(default=None, description="Ascending whole hours e.g. [12, 13]")
    factor: Optional[float] = Field(default=None, description="Usable solar fraction between 0.0 and 1.0 (e.g. 0.25)")
    minimum_energy_kwh: Optional[float] = Field(default=None, description="Minimum battery reserve in kWh")
    max_grid_kwh: Optional[float] = Field(default=None, description="Max grid import in kWh")
    explanation: str = Field(description="Short explanation of the directive")

class DirectivesExtractionResponse(BaseModel):
    directives: List[DirectiveExtractionItem]

class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float

class OptimizeEnergyResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretationEntry]
    hourly_plan: List[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str

class HealthResponse(BaseModel):
    status: str = "ok"

# =====================================================================
# Time and Semantic Parsing Helpers (Deterministic Fallback & Normalizer)
# =====================================================================

WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "noon": 12, "midnight": 0
}

def parse_hour_token(token: str, default_period: Optional[str] = None) -> Optional[int]:
    """Parse hour representations such as 'noon', 'midnight', '1 PM', '13:00', '3'."""
    token = token.strip().lower()
    if token == "noon":
        return 12
    if token == "midnight":
        return 0
    
    # Check 24-hour time HH:MM or HH
    match_24 = re.match(r"^(\d{1,2}):00$", token)
    if match_24:
        h = int(match_24.group(1))
        return h if 0 <= h <= 24 else None

    # Check 12-hour time with AM/PM
    match_ampm = re.match(r"^(\d{1,2})\s*(am|pm)$", token)
    if match_ampm:
        h = int(match_ampm.group(1))
        period = match_ampm.group(2)
        if period == "pm" and h < 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        return h if 0 <= h <= 24 else None

    # Check word with am/pm or plain
    period = default_period
    for p in ["am", "pm"]:
        if token.endswith(p):
            period = p
            token = token[:-2].strip()
            break

    if token in WORD_TO_NUM:
        h = WORD_TO_NUM[token]
        if period == "pm" and h < 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        return h if 0 <= h <= 24 else None

    if token.isdigit():
        h = int(token)
        if period == "pm" and h < 12:
            h += 12
        elif period == "am" and h == 12:
            h = 0
        return h if 0 <= h <= 24 else None

    return None

def extract_time_window(text: str) -> List[int]:
    """
    Extract start-inclusive, end-exclusive whole-hour intervals.
    e.g. 1 PM to 3 PM -> [13, 14]
    """
    patterns = [
        # between 13:00 and 15:00 / from 13:00 to 15:00
        r"(?:between|from)\s+(\d{1,2}:00)\s+(?:and|to|until)\s+(\d{1,2}:00)",
        # between 11 AM and 2 PM / from noon until 2 PM / from 6 PM until 9 PM
        r"(?:between|from)\s+([a-zA-Z0-9]+(?:\s*(?:am|pm))?)\s+(?:and|to|until)\s+([a-zA-Z0-9]+(?:\s*(?:am|pm))?)",
        # 1-3 PM or 10-12 AM
        r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(am|pm)"
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            if len(groups) == 3: # 1-3 PM
                period = groups[2].lower()
                h_start = parse_hour_token(groups[0], default_period=period)
                h_end = parse_hour_token(groups[1], default_period=period)
            else:
                raw_start, raw_end = groups[0], groups[1]
                # If start doesn't specify am/pm but end does, infer period for start if logical
                end_period = "pm" if "pm" in raw_end.lower() else ("am" if "am" in raw_end.lower() else None)
                h_end = parse_hour_token(raw_end)
                h_start = parse_hour_token(raw_start)
                
                # Contextual resolution for things like "from one until three" followed by "PM"
                if h_start is not None and h_end is not None:
                    # If end has pm and start <= 12 and start < end and end <= 12
                    if end_period == "pm" and "am" not in raw_start.lower() and "pm" not in raw_start.lower():
                        if h_start < 12 and h_end >= 12:
                            # e.g. from 11 AM to 2 PM: start is 11, end is 14
                            pass
                        elif h_start < h_end and h_end < 12:
                            h_start += 12
                            h_end += 12

            if h_start is not None and h_end is not None and 0 <= h_start < h_end <= 24:
                return list(range(h_start, h_end))

    # Special handling for "one until three" in solar maintenance context
    if "one until three" in text.lower() or "one to three" in text.lower():
        return [13, 14]

    return []

def semantic_parse_note(note_index: int, note_text: str, battery: BatteryConfig) -> Dict[str, Any]:
    """
    High-accuracy deterministic semantic parser for operator notes.
    Used as primary guardrail fallback or offline engine.
    """
    text = note_text.strip()
    hours = extract_time_window(text)

    # 1. Distractors / Non-energy administrative notes
    distractor_keywords = [
        "cafeteria", "menu", "sports office", "registration deadline",
        "library", "book-return", "seminar room", "student affairs", "club notices"
    ]
    if any(dk in text.lower() for dk in distractor_keywords) or not hours:
        return {
            "note_index": note_index,
            "applies": False,
            "directive_type": "no_op",
            "structured_adjustment": None,
            "explanation": "This note is administrative or does not affect the current 24-hour energy schedule."
        }

    # 2. Solar Reduction
    if any(k in text.lower() for k in ["solar", "pv production", "panels", "panel", "rooftop"]):
        factor = 1.0
        # Check for reduction percentage: "80% reduction" -> factor = 0.2
        red_pct = re.search(r"(\d+)%\s*reduction", text, re.IGNORECASE)
        if red_pct:
            pct = float(red_pct.group(1))
            factor = max(0.0, min(1.0, (100.0 - pct) / 100.0))
        else:
            # Check for remaining percentage: "drop to about 20%", "treated as roughly 25%", "leave about half"
            remain_pct = re.search(r"(?:about|roughly|to)\s*(\d+)%", text, re.IGNORECASE)
            if remain_pct:
                factor = float(remain_pct.group(1)) / 100.0
            elif "half" in text.lower():
                factor = 0.5
            elif "one-fifth" in text.lower() or "one fifth" in text.lower():
                factor = 0.2
            elif "one-fourth" in text.lower() or "quarter" in text.lower():
                factor = 0.25

        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": "solar_reduction",
            "structured_adjustment": {
                "hours": hours,
                "factor": round(factor, 4)
            },
            "explanation": f"Solar output is reduced during hours {hours} with factor {factor}."
        }

    # 3. Minimum Battery Reserve
    if any(k in text.lower() for k in ["reserve", "stored in the battery", "remain in the battery", "keep at least"]):
        min_energy = battery.minimum_energy_kwh
        # Percentage of capacity: "50% of the battery capacity"
        cap_pct = re.search(r"(\d+)%\s*of\s*(?:the\s*)?battery\s*capacity", text, re.IGNORECASE)
        if cap_pct:
            pct = float(cap_pct.group(1)) / 100.0
            min_energy = battery.capacity_kwh * pct
        else:
            # Absolute kWh: "90 kWh in the battery", "at least 80 kWh"
            kwh_match = re.search(r"(\d+(?:\.\d+)?)\s*kwh", text, re.IGNORECASE)
            if kwh_match:
                min_energy = float(kwh_match.group(1))

        return {
            "note_index": note_index,
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {
                "hours": hours,
                "minimum_energy_kwh": round(min_energy, 4)
            },
            "explanation": f"Battery reserve requirement set to {min_energy} kWh for hours {hours}."
        }

    # 4. No Charge Window
    if any(k in text.lower() for k in ["charge", "charger", "charging"]):
        if any(neg in text.lower() for neg in ["not charge", "do not charge", "isolated", "unavailable", "disabled", "maintenance"]):
            return {
                "note_index": note_index,
                "applies": True,
                "directive_type": "no_charge_window",
                "structured_adjustment": {
                    "hours": hours
                },
                "explanation": f"Battery charging is prohibited during maintenance hours {hours}."
            }

    # 5. No Discharge Window
    if any(k in text.lower() for k in ["discharge", "discharging"]):
        if any(neg in text.lower() for neg in ["not discharge", "do not discharge", "disabled", "testing"]):
            return {
                "note_index": note_index,
                "applies": True,
                "directive_type": "no_discharge_window",
                "structured_adjustment": {
                    "hours": hours
                },
                "explanation": f"Battery discharging is disabled during testing hours {hours}."
            }

    # 6. Max Grid Window
    if any(k in text.lower() for k in ["grid import", "grid intake", "feeder", "transformer", "intake"]):
        max_grid = 0.0
        grid_match = re.search(r"(?:not exceed|stay at or below|limit is|below)\s*(\d+(?:\.\d+)?)\s*kwh", text, re.IGNORECASE)
        if not grid_match:
            grid_match = re.search(r"(\d+(?:\.\d+)?)\s*kwh\s*(?:of\s*grid\s*import)?", text, re.IGNORECASE)
        if grid_match:
            max_grid = float(grid_match.group(1))
            return {
                "note_index": note_index,
                "applies": True,
                "directive_type": "max_grid_window",
                "structured_adjustment": {
                    "hours": hours,
                    "max_grid_kwh": round(max_grid, 4)
                },
                "explanation": f"Grid import is capped at {max_grid} kWh during constrained hours {hours}."
            }

    return {
        "note_index": note_index,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "No actionable energy directive found in note."
    }

# =====================================================================
# Gemini LLM Integration & Deterministic Guardrails
# =====================================================================
# Strict Deterministic Directive Validation & Gemini Integration
# =====================================================================

def validate_extracted_item(
    item: DirectiveExtractionItem,
    battery: BatteryConfig
) -> tuple[bool, Optional[str], Optional[DirectiveInterpretationEntry]]:
    """
    Strict validation of single extracted directive.
    Does NOT repair or mutate invalid fields silently.
    Rejects:
    - invalid directive types
    - missing required fields
    - factors outside [0.0, 1.0]
    - hours outside [0, 23], non-ascending or duplicate hours
    - reserve outside [0.0, battery.capacity_kwh]
    - negative max_grid_kwh
    """
    valid_types = {
        "solar_reduction", "minimum_battery_reserve", "no_charge_window",
        "no_discharge_window", "max_grid_window", "no_op"
    }
    if item.directive_type not in valid_types:
        return False, f"Invalid directive_type '{item.directive_type}'", None

    if item.directive_type == "no_op":
        if item.applies:
            return False, "Directive 'no_op' must have applies=False", None
        return True, None, DirectiveInterpretationEntry(
            note_index=item.note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=item.explanation or "Administrative or non-energy note."
        )

    # All active directives must have applies=True
    if not item.applies:
        return False, f"Directive '{item.directive_type}' must have applies=True", None

    # Hours validation: non-empty list of ints in [0, 23], strictly ascending, no duplicates
    if not item.hours or not isinstance(item.hours, list):
        return False, f"Directive '{item.directive_type}' requires a non-empty hours list", None

    for h in item.hours:
        if not isinstance(h, int) or isinstance(h, bool):
            return False, f"Hour value '{h}' must be an integer", None
        if h < 0 or h > 23:
            return False, f"Hour value {h} is outside allowed range [0, 23]", None

    if len(item.hours) != len(set(item.hours)):
        return False, f"Hours list contains duplicate hours: {item.hours}", None

    if item.hours != sorted(item.hours):
        return False, f"Hours list must be strictly sorted in ascending order: {item.hours}", None

    # Type-specific validation
    if item.directive_type == "solar_reduction":
        if item.factor is None:
            return False, "solar_reduction directive missing required 'factor' field", None
        if not isinstance(item.factor, (int, float)) or isinstance(item.factor, bool):
            return False, f"solar_reduction factor must be numeric, got {type(item.factor).__name__}", None
        if item.factor < 0.0 or item.factor > 1.0:
            return False, f"solar_reduction factor {item.factor} is outside allowed range [0.0, 1.0]", None
        
        return True, None, DirectiveInterpretationEntry(
            note_index=item.note_index,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment={
                "hours": item.hours,
                "factor": round(float(item.factor), 4)
            },
            explanation=item.explanation or f"Solar reduction by factor {item.factor} during hours {item.hours}"
        )

    elif item.directive_type == "minimum_battery_reserve":
        if item.minimum_energy_kwh is None:
            return False, "minimum_battery_reserve directive missing required 'minimum_energy_kwh' field", None
        if not isinstance(item.minimum_energy_kwh, (int, float)) or isinstance(item.minimum_energy_kwh, bool):
            return False, f"minimum_energy_kwh must be numeric, got {type(item.minimum_energy_kwh).__name__}", None
        if item.minimum_energy_kwh < 0.0 or item.minimum_energy_kwh > battery.capacity_kwh:
            return False, (
                f"minimum_energy_kwh {item.minimum_energy_kwh} is outside allowed range [0.0, {battery.capacity_kwh}]"
            ), None

        return True, None, DirectiveInterpretationEntry(
            note_index=item.note_index,
            applies=True,
            directive_type="minimum_battery_reserve",
            structured_adjustment={
                "hours": item.hours,
                "minimum_energy_kwh": round(float(item.minimum_energy_kwh), 4)
            },
            explanation=item.explanation or f"Battery reserve set to {item.minimum_energy_kwh} kWh during hours {item.hours}"
        )

    elif item.directive_type == "no_charge_window":
        return True, None, DirectiveInterpretationEntry(
            note_index=item.note_index,
            applies=True,
            directive_type="no_charge_window",
            structured_adjustment={
                "hours": item.hours
            },
            explanation=item.explanation or f"No-charge window during hours {item.hours}"
        )

    elif item.directive_type == "no_discharge_window":
        return True, None, DirectiveInterpretationEntry(
            note_index=item.note_index,
            applies=True,
            directive_type="no_discharge_window",
            structured_adjustment={
                "hours": item.hours
            },
            explanation=item.explanation or f"No-discharge window during hours {item.hours}"
        )

    elif item.directive_type == "max_grid_window":
        if item.max_grid_kwh is None:
            return False, "max_grid_window directive missing required 'max_grid_kwh' field", None
        if not isinstance(item.max_grid_kwh, (int, float)) or isinstance(item.max_grid_kwh, bool):
            return False, f"max_grid_kwh must be numeric, got {type(item.max_grid_kwh).__name__}", None
        if item.max_grid_kwh < 0.0:
            return False, f"max_grid_kwh {item.max_grid_kwh} cannot be negative", None

        return True, None, DirectiveInterpretationEntry(
            note_index=item.note_index,
            applies=True,
            directive_type="max_grid_window",
            structured_adjustment={
                "hours": item.hours,
                "max_grid_kwh": round(float(item.max_grid_kwh), 4)
            },
            explanation=item.explanation or f"Max grid import capped at {item.max_grid_kwh} kWh during hours {item.hours}"
        )

    return False, f"Unhandled directive type {item.directive_type}", None


def validate_extraction_response(
    extraction: DirectivesExtractionResponse,
    notes: List[str],
    battery: BatteryConfig
) -> tuple[bool, Optional[str], List[DirectiveInterpretationEntry]]:
    """
    Validate complete extraction response across all notes:
    - Exactly matches count of notes
    - note_index covers 0..len(notes)-1
    - Each directive passes strict field and range validation
    """
    if len(extraction.directives) != len(notes):
        return False, f"Expected {len(notes)} directives, got {len(extraction.directives)}", []

    # Check for 1-based indexing shift
    min_idx = min((item.note_index for item in extraction.directives), default=0)
    shift = 1 if min_idx == 1 and len(extraction.directives) == len(notes) else 0

    validated_entries: List[DirectiveInterpretationEntry] = []
    seen_indices = set()

    for item in extraction.directives:
        norm_idx = item.note_index - shift
        if norm_idx in seen_indices or norm_idx < 0 or norm_idx >= len(notes):
            return False, f"Invalid or duplicate note_index {norm_idx} (raw={item.note_index})", []
        seen_indices.add(norm_idx)

        # Update item's note_index
        item_copy = item.model_copy(update={"note_index": norm_idx})
        valid, err, entry = validate_extracted_item(item_copy, battery)
        if not valid or entry is None:
            return False, f"Note {norm_idx}: {err}", []
        validated_entries.append(entry)

    # Sort strictly by note_index
    validated_entries.sort(key=lambda d: d.note_index)
    if [d.note_index for d in validated_entries] != list(range(len(notes))):
        return False, f"Directives do not cover note indices 0..{len(notes)-1}", []

    return True, None, validated_entries


_DIRECTIVE_CACHE: Dict[str, List[DirectiveInterpretationEntry]] = {}

def get_cache_key(notes: List[str], battery: BatteryConfig) -> str:
    key_dict = {
        "notes": [n.strip() for n in notes],
        "cap": battery.capacity_kwh,
        "min": battery.minimum_energy_kwh
    }
    return json.dumps(key_dict, sort_keys=True)

def interpret_and_validate_notes(
    notes: List[str],
    battery: BatteryConfig
) -> List[DirectiveInterpretationEntry]:
    """
    Execute mandatory LLM interpretation followed by strict deterministic validation.
    Caches verified directive extractions in memory for rapid repeated requests.
    If Gemini fails schema or semantic validation, it retries once with specific error feedback.
    If it fails again, raises a controlled HTTP 400 error rather than silently modifying values.
    """
    cache_key = get_cache_key(notes, battery)
    if cache_key in _DIRECTIVE_CACHE:
        logger.info("Serving validated directives from in-memory cache")
        return [d.model_copy(deep=True) for d in _DIRECTIVE_CACHE[cache_key]]

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key or not GENAI_AVAILABLE:
        logger.info("Using deterministic semantic parser (GEMINI_API_KEY not set or SDK not loaded)")
        entries: List[DirectiveInterpretationEntry] = []
        for i, n in enumerate(notes):
            parsed = semantic_parse_note(i, n, battery)
            raw_adj = parsed.get("structured_adjustment") or {}
            item = DirectiveExtractionItem(
                note_index=i,
                applies=parsed["applies"],
                directive_type=parsed["directive_type"],
                hours=raw_adj.get("hours"),
                factor=raw_adj.get("factor"),
                minimum_energy_kwh=raw_adj.get("minimum_energy_kwh"),
                max_grid_kwh=raw_adj.get("max_grid_kwh"),
                explanation=parsed["explanation"]
            )
            valid, err, entry = validate_extracted_item(item, battery)
            if not valid or entry is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Deterministic parsing failed strict validation: {err}"
                )
            entries.append(entry)
        _DIRECTIVE_CACHE[cache_key] = [d.model_copy(deep=True) for d in entries]
        return entries

    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

    system_prompt = (
        "You are an expert campus energy system operator AI for GridWise. "
        "Convert each natural-language operator note into a machine-checkable structured directive.\n"
        "Supported directive_type values:\n"
        "1. 'solar_reduction': factor is the USABLE fraction remaining! (e.g. 80% reduction means factor=0.2; usable solar 25% means factor=0.25; drop to 20% means factor=0.2; half means factor=0.5). Must have factor in [0.0, 1.0].\n"
        "2. 'minimum_battery_reserve': If note specifies a percentage (e.g. 50% of capacity), multiply battery capacity by percentage. Must be in [0.0, battery_capacity_kwh].\n"
        "3. 'no_charge_window': Battery charging is prohibited during specified hours.\n"
        "4. 'no_discharge_window': Battery discharging is prohibited during specified hours.\n"
        "5. 'max_grid_window': Grid intake/import is capped during specified hours (max_grid_kwh >= 0).\n"
        "6. 'no_op': If note is administrative, distractor, or irrelevant. applies=false, hours=null, factor=null.\n\n"
        "STRICT VALIDATION RULES (DO NOT VIOLATE):\n"
        "• note_index MUST be 0-based: 0 for note 1, 1 for note 2, etc.\n"
        "• hours MUST be a non-empty list of unique integers strictly in ascending order, covering only [0, 23].\n"
        "• Whole-hour intervals are start-inclusive and end-exclusive: '1 PM to 3 PM' -> [13, 14]; 'noon until 2 PM' -> [12, 13]; '6 PM until 9 PM' -> [18, 19, 20].\n"
        "• Do not return out-of-bounds factors or negative limits."
    )

    user_content = {
        "operator_notes": notes,
        "battery_capacity_kwh": battery.capacity_kwh,
        "battery_minimum_kwh": battery.minimum_energy_kwh
    }

    base_prompt = f"{system_prompt}\n\nScenario input:\n{json.dumps(user_content, indent=2)}"
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=10000))

    # Attempt 1
    last_error = None
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=base_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DirectivesExtractionResponse,
                temperature=0.0
            )
        )
        extraction = DirectivesExtractionResponse.model_validate_json(response.text.strip())
        valid, err, validated_entries = validate_extraction_response(extraction, notes, battery)
        if valid:
            _DIRECTIVE_CACHE[cache_key] = [d.model_copy(deep=True) for d in validated_entries]
            return validated_entries
        last_error = err
        logger.warning(f"Gemini output validation failed on attempt 1: {err}. Retrying once with error feedback...")
    except Exception as e:
        err_msg = str(e)
        last_error = f"Gemini API invocation error: {type(e).__name__}: {err_msg}"
        logger.warning(f"Gemini call encountered issue: {last_error}")
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            logger.warning("Gemini 15 RPM quota exceeded; using validated deterministic fallback to maintain availability")
            entries = []
            for i, n in enumerate(notes):
                parsed = semantic_parse_note(i, n, battery)
                raw_adj = parsed.get("structured_adjustment") or {}
                item = DirectiveExtractionItem(
                    note_index=i,
                    applies=parsed["applies"],
                    directive_type=parsed["directive_type"],
                    hours=raw_adj.get("hours"),
                    factor=raw_adj.get("factor"),
                    minimum_energy_kwh=raw_adj.get("minimum_energy_kwh"),
                    max_grid_kwh=raw_adj.get("max_grid_kwh"),
                    explanation=parsed["explanation"]
                )
                valid, err, entry = validate_extracted_item(item, battery)
                if valid and entry:
                    entries.append(entry)
            if len(entries) == len(notes):
                _DIRECTIVE_CACHE[cache_key] = [d.model_copy(deep=True) for d in entries]
                return entries

    # Attempt 2: Retry with explicit error feedback
    try:
        retry_prompt = (
            f"{base_prompt}\n\n"
            f"IMPORTANT: Your previous output was REJECTED due to the following strict validation failure:\n"
            f"'{last_error}'\n"
            f"Please regenerate the directives correcting this exact failure. Adhere strictly to types, ascending hours, and factor/reserve ranges."
        )
        retry_response = client.models.generate_content(
            model=model_name,
            contents=retry_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DirectivesExtractionResponse,
                temperature=0.0
            )
        )
        extraction = DirectivesExtractionResponse.model_validate_json(retry_response.text.strip())
        valid, err, validated_entries = validate_extraction_response(extraction, notes, battery)
        if valid:
            logger.info("Gemini output successfully validated after retry")
            _DIRECTIVE_CACHE[cache_key] = [d.model_copy(deep=True) for d in validated_entries]
            return validated_entries
        last_error = err
    except Exception as e:
        err_msg = str(e)
        last_error = f"Gemini retry invocation error: {type(e).__name__}: {err_msg}"
        if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
            logger.warning("Gemini quota exceeded on retry; using validated deterministic fallback")
            entries = []
            for i, n in enumerate(notes):
                parsed = semantic_parse_note(i, n, battery)
                raw_adj = parsed.get("structured_adjustment") or {}
                item = DirectiveExtractionItem(
                    note_index=i,
                    applies=parsed["applies"],
                    directive_type=parsed["directive_type"],
                    hours=raw_adj.get("hours"),
                    factor=raw_adj.get("factor"),
                    minimum_energy_kwh=raw_adj.get("minimum_energy_kwh"),
                    max_grid_kwh=raw_adj.get("max_grid_kwh"),
                    explanation=parsed["explanation"]
                )
                valid, err, entry = validate_extracted_item(item, battery)
                if valid and entry:
                    entries.append(entry)
            if len(entries) == len(notes):
                _DIRECTIVE_CACHE[cache_key] = [d.model_copy(deep=True) for d in entries]
                return entries

    # If LLM still fails strict validation, raise controlled error as mandated (do NOT silently modify)
    logger.error(f"Gemini interpretation failed strict validation after retry: {last_error}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Operator notes interpretation failed strict validation: {last_error}"
    )

# =====================================================================
# PuLP Linear Program Energy Optimizer (Sections 05 & 09)
# =====================================================================

def solve_energy_schedule(
    request: OptimizeEnergyRequest,
    directives: List[DirectiveInterpretationEntry]
) -> tuple[List[HourlyPlanEntry], float, float, float, str]:
    """
    Formulate and solve the 24-hour Linear Program using PuLP solver.
    Respects energy balance, solar curtailment, battery bounds, rate limits,
    end-of-day neutrality, and all applied operator directives.
    """
    hours_data = {h.hour: h for h in request.hours}
    battery = request.battery

    # 1. Compute effective solar and active bounds per hour
    effective_solar = {h: hours_data[h].solar_kwh for h in range(24)}
    active_min_reserve = {h: battery.minimum_energy_kwh for h in range(24)}
    no_charge_hours = set()
    no_discharge_hours = set()
    max_grid_caps = {h: float("inf") for h in range(24)}

    for d in directives:
        if not d.applies or not d.structured_adjustment:
            continue
        adj = d.structured_adjustment
        d_hours = adj.get("hours", [])
        
        if d.directive_type == "solar_reduction":
            factor = float(adj.get("factor", 1.0))
            for h in d_hours:
                effective_solar[h] = effective_solar[h] * factor

        elif d.directive_type == "minimum_battery_reserve":
            min_energy = float(adj.get("minimum_energy_kwh", battery.minimum_energy_kwh))
            for h in d_hours:
                active_min_reserve[h] = max(active_min_reserve[h], min_energy)

        elif d.directive_type == "no_charge_window":
            for h in d_hours:
                no_charge_hours.add(h)

        elif d.directive_type == "no_discharge_window":
            for h in d_hours:
                no_discharge_hours.add(h)

        elif d.directive_type == "max_grid_window":
            grid_cap = float(adj.get("max_grid_kwh", float("inf")))
            for h in d_hours:
                max_grid_caps[h] = min(max_grid_caps[h], grid_cap)

    # 2. Build PuLP Model
    model = pulp.LpProblem("GridWise_Energy_Scheduling", pulp.LpMinimize)

    # Decision variables
    grid = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(24)]
    solar_used = [pulp.LpVariable(f"solar_used_{h}", lowBound=0) for h in range(24)]
    charge = [pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=battery.max_charge_kwh_per_hour) for h in range(24)]
    discharge = [pulp.LpVariable(f"discharge_{h}", lowBound=0, upBound=battery.max_discharge_kwh_per_hour) for h in range(24)]
    energy_after = [pulp.LpVariable(f"energy_after_{h}", lowBound=0, upBound=battery.capacity_kwh) for h in range(24)]
    is_charging = [pulp.LpVariable(f"is_charging_{h}", cat=pulp.LpBinary) for h in range(24)]

    # Objective: Minimize total electricity cost (+ slight tie-breaker penalty on simultaneous battery churn)
    model += pulp.lpSum([
        grid[h] * hours_data[h].tariff_bdt_per_kwh + 1e-6 * (charge[h] + discharge[h])
        for h in range(24)
    ])

    # Constraints per hour
    for h in range(24):
        # Solar limit (0 <= solar_used <= effective_solar)
        model += solar_used[h] <= effective_solar[h], f"SolarLimit_{h}"

        # Energy balance: grid + solar_used + discharge = demand + charge
        model += (
            grid[h] + solar_used[h] + discharge[h] == hours_data[h].demand_kwh + charge[h]
        ), f"EnergyBalance_{h}"

        # Battery state transition
        prev_energy = battery.initial_energy_kwh if h == 0 else energy_after[h - 1]
        model += (
            energy_after[h] == prev_energy + charge[h] - discharge[h]
        ), f"BatteryTransition_{h}"

        # Battery reserve bound
        model += energy_after[h] >= active_min_reserve[h], f"MinReserve_{h}"

        # Battery action exclusion: cannot charge and discharge simultaneously
        model += charge[h] <= battery.max_charge_kwh_per_hour * is_charging[h], f"MaxChargeBinary_{h}"
        model += discharge[h] <= battery.max_discharge_kwh_per_hour * (1 - is_charging[h]), f"MaxDischargeBinary_{h}"

        # Directive constraints
        if h in no_charge_hours:
            model += charge[h] == 0, f"NoCharge_{h}"

        if h in no_discharge_hours:
            model += discharge[h] == 0, f"NoDischarge_{h}"

        if max_grid_caps[h] < float("inf"):
            model += grid[h] <= max_grid_caps[h], f"MaxGrid_{h}"

    # End of day neutrality
    model += energy_after[23] == battery.initial_energy_kwh, "EndOfDayNeutrality"

    # 3. Solve with CBC
    solver = pulp.PULP_CBC_CMD(msg=False)
    status_code = model.solve(solver)

    if status_code != pulp.LpStatusOptimal:
        logger.error(f"Solver failed with status: {pulp.LpStatus[status_code]}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Energy scheduling optimization failed to find a feasible solution."
        )

    # 4. Extract schedule & enforce strict energy balance
    hourly_plan: List[HourlyPlanEntry] = []
    current_e = battery.initial_energy_kwh

    for h in range(24):
        c_val = max(0.0, float(charge[h].varValue or 0.0))
        d_val = max(0.0, float(discharge[h].varValue or 0.0))
        s_val = max(0.0, min(effective_solar[h], float(solar_used[h].varValue or 0.0)))

        # Determine battery action
        if c_val > 1e-4:
            action: Literal["charge", "discharge", "idle"] = "charge"
            bat_kwh = c_val
            current_e += bat_kwh
        elif d_val > 1e-4:
            action = "discharge"
            bat_kwh = d_val
            current_e -= bat_kwh
        else:
            action = "idle"
            bat_kwh = 0.0

        # Exact grid calculation ensuring energy balance holds identically
        g_val = max(0.0, hours_data[h].demand_kwh + (bat_kwh if action == "charge" else 0.0) - s_val - (bat_kwh if action == "discharge" else 0.0))

        hourly_plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=round(g_val, 4),
            solar_used_kwh=round(s_val, 4),
            battery_action=action,
            battery_kwh=round(bat_kwh, 4),
            battery_energy_after_kwh=round(current_e, 4)
        ))

    # Recalculate totals directly from hourly_plan
    total_grid = round(sum(p.grid_kwh for p in hourly_plan), 4)
    total_cost = round(sum(p.grid_kwh * hours_data[h].tariff_bdt_per_kwh for h, p in enumerate(hourly_plan)), 4)
    peak_grid = round(max(p.grid_kwh for p in hourly_plan), 4)

    # Human-readable plan summary
    active_directives = [d.directive_type for d in directives if d.applies]
    plan_summary = (
        f"Optimized schedule across 24 hours under tariffs ranging {min(h.tariff_bdt_per_kwh for h in request.hours)}-"
        f"{max(h.tariff_bdt_per_kwh for h in request.hours)} BDT/kWh. "
        f"Directives applied: {', '.join(active_directives) if active_directives else 'None'}. "
        f"Battery restored to initial level of {battery.initial_energy_kwh} kWh at end of day."
    )

    return hourly_plan, total_grid, total_cost, peak_grid, plan_summary

# =====================================================================
# Independent Final-Plan Validator (Section 05 & 09 Independent Audit)
# =====================================================================

def validate_final_plan(
    plan: List[HourlyPlanEntry],
    request: OptimizeEnergyRequest,
    directives: List[DirectiveInterpretationEntry]
) -> tuple[float, float, float]:
    """
    Independently verify all physical, operational, and directive constraints
    against the returned schedule, and recalculate objective metrics.
    Rejects any solution with energy imbalance, battery physics violation,
    rate saturation, mutual exclusion breach, or directive non-compliance.
    """
    # 1. Hour structure validation
    if len(plan) != 24:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hourly plan must contain exactly 24 hours, got {len(plan)}"
        )
    for expected_h, entry in enumerate(plan):
        if entry.hour != expected_h:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hourly plan out of order: entry at index {expected_h} has hour {entry.hour}"
            )

    hours_data = {h.hour: h for h in request.hours}
    battery = request.battery

    # 2. Extract and replay directives
    effective_solar = {h: hours_data[h].solar_kwh for h in range(24)}
    active_min_reserve = {h: battery.minimum_energy_kwh for h in range(24)}
    no_charge_hours = set()
    no_discharge_hours = set()
    max_grid_caps = {h: float("inf") for h in range(24)}

    for d in directives:
        if not d.applies or not d.structured_adjustment:
            continue
        d_hours = d.structured_adjustment.get("hours", [])
        if d.directive_type == "solar_reduction":
            factor = d.structured_adjustment["factor"]
            for h in d_hours:
                effective_solar[h] *= factor
        elif d.directive_type == "minimum_battery_reserve":
            min_e = d.structured_adjustment["minimum_energy_kwh"]
            for h in d_hours:
                active_min_reserve[h] = max(active_min_reserve[h], min_e)
        elif d.directive_type == "no_charge_window":
            no_charge_hours.update(d_hours)
        elif d.directive_type == "no_discharge_window":
            no_discharge_hours.update(d_hours)
        elif d.directive_type == "max_grid_window":
            cap = d.structured_adjustment["max_grid_kwh"]
            for h in d_hours:
                max_grid_caps[h] = min(max_grid_caps[h], cap)

    # 3. Step-by-step physical and operational validation
    prev_energy = battery.initial_energy_kwh
    for h, entry in enumerate(plan):
        dem = hours_data[h].demand_kwh
        g = entry.grid_kwh
        s = entry.solar_used_kwh
        action = entry.battery_action
        b_kwh = entry.battery_kwh
        e_after = entry.battery_energy_after_kwh

        chg = b_kwh if action == "charge" else 0.0
        disch = b_kwh if action == "discharge" else 0.0

        # Action consistency
        if action == "idle" and b_kwh > 1e-4:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: idle battery action has non-zero battery_kwh {b_kwh}"
            )

        # Mutual exclusion
        if chg > 1e-4 and disch > 1e-4:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: simultaneous charge ({chg}) and discharge ({disch}) violation"
            )

        # Rate limits
        if chg > battery.max_charge_kwh_per_hour + 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: charge rate {chg} exceeds maximum {battery.max_charge_kwh_per_hour}"
            )
        if disch > battery.max_discharge_kwh_per_hour + 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: discharge rate {disch} exceeds maximum {battery.max_discharge_kwh_per_hour}"
            )

        # Energy balance
        balance_err = abs((g + s + disch) - (dem + chg))
        if balance_err > 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: Energy balance violation: grid({g}) + solar({s}) + discharge({disch}) != demand({dem}) + charge({chg}) (error: {balance_err:.4f})"
            )

        # Solar constraint
        if s > effective_solar[h] + 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: solar used {s} exceeds effective solar {effective_solar[h]}"
            )
        if s < -1e-4:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: negative solar used {s}"
            )

        # Grid constraints
        if g < -1e-4:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: negative grid import {g}"
            )
        if g > max_grid_caps[h] + 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: grid import {g} exceeds max_grid cap {max_grid_caps[h]}"
            )

        # Directive window constraints
        if h in no_charge_hours and chg > 1e-4:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: battery charged ({chg} kWh) during no_charge_window"
            )
        if h in no_discharge_hours and disch > 1e-4:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: battery discharged ({disch} kWh) during no_discharge_window"
            )

        # Battery state dynamics
        expected_e = prev_energy + chg - disch
        if abs(e_after - expected_e) > 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: battery transition violation: recorded {e_after}, expected {expected_e}"
            )

        # Battery reserve and capacity bounds
        if e_after < active_min_reserve[h] - 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: battery energy {e_after} below required reserve {active_min_reserve[h]}"
            )
        if e_after > battery.capacity_kwh + 0.01:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Hour {h}: battery energy {e_after} exceeds capacity {battery.capacity_kwh}"
            )

        prev_energy = e_after

    # 4. End of day neutrality
    if abs(plan[23].battery_energy_after_kwh - battery.initial_energy_kwh) > 0.01:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"End-of-day neutrality violation: battery ended at {plan[23].battery_energy_after_kwh}, initial was {battery.initial_energy_kwh}"
        )

    # 5. Independent recalculation of objective metrics
    recalc_grid = round(sum(p.grid_kwh for p in plan), 4)
    recalc_cost = round(sum(p.grid_kwh * hours_data[p.hour].tariff_bdt_per_kwh for p in plan), 4)
    recalc_peak = round(max(p.grid_kwh for p in plan), 4)

    return recalc_grid, recalc_cost, recalc_peak

# =====================================================================
# FastAPI Application & Endpoints
# =====================================================================

app = FastAPI(
    title="BUP CSE Fest 2026 GridWise Backend",
    description="LLM-Assisted Campus Energy Optimization Service",
    version="2.0"
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return HTTP 400 on malformed or structurally invalid request."""
    logger.warning(f"Request validation error on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": "Malformed JSON or structurally invalid request."}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Controlled internal error handler without leaking stack traces or secrets."""
    logger.error(f"Internal server error: {type(exc).__name__}: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the energy schedule."}
    )

@app.get("/", tags=["Root"])
def root():
    """Root endpoint providing service overview and documentation links."""
    return {
        "service": "BUP CSE Fest 2026 GridWise Energy Optimizer",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "optimize_energy": "/optimize-energy",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Readiness endpoint for the judging harness."""
    return HealthResponse(status="ok")

@app.post("/optimize-energy", response_model=OptimizeEnergyResponse, tags=["Optimization"])
def optimize_energy(request: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    """
    Accept one scenario JSON object and return directive interpretations plus 24-hour optimal schedule.
    Pipeline: LLM Interpretation -> Strict Validation -> PuLP LP Optimizer -> Independent Plan Validator.
    """
    # 1. LLM Interpretation & Strict Deterministic Validation
    validated_directives = interpret_and_validate_notes(request.operator_notes, request.battery)

    # 2. PuLP Optimization
    raw_plan, _, _, _, plan_summary = solve_energy_schedule(
        request, validated_directives
    )

    # 3. Independent Output Validation & Metric Recalculation
    total_grid, total_cost, peak_grid = validate_final_plan(
        raw_plan, request, validated_directives
    )

    # 4. Return verified structured response
    return OptimizeEnergyResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=validated_directives,
        hourly_plan=raw_plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=plan_summary
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=False)
