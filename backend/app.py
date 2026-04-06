from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import os
import logging
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import re
from openai import AzureOpenAI
from dotenv import load_dotenv
import requests

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ClearPath")

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4-1-mini")

if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
    logger.warning("Azure OpenAI credentials not configured. Summary functionality will be disabled.")
    azure_client = None
else:
    azure_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-02-01",
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

app = FastAPI(title="ClearPath Voicebot - Clinical Triage")

WEBHOOK_DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "webhook_debug.json")
MAX_WEBHOOK_DEBUG_EVENTS = 10
CALLS_DB_PATH = os.path.join(os.path.dirname(__file__), "calls.db")


def init_calls_db() -> None:
    """Create calls.db and call_logs table if it does not exist."""
    conn = sqlite3.connect(CALLS_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS call_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT,
                timestamp TEXT,
                caller_name TEXT,
                caller_phone TEXT,
                transcript TEXT,
                category TEXT,
                action_taken TEXT,
                language TEXT,
                booking_status TEXT
            )
            """
        )

        cursor.execute("PRAGMA table_info(call_logs)")
        existing_columns = {row[1] for row in cursor.fetchall() if len(row) > 1}
        if "call_id" not in existing_columns:
            cursor.execute("ALTER TABLE call_logs ADD COLUMN call_id TEXT")

        cursor.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_call_logs_call_id_unique
            ON call_logs(call_id)
            WHERE call_id IS NOT NULL AND call_id != ''
            """
        )

        conn.commit()
    finally:
        conn.close()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def insert_call_log_db(
    call_id: Any,
    timestamp: Any,
    caller_name: Any,
    caller_phone: Any,
    transcript: Any,
    category: Any,
    action_taken: Any,
    language: Any,
    booking_status: Any,
) -> bool:
    """Insert one webhook call record into SQLite."""
    conn = sqlite3.connect(CALLS_DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO call_logs (
                call_id,
                timestamp,
                caller_name,
                caller_phone,
                transcript,
                category,
                action_taken,
                language,
                booking_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _safe_text(call_id),
                _safe_text(timestamp),
                _safe_text(caller_name),
                _safe_text(caller_phone),
                _safe_text(transcript),
                _safe_text(category),
                _safe_text(action_taken),
                _safe_text(language),
                _safe_text(booking_status),
            ),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


init_calls_db()

# Models
class WebhookResponse(BaseModel):
    action: str
    category: int


class ScheduleBookingRequest(BaseModel):
    patient_name: str
    patient_email: str
    start_time: str
    timezone: str = "America/New_York"
    phone: Optional[str] = None
    notes: Optional[str] = None


class ScheduleBookingResponse(BaseModel):
    success: bool
    message: str
    booking_reference: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


class TriageRequest(BaseModel):
    name: str
    symptoms: str
    email: Optional[str] = None
    phone: Optional[str] = None
    category: Optional[int] = None
    action: Optional[str] = None
    booking_start_time: Optional[str] = None
    timezone: str = "America/New_York"
    notes: Optional[str] = None


class TriageResponse(BaseModel):
    category: int
    category_name: str
    recommended_action: str
    symptoms_detected: list[str]
    escalation_required: bool
    next_available_offered: bool
    booking_attempt: Dict[str, Any]
    message: str


ACTION_TO_CATEGORY: Dict[str, int] = {
    "emergency_transfer": 1,
    "same_day_appointment": 2,
    "schedule_72hr": 3,
}


def collect_dict_candidates(payload: Any, max_nodes: int = 80) -> list[Dict[str, Any]]:
    """
    Collect dictionary candidates from nested payload structures.
    This helps normalize webhook shapes where fields can appear in envelope objects.
    """
    candidates: list[Dict[str, Any]] = []
    queue: list[Any] = [payload]
    seen: set[int] = set()

    while queue and len(candidates) < max_nodes:
        current = queue.pop(0)
        if isinstance(current, dict):
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)
            candidates.append(current)
            for value in current.values():
                if isinstance(value, (dict, list)):
                    queue.append(value)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    queue.append(item)

    return candidates


def append_webhook_debug_event(timestamp: str, payload: Dict[str, Any], call_id: Optional[str], call_completed: bool) -> None:
    """Persist recent webhook payloads for debugging and keep only the latest N events."""
    event = {
        "timestamp": timestamp,
        "call_id": call_id,
        "call_completed": call_completed,
        "payload": payload,
    }

    try:
        if os.path.exists(WEBHOOK_DEBUG_LOG_PATH):
            with open(WEBHOOK_DEBUG_LOG_PATH, "r") as f:
                existing = json.load(f)
                events = existing if isinstance(existing, list) else []
        else:
            events = []

        events.append(event)
        events = events[-MAX_WEBHOOK_DEBUG_EVENTS:]

        with open(WEBHOOK_DEBUG_LOG_PATH, "w") as f:
            json.dump(events, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to store webhook debug event: {str(e)}")


def read_webhook_debug_events(limit: int = 10) -> list[Dict[str, Any]]:
    """Read recent webhook debug events from disk."""
    safe_limit = max(1, min(limit, 100))
    if not os.path.exists(WEBHOOK_DEBUG_LOG_PATH):
        return []

    try:
        with open(WEBHOOK_DEBUG_LOG_PATH, "r") as f:
            events = json.load(f)
        if not isinstance(events, list):
            return []
        return events[-safe_limit:]
    except Exception as e:
        logger.warning(f"Failed to read webhook debug events: {str(e)}")
        return []

def extract_call_fields(payload: Dict[str, Any]) -> Dict[str, str]:
    """Extract transcript and caller number from common webhook payload shapes."""
    transcript = payload.get("transcript")
    caller_number = payload.get("caller_number")

    # Common alternative keys
    if not transcript:
        message_value = payload.get("message")
        transcript = message_value if isinstance(message_value, str) else payload.get("text")
    if not caller_number:
        caller_number = payload.get("phone_number") or payload.get("from")

    # Nested call object shapes
    call_obj = payload.get("call", {}) if isinstance(payload.get("call"), dict) else {}
    if not caller_number and call_obj:
        caller_number = call_obj.get("from") or call_obj.get("phoneNumber")

    # Transcript nested under message object
    message_obj = payload.get("message", {}) if isinstance(payload.get("message"), dict) else {}
    if not transcript and message_obj:
        transcript = message_obj.get("transcript") or message_obj.get("text")

    # Fallback normalization
    if not isinstance(transcript, str):
        transcript = ""
    if not isinstance(caller_number, str):
        caller_number = "unknown"

    transcript = transcript.strip()
    caller_number = caller_number.strip() or "unknown"

    return {"transcript": transcript, "caller_number": caller_number}


def extract_call_id(payload: Dict[str, Any]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None

    candidates = collect_dict_candidates(payload)

    for candidate_payload in candidates:
        candidate = candidate_payload.get("call_id") or candidate_payload.get("callId")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for candidate_payload in candidates:
        call_obj = candidate_payload.get("call")
        if isinstance(call_obj, dict):
            call_candidate = call_obj.get("id") or call_obj.get("callId") or call_obj.get("call_id")
            if isinstance(call_candidate, str) and call_candidate.strip():
                return call_candidate.strip()

    return None


def infer_call_completed(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False

    status_candidates = []
    for candidate_payload in collect_dict_candidates(payload):
        status_candidates.extend([
            candidate_payload.get("event"),
            candidate_payload.get("type"),
            candidate_payload.get("status"),
        ])

    normalized = " ".join([str(item).lower() for item in status_candidates if item is not None])
    if not normalized:
        return False

    completed_markers = [
        "end-of-call",
        "end-of-call-report",
        "ended",
        "completed",
        "conversation.completed",
        "call.ended",
        "call-ended",
    ]
    if any(marker in normalized for marker in completed_markers):
        return True

    progress_markers = [
        "in-progress",
        "in_progress",
        "started",
        "ongoing",
        "conversation-update",
        "speech-update",
        "status-update",
        "transcript-update",
    ]
    if any(marker in normalized for marker in progress_markers):
        return False

    return False

def map_category_to_tool(category: int) -> str:
    """Map triage category to Vapi tool name."""
    if category == 1:
        return "emergency_transfer"
    if category == 2:
        return "same_day_appointment"
    return "schedule_72hr"


def parse_category_value(value: Any) -> Optional[int]:
    if isinstance(value, int) and value in (1, 2, 3):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.isdigit():
            parsed = int(cleaned)
            if parsed in (1, 2, 3):
                return parsed
    return None


def normalize_action_name(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in ACTION_TO_CATEGORY:
        return normalized

    alias_map = {
        "emergency": "emergency_transfer",
        "urgent": "same_day_appointment",
        "same_day": "same_day_appointment",
        "within_72_hours": "schedule_72hr",
        "72hr": "schedule_72hr",
        "72_hours": "schedule_72hr",
        "moderate": "schedule_72hr",
    }
    return alias_map.get(normalized)


def extract_vapi_tool_action(candidates: list[Dict[str, Any]]) -> Optional[str]:
    for candidate in candidates:
        for key in ("action", "recommended_action", "tool", "toolName", "name"):
            normalized = normalize_action_name(candidate.get(key))
            if normalized:
                return normalized

    for candidate in candidates:
        tool_calls = candidate.get("toolCalls") or candidate.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            direct_name = normalize_action_name(tool_call.get("name"))
            if direct_name:
                return direct_name
            function_payload = tool_call.get("function")
            if isinstance(function_payload, dict):
                function_name = normalize_action_name(function_payload.get("name"))
                if function_name:
                    return function_name

    return None


def create_cal_booking(request_data: ScheduleBookingRequest, booking_window: str = "same_day") -> Dict[str, Any]:
    """
    Create a booking in Cal.com via backend-only API call.
    Uses environment variables so API keys are never exposed in frontend clients.
    """
    cal_api_key = os.getenv("CAL_API_KEY")
    cal_event_type_id_default = os.getenv("CAL_EVENT_TYPE_ID")
    cal_event_type_id_same_day = os.getenv("CAL_EVENT_TYPE_ID_SAME_DAY")
    cal_event_type_id_72_hour = os.getenv("CAL_EVENT_TYPE_ID_72_HOUR")
    cal_api_base_url = os.getenv("CAL_API_BASE_URL", "https://api.cal.com/v1")
    cal_booking_language = os.getenv("CAL_BOOKING_LANGUAGE", "en")

    cal_event_type_id = cal_event_type_id_default
    if booking_window == "same_day":
        cal_event_type_id = cal_event_type_id_same_day or cal_event_type_id_default
    elif booking_window == "within_72_hours":
        cal_event_type_id = cal_event_type_id_72_hour or cal_event_type_id_default

    if not cal_api_key:
        raise HTTPException(status_code=500, detail="CAL_API_KEY is not configured")
    if not cal_event_type_id:
        raise HTTPException(status_code=500, detail="CAL_EVENT_TYPE_ID is not configured")

    url = f"{cal_api_base_url}/bookings"
    payload = {
        "eventTypeId": int(cal_event_type_id),
        "start": request_data.start_time,
        "language": cal_booking_language,
        "responses": {
            "name": request_data.patient_name,
            "email": request_data.patient_email,
        },
        "timeZone": request_data.timezone,
        "metadata": {
            "phone": request_data.phone,
            "notes": request_data.notes,
        },
    }

    try:
        response = requests.post(
            url,
            params={"apiKey": cal_api_key},
            json=payload,
            timeout=20,
        )
    except requests.RequestException as e:
        logger.error(f"Cal.com request failed: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Scheduling provider connection failed: {str(e)}")

    if response.status_code >= 400:
        logger.error(f"Cal.com booking failed: status={response.status_code}, body={response.text}")
        raise HTTPException(status_code=502, detail=f"Scheduling provider error ({response.status_code})")

    return response.json()


def attempt_category_booking(
    payload: Dict[str, Any],
    caller_name: str,
    caller_phone: str,
    transcript: str,
    booking_window: str,
) -> Dict[str, Any]:
    """
    Attempts booking for non-emergency categories.
    Supports simulation mode for safe testing.
    """
    simulation_mode = os.getenv("BOOKING_SIMULATION_MODE", "off").strip().lower()

    booking_log: Dict[str, Any] = {
        "attempted": True,
        "window": booking_window,
        "simulation_mode": simulation_mode,
        "success": False,
        "reason": None,
        "provider_status": None,
        "provider_response": None,
    }

    if simulation_mode in {"success", "fail", "rate_limit", "unexpected"}:
        if simulation_mode == "success":
            booking_log["success"] = True
            booking_log["reason"] = "simulated_success"
            booking_log["provider_response"] = {
                "booking": {
                    "id": "sim-booking-001",
                    "uid": "sim-uid-001",
                }
            }
            return booking_log
        if simulation_mode == "rate_limit":
            booking_log["reason"] = "simulated_rate_limit"
            booking_log["provider_status"] = 429
            return booking_log
        if simulation_mode == "unexpected":
            booking_log["reason"] = "simulated_unexpected_response"
            booking_log["provider_response"] = {"unexpected": True}
            return booking_log

        booking_log["reason"] = "simulated_failure"
        booking_log["provider_status"] = 500
        return booking_log

    patient_email = payload.get("patient_email") or payload.get("email")
    booking_start_time = payload.get("booking_start_time") or payload.get("start_time")
    timezone = payload.get("timezone") or "America/New_York"
    notes = payload.get("notes")

    if not isinstance(patient_email, str) or not patient_email.strip() or not isinstance(booking_start_time, str) or not booking_start_time.strip():
        booking_log["reason"] = "missing_patient_email_or_start_time"
        return booking_log

    request_data = ScheduleBookingRequest(
        patient_name=caller_name,
        patient_email=patient_email.strip(),
        start_time=booking_start_time.strip(),
        timezone=timezone.strip() if isinstance(timezone, str) and timezone.strip() else "America/New_York",
        phone=caller_phone,
        notes=notes if isinstance(notes, str) else transcript[:500],
    )

    try:
        provider_response = create_cal_booking(request_data, booking_window=booking_window)
        booking_log["success"] = True
        booking_log["reason"] = "booking_created"
        booking_log["provider_response"] = provider_response
        return booking_log
    except HTTPException as e:
        booking_log["reason"] = str(e.detail)
        booking_log["provider_status"] = e.status_code
        return booking_log
    except Exception as e:
        booking_log["reason"] = f"unexpected_error: {str(e)}"
        booking_log["provider_status"] = 500
        return booking_log


def resolve_control_flow(
    category: int,
    payload: Dict[str, Any],
    caller_name: str,
    caller_phone: str,
    transcript: str,
) -> Dict[str, Any]:
    """
    Refined safe control flow:
    - Category 1: No booking, escalate only
    - Category 2: Try same-day booking, if fail escalate
    - Category 3: Try 72-hour booking, if fail offer next available/log
    """
    if category == 1:
        return {
            "final_action": "emergency_transfer",
            "booking": {
                "attempted": False,
                "success": False,
                "reason": "category_1_no_booking_escalate_only",
            },
            "escalation_required": True,
            "next_available_offered": False,
        }

    if category == 2:
        booking = attempt_category_booking(payload, caller_name, caller_phone, transcript, booking_window="same_day")
        if booking.get("success"):
            return {
                "final_action": "same_day_appointment",
                "booking": booking,
                "escalation_required": False,
                "next_available_offered": False,
            }
        if booking.get("reason") == "missing_patient_email_or_start_time":
            return {
                "final_action": "same_day_appointment",
                "booking": booking,
                "escalation_required": False,
                "next_available_offered": False,
            }
        return {
            "final_action": "emergency_transfer",
            "booking": booking,
            "escalation_required": True,
            "next_available_offered": False,
        }

    booking = attempt_category_booking(payload, caller_name, caller_phone, transcript, booking_window="within_72_hours")
    if booking.get("success"):
        return {
            "final_action": "schedule_72hr",
            "booking": booking,
            "escalation_required": False,
            "next_available_offered": False,
        }
    if booking.get("reason") == "missing_patient_email_or_start_time":
        return {
            "final_action": "schedule_72hr",
            "booking": booking,
            "escalation_required": False,
            "next_available_offered": False,
        }

    return {
        "final_action": "schedule_72hr",
        "booking": booking,
        "escalation_required": False,
        "next_available_offered": True,
    }

# Load triage rules at startup
def load_triage_rules() -> Dict[str, Any]:
    """Load triage rules from JSON file."""
    rules_path = os.path.join(os.path.dirname(__file__), "triage_rules.json")
    try:
        with open(rules_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise RuntimeError(f"triage_rules.json not found at {rules_path}")
    except json.JSONDecodeError:
        raise RuntimeError("triage_rules.json is not valid JSON")

# Load rules on startup
triage_rules = load_triage_rules()

def normalize_transcript(text: str) -> str:
    """
    Normalize transcript for matching:
    - Lowercase
    - Remove punctuation (commas, periods, quotes, etc.)
    - Collapse multiple spaces to single space
    - Strip leading/trailing whitespace
    """
    import string
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Lowercase
    text = text.lower()
    # Collapse multiple spaces
    text = ' '.join(text.split())
    # Strip whitespace
    text = text.strip()
    return text


def extract_user_lines(transcript: str) -> list[str]:
    if not isinstance(transcript, str) or not transcript.strip():
        return []

    lines = [line.strip() for line in transcript.splitlines() if isinstance(line, str) and line.strip()]
    user_lines: list[str] = []
    for line in lines:
        lower = line.lower()
        if lower.startswith("user:"):
            content = line.split(":", 1)[1].strip()
            if content:
                user_lines.append(content)
    return user_lines


def get_user_focused_transcript(transcript: str) -> str:
    user_lines = extract_user_lines(transcript)
    if user_lines:
        return " ".join(user_lines)
    return transcript if isinstance(transcript, str) else ""


def build_symptom_summary(transcript: str) -> str:
    user_lines = extract_user_lines(transcript)
    if not user_lines:
        fallback_excerpt = build_conversation_excerpt(transcript)
        if fallback_excerpt:
            return fallback_excerpt
        return transcript.strip() if isinstance(transcript, str) else ""

    number_words = {
        "zero", "oh", "o", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    }

    filtered: list[str] = []
    for line in user_lines:
        line_normalized = line.strip()
        lower = line_normalized.lower()
        tokens = re.findall(r"[A-Za-z0-9]+", lower)

        if any(phrase in lower for phrase in [
            "my name is",
            "it's",
            "it is",
            "my callback",
            "my phone",
            "reach me at",
        ]):
            continue

        if len(line_normalized) <= 2:
            continue

        if 1 <= len(tokens) <= 3 and all(re.fullmatch(r"[a-z]+", token or "") for token in tokens):
            continue

        if tokens and all(token.isdigit() or token in number_words for token in tokens):
            continue

        filtered.append(line_normalized)

    if not filtered:
        filtered = user_lines[:2]

    summary_text = " ".join(filtered[:3]).strip()
    summary_lower = summary_text.lower()
    weak_summary_markers = [
        "no symptoms",
        "not really feeling any symptoms",
        "i just wanna",
        "can i see the doctor",
    ]

    if not summary_text or any(marker in summary_lower for marker in weak_summary_markers):
        fallback_excerpt = build_conversation_excerpt(transcript)
        if fallback_excerpt:
            return fallback_excerpt

    return summary_text


def build_conversation_excerpt(transcript: str, max_turns: int = 6) -> str:
    if not isinstance(transcript, str) or not transcript.strip():
        return ""

    lines = [line.strip() for line in transcript.splitlines() if isinstance(line, str) and line.strip()]
    turns: list[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("ai:") or lowered.startswith("assistant:"):
            turns.append(f"AI: {line.split(':', 1)[1].strip()}")
        elif lowered.startswith("user:") or lowered.startswith("patient:"):
            turns.append(f"User: {line.split(':', 1)[1].strip()}")

    if not turns:
        return ""

    return " | ".join(turns[:max_turns])


def is_non_patient_name(value: Optional[str]) -> bool:
    if not isinstance(value, str):
        return True

    normalized = value.strip().lower()
    if not normalized:
        return True

    blocked_names = {
        "unknown",
        "clearpath",
        "clearpath_demo",
        "demo",
        "assistant",
        "triage bot",
        "voicebot",
        "bot",
        "ai",
        "p",
    }
    if normalized in blocked_names:
        return True

    blocked_fragments = ["clearpath", "assistant", "voicebot", "demo", "bot"]
    if any(fragment in normalized for fragment in blocked_fragments):
        return True

    blocked_tokens = {"forwarding", "call", "triage", "nurse", "reached", "services", "hello"}
    normalized_tokens = set(re.findall(r"[a-z]+", normalized))
    if normalized_tokens and any(token in blocked_tokens for token in normalized_tokens):
        return True

    if not re.fullmatch(r"[A-Za-z'\- ]{1,60}", value.strip()):
        return True

    return False


def infer_name_from_transcript(transcript: str) -> Optional[str]:
    if not isinstance(transcript, str) or not transcript.strip():
        return None

    extraction_patterns = [
        r"\bmy name is\s+([A-Za-z][A-Za-z'\- ]{0,40})",
        r"\bi(?:'m| am)\s+([A-Za-z][A-Za-z'\- ]{0,40})",
        r"\bthis is\s+([A-Za-z][A-Za-z'\- ]{0,40})",
        r"\bme llamo\s+([A-Za-z][A-Za-z'\- ]{0,40})",
        r"\bmi nombre es\s+([A-Za-z][A-Za-z'\- ]{0,40})",
        r"\bsoy\s+([A-Za-z][A-Za-z'\- ]{0,40})",
    ]

    def sanitize_name_candidate(candidate: str) -> str:
        if not isinstance(candidate, str):
            return ""
        trimmed = candidate.strip(" .,!?")
        if not trimmed:
            return ""
        trimmed = re.split(r"\b(and|y|because|que|i|tengo|had|tuve|accidente|sintomas|síntomas)\b", trimmed, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,!?")
        return " ".join(trimmed.split()[:3])

    user_lines = extract_user_lines(transcript)
    user_joined = " ".join(user_lines)
    if user_joined:
        for pattern in extraction_patterns:
            name_match = re.search(pattern, user_joined, flags=re.IGNORECASE)
            if not name_match:
                continue

            candidate = sanitize_name_candidate(name_match.group(1))
            if not is_non_patient_name(candidate):
                return " ".join(word.capitalize() for word in candidate.split())

    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if ("name" in line.lower() or "nombre" in line.lower()) and idx + 1 < len(lines):
            next_line = lines[idx + 1]
            if next_line.lower().startswith("user:"):
                possible_name = next_line.split(":", 1)[1].strip(" .,!?")
                if 1 <= len(possible_name.split()) <= 3 and re.fullmatch(r"[A-Za-z'\- ]+", possible_name):
                    if not is_non_patient_name(possible_name):
                        return " ".join(word.capitalize() for word in possible_name.split())

    for pattern in extraction_patterns:
        name_match = re.search(pattern, transcript, flags=re.IGNORECASE)
        if not name_match:
            continue
        candidate = sanitize_name_candidate(name_match.group(1))
        if not is_non_patient_name(candidate):
            return " ".join(word.capitalize() for word in candidate.split())

    return None


def infer_phone_from_transcript(transcript: str) -> Optional[str]:
    if not isinstance(transcript, str) or not transcript.strip():
        return None

    number_words = {
        "zero": "0", "oh": "0", "o": "0",
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "cero": "0",
        "uno": "1", "dos": "2", "tres": "3", "cuatro": "4", "cinco": "5",
        "seis": "6", "siete": "7", "ocho": "8", "nueve": "9",
    }

    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    candidate_lines: list[str] = []
    for idx, line in enumerate(lines):
        lower_line = line.lower()
        if "callback" in lower_line or "phone" in lower_line or "reach you" in lower_line or "numero" in lower_line or "número" in lower_line or "telefono" in lower_line or "teléfono" in lower_line:
            if idx + 1 < len(lines):
                candidate_lines.append(lines[idx + 1])
        if lower_line.startswith("user:"):
            candidate_lines.append(line)

    if not candidate_lines:
        candidate_lines = lines

    for raw_line in candidate_lines:
        line = raw_line.split(":", 1)[1].strip() if ":" in raw_line else raw_line
        tokens = re.findall(r"[A-Za-z0-9]+", line.lower())
        digits = ""
        for token in tokens:
            if token.isdigit():
                digits += token
            elif token in number_words:
                digits += number_words[token]
        if len(digits) >= 10:
            return digits[:10]

    return None


def infer_phone_words_from_transcript(transcript: str) -> Optional[str]:
    if not isinstance(transcript, str) or not transcript.strip():
        return None

    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        lower = line.lower()
        if "callback" in lower or "phone" in lower or "reach" in lower or "numero" in lower or "número" in lower or "telefono" in lower or "teléfono" in lower:
            if idx + 1 < len(lines):
                candidate = lines[idx + 1]
                if candidate.lower().startswith("user:"):
                    candidate = candidate.split(":", 1)[1].strip()
                if candidate:
                    return candidate

    for line in lines:
        lower = line.lower()
        if lower.startswith("user:") and any(word in lower for word in ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "oh"]):
            return line.split(":", 1)[1].strip()

    return None

def classify_transcript(transcript: str) -> Dict[str, Any]:
    """
    Classify transcript using triage_rules.json with normalized full-phrase matching.
    
    Enhanced with:
    - Transcript and symptom normalization for robust matching
    - Full phrase matching ONLY (no single-word extraction from parentheses)
    - Matched symptoms array for audit trail
    - Debug logging for troubleshooting
    - Ordered category checking (1 → 2 → 3) for highest urgency first

    Returns:
    {
      "category": <category_number>,
      "action": <category_action>,
      "matched_symptoms": [<symptom_text>, ...]
    }
    """
    user_focused_transcript = get_user_focused_transcript(transcript)
    # Normalize transcript for matching
    transcript_normalized = normalize_transcript(user_focused_transcript)
    logger.info(f"[CLASSIFY] Original transcript: {transcript!r}")
    logger.info(f"[CLASSIFY] User-focused transcript: {user_focused_transcript!r}")
    logger.info(f"[CLASSIFY] Normalized transcript: {transcript_normalized!r}")

    # Track all matched symptoms found during classification
    matched_symptoms = []

    emergency_regex_patterns = [
        r"\b(accident|car accident|car crash|crash|freeway|highway collision|motor vehicle collision|hit by)\b",
        r"\b(uncontrolled bleeding|bleeding heavily|severe bleeding)\b",
        r"\b(passed out|unconscious|not breathing|cant breathe|can't breathe)\b",
        r"\b(dolor de pecho|presion en el pecho|presión en el pecho|falta de aire|dificultad para respirar|accidente|choque|colision|colisión|sangrado abundante|inconsciente|convulsiones)\b",
    ]
    for pattern in emergency_regex_patterns:
        if re.search(pattern, transcript_normalized):
            category_1 = next(
                (c for c in triage_rules.get("categories", []) if c.get("category") == 1),
                None,
            )
            logger.info(f"[CLASSIFY] Heuristic emergency match pattern={pattern!r} -> Category 1")
            return {
                "category": 1,
                "action": category_1.get("action", "Warm hand-off to triage nurse.") if category_1 else "Warm hand-off to triage nurse.",
                "matched_symptoms": ["Major trauma / emergency incident"],
            }

    # Categories are checked in order (1, 2, 3) - highest urgency first
    # We check all categories to find any matches, then return highest urgency found
    for category in triage_rules.get("categories", []):
        category_number = category.get("category")
        category_action = category.get("action", "")
        symptoms = category.get("symptoms", [])
        
        logger.info(f"[CLASSIFY] Checking Category {category_number}")

        for symptom in symptoms:
            # Normalize symptom the same way as transcript
            symptom_normalized = normalize_transcript(symptom)
            logger.info(f"[CLASSIFY]   Symptom: {symptom!r}")
            logger.info(f"[CLASSIFY]   Normalized: {symptom_normalized!r}")

            # Generate candidates from symptom variants (/) only
            # Do NOT extract single words from parentheses - match full phrase only
            candidates = [symptom_normalized]
            
            # Handle "/" variants (e.g., "Chest pain/Chest pressure")
            if "/" in symptom:
                parts = symptom.split("/")
                for part in parts:
                    normalized_part = normalize_transcript(part.strip())
                    if normalized_part and normalized_part not in candidates:
                        candidates.append(normalized_part)
                        logger.info(f"[CLASSIFY]     Candidate (from /): {normalized_part!r}")

            # Match full phrase only (exactly as defined symptom appears in transcript)
            for candidate in candidates:
                if candidate and candidate in transcript_normalized:
                    logger.info(f"[CLASSIFY] ✓ MATCH FOUND: '{candidate}' found in normalized transcript")
                    logger.info(f"[CLASSIFY] → Matched symptom: {symptom!r}")
                    if symptom not in matched_symptoms:
                        matched_symptoms.append(symptom)
                    # Return immediately on first match (highest urgency first due to category order)
                    logger.info(f"[CLASSIFY] → Returning Category {category_number} with action: {category_action!r}")
                    logger.info(f"FINAL CATEGORY: {category_number}")
                    return {
                        "category": category_number,
                        "action": category_action,
                        "matched_symptoms": matched_symptoms,
                    }

    # Default to Category 3 (no match)
    spanish_urgent_terms = [
        "palpitaciones",
        "fiebre alta",
        "dolor abdominal",
        "vision borrosa",
        "visión borrosa",
        "sangre en la orina",
        "incontinencia",
        "dolor testicular",
        "coagulo",
        "coágulo",
    ]
    if any(term in transcript_normalized for term in spanish_urgent_terms):
        category_2 = next(
            (c for c in triage_rules.get("categories", []) if c.get("category") == 2),
            None,
        )
        logger.info("[CLASSIFY] Spanish urgent terms matched -> Category 2")
        return {
            "category": 2,
            "action": category_2.get("action", "If none is available, contact/leave a message for the triage nurse.") if category_2 else "If none is available, contact/leave a message for the triage nurse.",
            "matched_symptoms": ["Spanish urgent symptom"],
        }

    spanish_moderate_terms = [
        "tos persistente",
        "ardor al orinar",
        "sarpullido",
        "dolor de espalda",
        "dolor articular",
        "migraña",
        "migrana",
    ]
    if any(term in transcript_normalized for term in spanish_moderate_terms):
        category_3 = next(
            (c for c in triage_rules.get("categories", []) if c.get("category") == 3),
            None,
        )
        logger.info("[CLASSIFY] Spanish moderate terms matched -> Category 3")
        return {
            "category": 3,
            "action": category_3.get("action", "continue") if category_3 else "continue",
            "matched_symptoms": ["Spanish moderate symptom"],
        }

    if "fever" in transcript_normalized and "vomiting" in transcript_normalized:
        category_2 = next(
            (c for c in triage_rules.get("categories", []) if c.get("category") == 2),
            None,
        )
        logger.info("[CLASSIFY] Heuristic match: fever + vomiting -> Category 2")
        return {
            "category": 2,
            "action": category_2.get("action", "If none is available, contact/leave a message for the triage nurse.") if category_2 else "If none is available, contact/leave a message for the triage nurse.",
            "matched_symptoms": ["Fever with vomiting"],
        }

    if ("fiebre" in transcript_normalized and ("vomito" in transcript_normalized or "vomitos" in transcript_normalized or "vomitando" in transcript_normalized)):
        category_2 = next(
            (c for c in triage_rules.get("categories", []) if c.get("category") == 2),
            None,
        )
        logger.info("[CLASSIFY] Heuristic Spanish match: fiebre + vomito -> Category 2")
        return {
            "category": 2,
            "action": category_2.get("action", "If none is available, contact/leave a message for the triage nurse.") if category_2 else "If none is available, contact/leave a message for the triage nurse.",
            "matched_symptoms": ["Fiebre con vómitos"],
        }

    category_3 = next(
        (c for c in triage_rules.get("categories", []) if c.get("category") == 3),
        None,
    )
    logger.info(f"[CLASSIFY] ✗ No symptoms matched; defaulting to Category 3")
    logger.info(f"[CLASSIFY] → Returning Category 3 with action: {category_3.get('action') if category_3 else 'continue'!r}")
    logger.info(f"FINAL CATEGORY: 3")
    return {
        "category": 3,
        "action": category_3.get("action", "continue") if category_3 else "continue",
        "matched_symptoms": matched_symptoms,
    }

def generate_summary(transcript: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a structured summary from the transcript using Azure OpenAI.
    
    Extracts:
    - patient_name
    - category
    - symptoms
    - action
    
    Returns a dictionary with the extracted information.
    """
    if not azure_client:
        logger.warning("Azure OpenAI not configured. Returning minimal summary.")
        return {
            "patient_name": "Unknown Patient",
            "symptoms": [],
            "category": classification.get("category", 3),
            "action": classification.get("action", "")
        }

    system_prompt = f"""You are a clinical triage assistant. Use the triage rules below.

STRICT MAPPING: You must first identify the Category (1, 2, or 3) based on the triage_rules.json.

MATCHING ACTION: The 'action' must ONLY be the exact 'action' string from that specific category in the JSON. Do not summarize or combine categories.

OUTPUT FORMAT: Return a clean JSON object only with:
{{
    "patient_name": "string",
    "category": 1,
    "symptoms": ["..."],
    "action": "exact action string from matched category"
}}

FALLBACK: If patient name is not found, use "Unknown Patient".

Return JSON only. No markdown. No explanation.

TRIAGE_RULES_JSON:
{json.dumps(triage_rules)}
"""

    try:
        response = azure_client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Transcript:\n{transcript}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Extract JSON from response
        response_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        # Parse JSON
        summary = json.loads(response_text)
        
        # Enforce strict mapping from rule-based backend classification
        summary["category"] = classification.get("category", 3)
        summary["action"] = classification.get("action", "")
        if not summary.get("patient_name"):
            summary["patient_name"] = "Unknown Patient"
        if not isinstance(summary.get("symptoms"), list):
            summary["symptoms"] = []

        logger.info(f"Summary generated: Category {summary.get('category')}, Action: {summary.get('action')}")
        
        return summary
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from Azure OpenAI response: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse AI response")
    except Exception as e:
        logger.error(f"Error calling Azure OpenAI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


def log_call(call_data: Dict[str, Any]) -> None:
    """
    Append call data to call_logs.json.
    Creates file if it doesn't exist.
    """
    logs_path = os.path.join(os.path.dirname(__file__), "call_logs.json")
    
    try:
        def merge_call_entries(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            merged.update(incoming)

            existing_category = parse_category_value(existing.get("category"))
            incoming_category = parse_category_value(incoming.get("category"))
            if existing_category and incoming_category:
                merged["category"] = min(existing_category, incoming_category)
            elif existing_category and not incoming_category:
                merged["category"] = existing_category

            final_category = parse_category_value(merged.get("category")) or 3
            merged["category_name"] = {1: "Emergency", 2: "Urgent", 3: "Moderate"}.get(final_category, "Moderate")

            existing_action = normalize_action_name(existing.get("recommended_action"))
            incoming_action = normalize_action_name(incoming.get("recommended_action"))
            merged_action = incoming_action or existing_action
            if merged_action:
                merged["recommended_action"] = merged_action

            if merged.get("recommended_action") == "schedule_72hr" and final_category == 1:
                merged["recommended_action"] = "emergency_transfer"
            elif merged.get("recommended_action") == "schedule_72hr" and final_category == 2:
                merged["recommended_action"] = "same_day_appointment"

            existing_name = existing.get("caller_name")
            incoming_name = incoming.get("caller_name")
            if isinstance(existing_name, str) and existing_name.strip() and not is_non_patient_name(existing_name):
                if not (isinstance(incoming_name, str) and incoming_name.strip() and not is_non_patient_name(incoming_name)):
                    merged["caller_name"] = existing_name

            merged_name = merged.get("caller_name")
            if isinstance(merged_name, str) and merged_name.strip() and not is_non_patient_name(merged_name):
                merged["caller_first_name"] = merged_name.split()[0]
            else:
                merged["caller_first_name"] = "Unknown"

            existing_phone = existing.get("caller_phone")
            incoming_phone = incoming.get("caller_phone")
            existing_phone_valid = isinstance(existing_phone, str) and existing_phone.strip().lower() not in {"", "unknown", "n/a", "none"}
            incoming_phone_valid = isinstance(incoming_phone, str) and incoming_phone.strip().lower() not in {"", "unknown", "n/a", "none"}
            if existing_phone_valid and not incoming_phone_valid:
                merged["caller_phone"] = existing_phone

            existing_words = existing.get("caller_phone_words")
            incoming_words = incoming.get("caller_phone_words")
            if (not incoming_words) and existing_words:
                merged["caller_phone_words"] = existing_words

            existing_transcript = existing.get("full_transcript")
            incoming_transcript = incoming.get("full_transcript")
            if isinstance(existing_transcript, str) and isinstance(incoming_transcript, str):
                if len(existing_transcript.strip()) > len(incoming_transcript.strip()):
                    merged["full_transcript"] = existing_transcript

            if not isinstance(merged.get("symptoms"), str) or not merged.get("symptoms", "").strip():
                if isinstance(existing.get("symptoms"), str) and existing.get("symptoms", "").strip():
                    merged["symptoms"] = existing.get("symptoms")

            return merged

        # Load existing logs or create empty list
        if os.path.exists(logs_path):
            with open(logs_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []
        
        call_id = call_data.get("call_id") if isinstance(call_data, dict) else None

        updated_existing = False
        if isinstance(call_id, str) and call_id.strip():
            for index in range(len(logs) - 1, -1, -1):
                existing = logs[index]
                if isinstance(existing, dict) and existing.get("call_id") == call_id:
                    merged = merge_call_entries(existing, call_data)
                    merged["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logs[index] = merged
                    updated_existing = True
                    break

        if not updated_existing:
            dedupe_index = None
            for index in range(len(logs) - 1, -1, -1):
                existing = logs[index]
                if not isinstance(existing, dict):
                    continue
                if existing.get("caller_phone") != call_data.get("caller_phone"):
                    continue
                if existing.get("category") != call_data.get("category"):
                    continue
                if existing.get("full_transcript") != call_data.get("full_transcript"):
                    continue

                existing_timestamp = existing.get("timestamp")
                current_timestamp = call_data.get("timestamp")
                try:
                    if isinstance(existing_timestamp, str) and isinstance(current_timestamp, str):
                        existing_dt = datetime.strptime(existing_timestamp, "%Y-%m-%d %H:%M:%S")
                        current_dt = datetime.strptime(current_timestamp, "%Y-%m-%d %H:%M:%S")
                        if abs(current_dt - existing_dt) <= timedelta(minutes=2):
                            dedupe_index = index
                            break
                except ValueError:
                    continue

            if dedupe_index is not None:
                existing = logs[dedupe_index] if isinstance(logs[dedupe_index], dict) else {}
                merged = merge_call_entries(existing, call_data)
                merged["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logs[dedupe_index] = merged
            else:
                if not (isinstance(call_id, str) and call_id.strip()):
                    generated_call_id = f"generated-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                    logger.warning(f"Missing call_id in webhook payload; generated fallback call_id={generated_call_id}")
                    if isinstance(call_data, dict):
                        call_data = dict(call_data)
                        call_data["call_id"] = generated_call_id

                logs.append(call_data)
        
        # Write back to file
        with open(logs_path, "w") as f:
            json.dump(logs, f, indent=2)
        
        logger.info(f"Call logged to call_logs.json")
    except Exception as e:
        logger.error(f"Error logging call: {str(e)}")


@app.post("/webhook", response_model=WebhookResponse)
async def webhook(request: Request) -> WebhookResponse:
    """
    Triage webhook endpoint.
    
    Process flow:
    1. Validate request
    2. Read Vapi-provided transcript/category/action/name/phone
    3. Log call to call_logs.json
    4. Return Vapi-provided action/category
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"[{timestamp}] Invalid webhook JSON payload: {str(e)}")
        payload = {}

    payload = payload if isinstance(payload, dict) else {}
    candidates = collect_dict_candidates(payload)
    call_id = extract_call_id(payload)
    call_completed = infer_call_completed(payload)
    append_webhook_debug_event(timestamp=timestamp, payload=payload, call_id=call_id, call_completed=call_completed)

    def pick_first(keys: tuple[str, ...]) -> Any:
        for candidate_payload in candidates:
            for key in keys:
                if key in candidate_payload:
                    value = candidate_payload.get(key)
                    if value is None:
                        continue
                    if isinstance(value, str) and not value.strip():
                        continue
                    return value
        return None

    def coerce_transcript(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    role = item.get("role") or item.get("speaker") or item.get("from")
                    if isinstance(role, str) and role.strip().lower() in {"system", "tool", "function"}:
                        continue
                    text = item.get("message") or item.get("text") or item.get("content") or item.get("transcript")
                    if isinstance(text, str) and text.strip():
                        normalized_role = role.strip().lower() if isinstance(role, str) else ""
                        if normalized_role in {"bot", "assistant", "ai"}:
                            parts.append(f"AI: {text.strip()}")
                        elif normalized_role in {"user", "caller", "customer"}:
                            parts.append(f"User: {text.strip()}")
                        else:
                            parts.append(text.strip())
            return "\n".join([p for p in parts if p])
        if isinstance(value, dict):
            if isinstance(value.get("messages"), list):
                messages_text = coerce_transcript(value.get("messages"))
                if messages_text:
                    return messages_text
            text = value.get("transcript") or value.get("text") or value.get("message") or value.get("content")
            return text.strip() if isinstance(text, str) else ""
        return ""

    transcript = ""
    transcript_keys = (
        "messages",
        "message",
        "transcript",
        "text",
        "summary",
        "content",
    )
    for candidate_payload in candidates:
        for key in transcript_keys:
            text_candidate = coerce_transcript(candidate_payload.get(key))
            if text_candidate:
                transcript = text_candidate
                break
        if transcript:
            break

    category_raw = pick_first(("category",))
    if category_raw is None:
        for candidate_payload in candidates:
            classification_raw = candidate_payload.get("classification")
            if isinstance(classification_raw, dict) and classification_raw.get("category") is not None:
                category_raw = classification_raw.get("category")
                break
            structured_raw = candidate_payload.get("structuredData")
            if isinstance(structured_raw, dict) and structured_raw.get("category") is not None:
                category_raw = structured_raw.get("category")
                break

    vapi_tool_action = extract_vapi_tool_action(candidates)
    action_raw = pick_first(("action", "recommended_action"))
    if not action_raw:
        for candidate_payload in candidates:
            classification_raw = candidate_payload.get("classification")
            if isinstance(classification_raw, dict) and isinstance(classification_raw.get("action"), str):
                action_raw = classification_raw.get("action")
                break
            structured_raw = candidate_payload.get("structuredData")
            if isinstance(structured_raw, dict) and isinstance(structured_raw.get("action"), str):
                action_raw = structured_raw.get("action")
                break
    if not action_raw and vapi_tool_action:
        action_raw = vapi_tool_action

    caller_name = pick_first(("caller_name", "name", "patient_name"))
    if not caller_name:
        patient_raw = pick_first(("patient",))
        if isinstance(patient_raw, dict):
            caller_name = patient_raw.get("name")

    if isinstance(caller_name, str) and is_non_patient_name(caller_name):
        caller_name = None

    if (not caller_name or not isinstance(caller_name, str) or not caller_name.strip()) and transcript:
        inferred_name = infer_name_from_transcript(transcript)
        if inferred_name:
            caller_name = inferred_name
    caller_name = caller_name.strip() if isinstance(caller_name, str) else "Unknown Patient"

    caller_phone = pick_first(("caller_phone", "phone", "caller_number", "phone_number", "from", "phoneNumber"))
    if not caller_phone:
        call_raw = pick_first(("call",))
        if isinstance(call_raw, dict):
            caller_phone = call_raw.get("from") or call_raw.get("phoneNumber")
    caller_phone_is_placeholder = isinstance(caller_phone, str) and caller_phone.strip().lower() in {"unknown", "n/a", "none"}
    if (not caller_phone or not isinstance(caller_phone, str) or not caller_phone.strip() or caller_phone_is_placeholder) and transcript:
        inferred_phone = infer_phone_from_transcript(transcript)
        if inferred_phone:
            caller_phone = inferred_phone
    caller_phone = caller_phone.strip() if isinstance(caller_phone, str) else "unknown"
    caller_phone_words = None
    if caller_phone == "unknown" and transcript:
        caller_phone_words = infer_phone_words_from_transcript(transcript)

    if not transcript:
        logger.warning(f"[{timestamp}] Missing transcript in request payload; storing minimal webhook event")
        logger.warning(f"[{timestamp}] Payload keys: {list(payload.keys())}")

        fallback_category = 3
        if isinstance(category_raw, int) and category_raw in (1, 2, 3):
            fallback_category = category_raw
        elif isinstance(category_raw, str):
            try:
                parsed_fallback_category = int(category_raw)
                if parsed_fallback_category in (1, 2, 3):
                    fallback_category = parsed_fallback_category
            except ValueError:
                pass

        fallback_action = action_raw.strip() if isinstance(action_raw, str) and action_raw.strip() else map_category_to_tool(fallback_category)

        if isinstance(call_id, str) and call_id.strip():
            if call_completed:
                call_log_entry = {
                    "timestamp": timestamp,
                    "source": "webhook",
                    "call_id": call_id,
                    "call_completed": call_completed,
                    "caller_name": caller_name,
                    "caller_phone": caller_phone,
                    "symptoms": "",
                    "full_transcript": "",
                    "category": fallback_category,
                    "category_name": {1: "Emergency", 2: "Urgent", 3: "Moderate"}.get(fallback_category, "Moderate"),
                    "symptoms_detected": [],
                    "recommended_action": fallback_action,
                    "rule_action": "",
                    "vapi_category": fallback_category,
                    "vapi_action": action_raw.strip() if isinstance(action_raw, str) else None,
                    "category_mismatch": False,
                    "booking_attempt": {},
                    "escalation_required": fallback_category == 1,
                    "next_available_offered": False,
                }
                log_call(call_log_entry)

        return WebhookResponse(action=fallback_action, category=fallback_category)

    backend_classification = classify_transcript(transcript)
    backend_category = int(backend_classification.get("category", 3))

    vapi_category = parse_category_value(category_raw)
    vapi_action = normalize_action_name(action_raw)
    if not vapi_action:
        vapi_action = vapi_tool_action

    if vapi_category is None and vapi_action:
        vapi_category = ACTION_TO_CATEGORY.get(vapi_action)

    final_category = backend_category
    if vapi_category in (1, 2, 3):
        final_category = min(final_category, vapi_category)

    category_name_map = {
        1: "Emergency",
        2: "Urgent",
        3: "Moderate",
    }
    category_name = category_name_map.get(final_category, "Moderate")
    category_mismatch = vapi_category is not None and vapi_category != final_category

    logger.info(f"[{timestamp}] Incoming Vapi call from: {caller_name} ({caller_phone})")
    logger.info(f"[{timestamp}] Transcript: {transcript[:100]}...")
    logger.info(f"[{timestamp}] Vapi category: {vapi_category}")
    logger.info(f"[{timestamp}] Vapi action: {vapi_action}")
    logger.info(f"[{timestamp}] Backend category (rules): {backend_category}")
    logger.info(f"[{timestamp}] Final category (reconciled): {final_category}")
    if category_mismatch:
        logger.warning(f"[{timestamp}] Category mismatch: vapi_category={vapi_category}, backend_category={backend_category}")

    control_flow_result = resolve_control_flow(
        category=final_category,
        payload=payload,
        caller_name=caller_name,
        caller_phone=caller_phone,
        transcript=transcript,
    )
    final_action = control_flow_result.get("final_action", map_category_to_tool(final_category))
    booking_result = control_flow_result.get("booking", {})

    logger.info(f"[{timestamp}] Final action (control flow): {final_action}")
    logger.info(f"[{timestamp}] Booking attempt result: {booking_result}")

    response_payload = {
        "name": caller_name,
        "phone": caller_phone,
        "category": final_category,
        "category_name": category_name,
        "symptoms_detected": backend_classification.get("matched_symptoms", []),
        "recommended_action": final_action,
    }

    call_log_entry = {
        "timestamp": timestamp,
        "source": "webhook",
        "call_id": call_id,
        "call_completed": call_completed,
        "caller_name": response_payload["name"],
        "caller_phone": response_payload["phone"],
        "caller_phone_words": caller_phone_words,
        "caller_first_name": response_payload["name"].split()[0] if isinstance(response_payload["name"], str) and response_payload["name"].strip() else "Unknown",
        "symptoms": build_symptom_summary(transcript),
        "full_transcript": transcript,
        "category": response_payload["category"],
        "category_name": response_payload["category_name"],
        "symptoms_detected": response_payload["symptoms_detected"],
        "recommended_action": response_payload["recommended_action"],
        "rule_action": backend_classification.get("action", ""),
        "vapi_category": vapi_category,
        "vapi_action": vapi_action,
        "category_mismatch": category_mismatch,
        "booking_attempt": booking_result,
        "escalation_required": control_flow_result.get("escalation_required", False),
        "next_available_offered": control_flow_result.get("next_available_offered", False),
    }

    payload_language = pick_first(("language", "lang"))
    if not isinstance(payload_language, str):
        payload_language = ""

    booking_status = ""
    if isinstance(booking_result, dict):
        if booking_result.get("attempted"):
            booking_status = "success" if booking_result.get("success") else "failed"
        else:
            booking_status = "not_attempted"

    if call_completed:
        try:
            inserted = insert_call_log_db(
                call_id=call_id,
                timestamp=timestamp,
                caller_name=response_payload.get("name", ""),
                caller_phone=response_payload.get("phone", ""),
                transcript=transcript,
                category=response_payload.get("category", ""),
                action_taken=response_payload.get("recommended_action", ""),
                language=payload_language,
                booking_status=booking_status,
            )
            if inserted:
                logger.info(f"[{timestamp}] Saved webhook call to calls.db")
            else:
                logger.info(f"[{timestamp}] Skipped duplicate call log for call_id={call_id}")
        except Exception as db_error:
            logger.error(f"[{timestamp}] Failed to save call to SQLite: {str(db_error)}")

        log_call(call_log_entry)
        logger.info(f"[{timestamp}] Logged Vapi payload to call_logs.json")
    else:
        logger.info(f"[{timestamp}] Skipping log (call not completed yet - event type may be in-progress or intermediate update)")
    print(response_payload)
    logger.info(f"[{timestamp}] response={response_payload}")

    return WebhookResponse(action=response_payload["recommended_action"], category=response_payload["category"])


@app.post("/schedule-booking", response_model=ScheduleBookingResponse)
async def schedule_booking(payload: ScheduleBookingRequest) -> ScheduleBookingResponse:
    """
    Creates a booking via Cal.com API from backend.
    This keeps scheduling credentials on the server side only.
    """
    booking_response = create_cal_booking(payload)

    booking_reference = None
    if isinstance(booking_response, dict):
        data = booking_response.get("booking") or booking_response.get("data")
        if isinstance(data, dict):
            booking_reference = str(data.get("uid") or data.get("id") or "") or None

    return ScheduleBookingResponse(
        success=True,
        message="Booking created successfully",
        booking_reference=booking_reference,
        raw_response=booking_response,
    )


@app.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest) -> TriageResponse:
    """
    Main triage endpoint for Vapi tool/function calls.
    Vapi sends patient data + symptoms, backend performs:
    - Rule-based classification from triage_rules.json
    - Safe control-flow (category 1 escalate only, category 2/3 booking attempts)
    - Structured response for voice assistant follow-up
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    transcript = payload.symptoms.strip() if isinstance(payload.symptoms, str) else ""
    if not transcript:
        raise HTTPException(status_code=400, detail="symptoms is required")

    caller_name = payload.name.strip() if isinstance(payload.name, str) and payload.name.strip() else "Unknown Patient"
    caller_phone = payload.phone.strip() if isinstance(payload.phone, str) and payload.phone.strip() else "unknown"

    backend_classification = classify_transcript(transcript)
    backend_category = int(backend_classification.get("category", 3))
    category_name_map = {1: "Emergency", 2: "Urgent", 3: "Moderate"}
    category_name = category_name_map.get(backend_category, "Moderate")

    control_payload = {
        "patient_email": payload.email,
        "start_time": payload.booking_start_time,
        "timezone": payload.timezone,
        "notes": payload.notes,
    }
    control_flow_result = resolve_control_flow(
        category=backend_category,
        payload=control_payload,
        caller_name=caller_name,
        caller_phone=caller_phone,
        transcript=transcript,
    )

    final_action = control_flow_result.get("final_action", map_category_to_tool(backend_category))
    booking_result = control_flow_result.get("booking", {})
    escalation_required = bool(control_flow_result.get("escalation_required", False))
    next_available_offered = bool(control_flow_result.get("next_available_offered", False))

    call_log_entry = {
        "timestamp": timestamp,
        "source": "triage_endpoint",
        "call_id": None,
        "call_completed": True,
        "caller_name": caller_name,
        "caller_phone": caller_phone,
        "symptoms": build_symptom_summary(transcript),
        "full_transcript": transcript,
        "category": backend_category,
        "category_name": category_name,
        "symptoms_detected": backend_classification.get("matched_symptoms", []),
        "recommended_action": final_action,
        "rule_action": backend_classification.get("action", ""),
        "vapi_category": payload.category,
        "vapi_action": payload.action,
        "category_mismatch": payload.category is not None and payload.category != backend_category,
        "booking_attempt": booking_result,
        "escalation_required": escalation_required,
        "next_available_offered": next_available_offered,
    }
    log_call(call_log_entry)

    message = ""
    if backend_category == 1:
        message = "Emergency symptoms detected. Escalating immediately; no auto-booking attempted."
    elif backend_category == 2 and booking_result.get("reason") == "missing_patient_email_or_start_time":
        message = "Urgent symptoms detected. Please collect patient email and preferred same-day time, then submit booking details."
    elif backend_category == 2 and escalation_required:
        message = "Urgent symptoms detected. Same-day booking failed, escalated to triage nurse."
    elif backend_category == 2:
        message = "Urgent symptoms detected. Same-day booking completed."
    elif backend_category == 3 and booking_result.get("reason") == "missing_patient_email_or_start_time":
        message = "Moderate symptoms detected. Please collect patient email and preferred appointment time within 72 hours."
    elif backend_category == 3 and next_available_offered:
        message = "Moderate symptoms detected. 72-hour booking failed; offer next available slot."
    else:
        message = "Moderate symptoms detected. 72-hour booking completed."

    response_payload = {
        "category": backend_category,
        "category_name": category_name,
        "recommended_action": final_action,
        "symptoms_detected": backend_classification.get("matched_symptoms", []),
        "escalation_required": escalation_required,
        "next_available_offered": next_available_offered,
        "booking_attempt": booking_result,
        "message": message,
    }
    print(response_payload)
    logger.info(f"[{timestamp}] triage_response={response_payload}")

    return TriageResponse(**response_payload)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Clinical Triage"}


@app.get("/webhook-debug")
async def webhook_debug(limit: int = 10):
    """Return the most recent webhook payloads for troubleshooting integration issues."""
    events = read_webhook_debug_events(limit=limit)
    return {
        "count": len(events),
        "limit": max(1, min(limit, 100)),
        "events": events,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
