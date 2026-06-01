"""
robot.py — Social robot for aphasia speech therapy.
Production implementation: intelligent merge of robot1.py (Claude) + robot2.py (Gemini).

Design decisions — justified against both source files:
  RobotController class          [r2]  — all session state in one place, no globals
  chat.completions + json_mode   [r2]  — guaranteed JSON output; no regex fallbacks needed
  timeout=15 s on every API call [r2]  — prevents indefinite hangs on slow networks
  deferLater() async sleep       [r2]  — Twisted-native; avoids autobahn import dependency
  APIError / APIConnectionError  [r2]  — specific exception types catch more failure modes
  response_times in asr_callback [r2]  — correct measurement: starts when robot stops talking
  Timeout checked in poll loop   [r2]  — fires even when user gives no input at all
  json_object on profile call    [r2]  — same reliability for post-session analysis
  Rich per-scenario prompts      [r1]  — full aphasia-aware rules, not one-liners
  _build_memory_block()          [r1]  — clean cross-session continuity injection
  System prompt built once       [r1]  — not rebuilt every turn; cheaper and consistent
  CLOSING_PROMPT: LLM farewell   [r1]  — contextual goodbye when time runs out mid-sentence
  errors_encountered in log      [r1]  — production observability without crashing
  Three-tier greeting            [r1]  — adapts to new / returning / name-unknown user
  merge_profile_updates()        [r1]  — safe additive merge; no duplicate topics
  Log turns after LLM call       [r1]  — user msg excluded from history during API call
  _speak() with TTS error trap   [r1]  — robot never hard-crashes on TTS failure
  STT re-opened after all speech [fix] — was missing after clarification branches
  KeyboardInterrupt log save     [fix] — both source files skip saving on Ctrl+C
  pacing_metadata in session log [new] — avg response time + turn count for research use
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from autobahn.twisted.component import Component, run
from openai import APIConnectionError, APIError, OpenAI
from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks
from twisted.internet.task import deferLater


# =============================================================================
# CONSTANTS
# =============================================================================

WAMP_REALM = "rie.6a16d460f2a08d602afbbb9a"
LLM_MODEL = "gpt-4o"
LLM_TIMEOUT = 15.0         # seconds — abandon API call after this

MAX_DURATION = 60 * 5       # maximum session length in seconds

PROFILES_DIR = "profiles"   # one JSON file per participant
SESSIONS_DIR = "sessions"   # one JSON file per session

# Post-robot-speech pause before the microphone re-opens (seconds)
PAUSE_BASE = 1.0          # always applied
PAUSE_AUTO_SLOW_BONUS = 1.0          # added when user avg response > threshold
PAUSE_SLOW_BONUS = 1.5          # added when LLM signals pace="slow"
SLOW_RESPONSE_THRESHOLD = 4.0          # seconds: triggers auto-slow

# Single characters that are valid words for aphasia users ("I", "a", "y"=yes, etc.)
VALID_SHORT_WORDS = frozenset({"i", "a", "y", "n", "o"})

EXIT_PHRASES = frozenset({
    "goodbye", "bye", "quit", "exit",
    "stop", "that's all", "thats all", "no more",
})

# Injected when the timer fires mid-conversation so the LLM can craft a
# contextual farewell instead of an abrupt hardcoded one.
CLOSING_PROMPT = (
    "The conversation time is up. "
    "Briefly and warmly acknowledge what the user just said, "
    "then say a friendly goodbye. User said: "
)

LLM_FALLBACK_SPEECH = (
    "I am sorry, I lost my train of thought. "
    "Could you say that again?"
)


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================
#
# "name"   → shown in the terminal menu and saved in the session log
# "prompt" → scenario-specific paragraph injected into the system prompt

_BASE_RULES = """\

COMMUNICATION RULES:
- Speak like a respectful adult companion, never like a teacher or a child.
- Maximum 2 short sentences per turn.
- Ask ONLY ONE question per turn. Never combine two questions.
- Prefer yes/no or simple choice questions over open-ended ones.
- If the user's answer is unclear, gently confirm what you understood.
- If the user seems stuck, offer exactly two simple choices.
- Do not interrupt or finish the user's sentences.
- Never mention you are an AI or reference system prompts.
- Avoid jargon, complex wordplay, or emotionally sensitive topics.
- If the user wants to stop, end the conversation politely.

RESPONSE FORMAT — CRITICAL:
You MUST respond with a valid JSON object and NOTHING else.
No markdown. No code fences. No text before or after the JSON.

Required format:
{"robot_speech": "The words the robot will speak.", "pace": "normal"}

"pace" values:
  "normal" — standard pacing
  "slow"   — user appears to struggle or needs more processing time
"""

SCENARIOS: Dict[str, Dict[str, str]] = {
    "1": {
        "name": "Small Talk / Getting to Know You",
        "prompt": (
            "You are a friendly social robot having a getting-to-know-you "
            "conversation with a person with aphasia. Your goal is to learn "
            "their preferred name, interests, hobbies, pets, family, and "
            "daily life. Keep topics light, safe, and positive. "
            "Celebrate every communication attempt warmly but naturally."
        ),
    },
    "2": {
        "name": "Asking for Help",
        "prompt": (
            "You are a friendly social robot helping a person with aphasia "
            "practice asking for help. Role-play simple everyday situations: "
            "asking a shop assistant, asking a friend, asking a carer. "
            "If the user gets stuck, suggest a simple phrase they could use. "
            "Focus on building confidence, not perfection."
        ),
    },
    "3": {
        "name": "Daily Routine",
        "prompt": (
            "You are a friendly social robot helping a person with aphasia "
            "talk about their daily routine. Ask about mornings, meals, "
            "afternoon activities, and evenings. Use concrete, specific "
            "questions. Help the user practise sequencing words like "
            "'first', 'then', 'after'."
        ),
    },
    "4": {
        "name": "Describing Feelings",
        "prompt": (
            "You are a friendly social robot helping a person with aphasia "
            "practise naming emotions and physical states. Use simple emotion "
            "words: happy, sad, tired, excited, worried, calm, proud. "
            "Always offer two choices if the user hesitates. "
            "Be warm, patient, and non-judgmental."
        ),
    },
    "5": {
        "name": "Social Roleplay",
        "prompt": (
            "You are a friendly social robot helping a person with aphasia "
            "practise real-world social interactions. "
            "Role-play common situations: greeting a neighbour, ordering at "
            "a café, talking to a receptionist. "
            "Play the other role (neighbour, barista, etc.) and guide the "
            "user through each short exchange step by step."
        ),
    },
}


# =============================================================================
# FILESYSTEM HELPERS
# =============================================================================

def ensure_directories() -> None:
    """Create required storage directories if they don't exist yet."""
    os.makedirs(PROFILES_DIR, exist_ok=True)
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def generate_participant_id(full_name: str) -> str:
    """
    Convert a full name to a filesystem-safe participant ID.
    'John Smith' → 'john_smith'
    Spaces become underscores; all other non-alphanumeric characters are removed.
    """
    normalized = full_name.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", normalized)


# =============================================================================
# PROFILE MANAGEMENT
# =============================================================================

def _empty_profile(pid: str) -> Dict[str, Any]:
    """Return a blank profile with all required fields initialised."""
    return {
        "participant_id":           pid,
        "preferred_name":           "",
        "preferred_question_style": "",
        "enjoyed_topics":           [],
        "avoided_topics":           [],
        "session_notes":            [],   # list of {"date": iso, "note": str}
        "last_session_summary":     "",
        "suggested_next_scenario":  "",
    }


def load_profile(pid: str) -> Dict[str, Any]:
    """
    Load an existing profile from disk.
    Returns a blank profile if none exists or if the file is corrupted.
    """
    path = os.path.join(PROFILES_DIR, f"{pid}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[PROFILE] Loaded existing profile for '{pid}'.")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[PROFILE] Warning — could not read profile ({exc}). Starting fresh.")
    else:
        print(f"[PROFILE] No profile found for '{pid}'. Creating new profile.")
    return _empty_profile(pid)


def save_profile(pid: str, data: Dict[str, Any]) -> None:
    """Write a profile dict to disk. Logs a warning on failure rather than crashing."""
    path = os.path.join(PROFILES_DIR, f"{pid}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[PROFILE] Saved → {path}")
    except OSError as exc:
        print(f"[PROFILE] Error saving profile: {exc}")


def merge_profile_updates(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safely merge LLM-generated post-session data into the stored profile.

    Merge rules:
    - enjoyed_topics / avoided_topics : extend, no duplicates
    - session_notes                   : append a new {date, note} entry
    - String fields                   : replace only when update value is non-empty
    """
    # Extend topic lists without introducing duplicates
    for field in ("enjoyed_topics", "avoided_topics"):
        incoming = updates.get(field) or []
        if isinstance(incoming, list):
            current = existing.get(field, [])
            seen = set(current)
            existing[field] = current + [t for t in incoming if t not in seen]

    # Append a dated note; handle both "session_notes" and "session_note" keys
    raw_note = updates.get("session_notes") or updates.get(
        "session_note") or ""
    if raw_note:
        note_text = raw_note if isinstance(raw_note, str) else str(raw_note)
        existing.setdefault("session_notes", []).append({
            "date": datetime.now().isoformat(),
            "note": note_text,
        })

    # Replace scalar string fields only when the update is non-empty
    for field in ("preferred_name", "preferred_question_style",
                  "last_session_summary", "suggested_next_scenario"):
        value = updates.get(field) or ""
        if value:
            existing[field] = value

    return existing


# =============================================================================
# TERMINAL SETUP  (blocking — runs BEFORE the Twisted reactor starts)
# =============================================================================

def setup_session() -> Dict[str, Any]:
    """
    Collect participant name and scenario selection from the terminal operator.
    Returns a config dict passed to RobotController.__init__.
    """
    ensure_directories()

    print("\n" + "=" * 56)
    print("  APHASIA ROBOT SESSION SETUP")
    print("=" * 56)

    # ── Participant name ──────────────────────────────────────────────────────
    while True:
        raw = input("\nEnter participant first and last name: ").strip()
        parts = raw.split()
        if len(parts) >= 2:
            break
        print("  Please enter both a first name and a last name.")

    pid = generate_participant_id(raw)
    profile = load_profile(pid)

    # Set preferred_name only on the first session (don't overwrite a saved value)
    if not profile.get("preferred_name"):
        profile["preferred_name"] = parts[0].capitalize()

    # Surface any reminders from the last session
    if profile.get("last_session_summary"):
        print(f"\n  Last session   : {profile['last_session_summary']}")
    if profile.get("suggested_next_scenario"):
        print(f"  Suggested today: {profile['suggested_next_scenario']}")

    # ── Scenario selection ────────────────────────────────────────────────────
    print("\nSelect a scenario:")
    for key, sc in SCENARIOS.items():
        print(f"  [{key}] {sc['name']}")

    while True:
        raw_choice = input(
            "\nEnter scenario number (default 1): ").strip() or "1"
        if raw_choice in SCENARIOS:
            break
        print("  Please enter a number between 1 and 5.")

    scenario = SCENARIOS[raw_choice]
    print(f"\n  Participant : {pid}")
    print(f"  Scenario    : {scenario['name']}")
    print("=" * 56 + "\n")

    return {
        "participant_id": pid,
        "profile":        profile,
        "scenario":       scenario,
    }


# =============================================================================
# ROBOT CONTROLLER
# =============================================================================

class RobotController:
    """
    Owns all session state and robot behaviour for one therapy session.
    Instantiated once per session inside the WAMP on_join callback.

    Public interface:
      start()          — @inlineCallbacks coroutine; runs the full session
      save_session_log() — safe to call from outside the reactor (e.g. Ctrl+C)
    """

    def __init__(self, session, config: Dict[str, Any]) -> None:
        # WAMP session handle
        self.session = session

        # Session identity
        self.participant_id: str = config["participant_id"]
        self.profile:        Dict = config["profile"]
        self.scenario:       Dict = config["scenario"]

        # OpenAI client — one instance per session
        self.client = OpenAI()

        # System prompt built once at init; doesn't change mid-session
        self.system_prompt: str = self._build_system_prompt()

        # LLM conversation history (role/content pairs fed to chat.completions)
        self.conversation_history: List[Dict[str, str]] = []

        # Timestamped transcript written to disk at session end
        self.session_log: List[Dict[str, Any]] = []

        # STT / speech flags
        self.user_input:       str = ""
        self.new_input_ready:  bool = False
        self.robot_speaking:   bool = False

        # Pacing metrics
        self.session_start:     float = time.time()
        self.session_start_iso: str = datetime.now().isoformat()
        self.last_prompt_time:  float = 0.0
        self.response_times:    List[float] = []
        self.slow_pace_count:   int = 0   # how often LLM requested slow pace

        # Session outcome
        self.exit_reason:        str = "normal_end"
        self.errors_encountered: List[str] = []
        self._log_saved:         bool = False  # guard against double-save

    # =========================================================================
    # PROMPT BUILDER
    # =========================================================================

    def _build_memory_block(self) -> str:
        """
        Convert the stored profile into a plain-English paragraph injected into
        the system prompt. Gives the LLM cross-session continuity.
        Returns an empty string when the profile contains no useful data.
        """
        p = self.profile
        lines: List[str] = []

        if p.get("preferred_name"):
            lines.append(
                f"The user's preferred name is {p['preferred_name']}.")
        if p.get("preferred_question_style"):
            lines.append(
                f"They prefer {p['preferred_question_style']} questions.")
        if p.get("enjoyed_topics"):
            lines.append(
                f"Topics they enjoy: {', '.join(p['enjoyed_topics'])}.")
        if p.get("avoided_topics"):
            lines.append(f"Topics to avoid: {', '.join(p['avoided_topics'])}.")
        if p.get("last_session_summary"):
            lines.append(f"Last session summary: {p['last_session_summary']}")
        if p.get("suggested_next_scenario"):
            lines.append(
                f"Suggested focus for today: {p['suggested_next_scenario']}")

        if not lines:
            return ""
        return (
            "\nPERSONALIZATION MEMORY (from previous sessions):\n"
            + "\n".join(lines)
            + "\n"
        )

    def _build_system_prompt(self) -> str:
        """Combine scenario context + base rules + cross-session memory block."""
        return (
            "You are a friendly, respectful social robot conversing with a "
            "person with aphasia.\n\n"
            f"SCENARIO: {self.scenario['prompt']}"
            + _BASE_RULES
            + self._build_memory_block()
        )

    # =========================================================================
    # LLM  — single entry point for all conversation turns
    # =========================================================================

    def get_llm_response(self, user_text: str) -> Tuple[str, str]:
        """
        Send the current conversation + user_text to the LLM.
        Returns (robot_speech: str, pace: str).

        - Does NOT append to conversation_history — caller logs turns.
        - Retries once on any failure.
        - Returns a safe fallback string if both attempts fail.
        """
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.conversation_history           # previous turns only
            # current turn (not yet logged)
            + [{"role": "user", "content": user_text}]
        )

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=LLM_TIMEOUT,
                )
                content = response.choices[0].message.content or ""
                parsed = json.loads(content)

                speech = parsed.get("robot_speech", "").strip()
                pace = parsed.get("pace", "normal").lower()

                if not speech:
                    raise ValueError("'robot_speech' is empty in LLM response")
                if pace not in ("normal", "slow"):
                    pace = "normal"

                return speech, pace

            except (APIError, APIConnectionError) as exc:
                msg = f"API error (attempt {attempt + 1}): {exc}"
                print(f"[LLM] {msg}")
                self.errors_encountered.append(msg)

            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                msg = f"Response parse error (attempt {attempt + 1}): {exc}"
                print(f"[LLM] {msg}")
                self.errors_encountered.append(msg)

            except Exception as exc:
                msg = f"Unexpected error (attempt {attempt + 1}): {exc}"
                print(f"[LLM] {msg}")
                self.errors_encountered.append(msg)

            if attempt == 0:
                print("[LLM] Retrying in 1 second...")
                time.sleep(1)

        print("[LLM] Both attempts failed. Using fallback reply.")
        return LLM_FALLBACK_SPEECH, "slow"

    # =========================================================================
    # POST-SESSION PROFILE UPDATE
    # =========================================================================

    def _generate_profile_update(self) -> Dict[str, Any]:
        """
        Ask the LLM to analyse the finished conversation and return a profile
        update dict.  Returns {} on failure so the caller skips safely.
        """
        scenario_names = [sc["name"] for sc in SCENARIOS.values()]
        prompt = f"""Analyse the conversation and produce a personalisation profile update.

Participant: {self.participant_id}
Scenario: {self.scenario['name']}

Return ONLY valid JSON. No markdown, no code fences, no explanations.

Required schema:
{{
  "preferred_name": null,
  "preferred_question_style": null,
  "enjoyed_topics": [],
  "avoided_topics": [],
  "session_notes": "",
  "last_session_summary": "",
  "suggested_next_scenario": ""
}}

Rules:
- null or "" = unknown; [] = nothing found.
- Do NOT invent facts — base everything only on the conversation.
- session_notes  : 1–3 short observations about communication style or engagement.
- last_session_summary : 1–2 sentences summarising what happened.
- suggested_next_scenario : choose ONE of {scenario_names} or leave empty.
"""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(self.conversation_history,
                                              ensure_ascii=False),
                    },
                ],
                # guaranteed JSON [r2]
                response_format={"type": "json_object"},
                timeout=LLM_TIMEOUT,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)

        except Exception as exc:
            print(f"[PROFILE] Could not generate profile update: {exc}")
            return {}

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    @staticmethod
    def is_unclear_input(text: str) -> bool:
        """
        Return True when the transcript is too short or looks like STT noise.
        Single-letter valid words (i/a/y/n/o) are allowed — aphasia users
        often communicate meaningfully with single words.
        """
        cleaned = text.strip()
        if not cleaned:
            return True
        if len(cleaned) == 1 and cleaned.lower() not in VALID_SHORT_WORDS:
            return True
        return False

    # =========================================================================
    # PACING
    # =========================================================================

    def calculate_pause(self, pace: str) -> float:
        """
        Additive pause duration after the robot finishes speaking:
          PAUSE_BASE             — always applied
          + PAUSE_AUTO_SLOW_BONUS — when recent average response time > threshold
          + PAUSE_SLOW_BONUS      — when LLM explicitly signals pace="slow"
        """
        pause = PAUSE_BASE

        if self.response_times:
            recent = self.response_times[-5:]
            avg = sum(recent) / len(recent)
            if avg > SLOW_RESPONSE_THRESHOLD:
                print(f"[PACING] Auto-slow active (avg: {avg:.1f}s)")
                pause += PAUSE_AUTO_SLOW_BONUS

        if pace == "slow":
            pause += PAUSE_SLOW_BONUS
            self.slow_pace_count += 1

        return pause

    # =========================================================================
    # ASYNC SLEEP  (Twisted-native; avoids blocking the reactor)
    # =========================================================================

    @staticmethod
    def _sleep(seconds: float):
        """Non-blocking sleep that cooperates with the Twisted reactor."""
        return deferLater(reactor, seconds, lambda: None)

    # =========================================================================
    # LOGGING
    # =========================================================================

    def log_turn(self, role: str, text: str) -> None:
        """
        Append a turn to both:
          conversation_history — the list sent to the LLM on each turn
          session_log          — the timestamped list written to disk

        Only "user" and "assistant" roles go into conversation_history.
        The "system" role (used for internal notes) goes only into session_log.
        """
        if role in ("user", "assistant"):
            self.conversation_history.append({"role": role, "content": text})

        self.session_log.append({
            "speaker":   "robot" if role == "assistant" else role,
            "text":      text,
            "timestamp": datetime.now().isoformat(),
        })

        label = "ROBOT" if role == "assistant" else "USER"
        print(f"  [{label}] {text}")

    def save_session_log(self) -> None:
        """
        Write the full session log JSON to the sessions directory.
        Idempotent — safe to call more than once (e.g. on Ctrl+C after normal end).
        """
        if self._log_saved:
            return
        self._log_saved = True

        ts = datetime.now().isoformat().replace(":", "-")
        filename = os.path.join(
            SESSIONS_DIR, f"{self.participant_id}_{ts}.json")

        avg_response = (
            round(sum(self.response_times) / len(self.response_times), 2)
            if self.response_times else None
        )

        data = {
            "participant_id":   self.participant_id,
            "scenario":         self.scenario["name"],
            "timestamp":        self.session_start_iso,
            "duration_seconds": round(time.time() - self.session_start, 2),
            "exit_reason":      self.exit_reason,
            "pacing_metadata": {
                "avg_response_time_seconds": avg_response,
                "total_user_turns":          len(self.response_times),
                "slow_pace_count":           self.slow_pace_count,
            },
            "errors_encountered": self.errors_encountered,
            "conversation":       self.session_log,
            "questionnaire":      {},   # populated externally after the session
        }

        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[LOG] Session saved → {filename}")
        except OSError as exc:
            print(f"[LOG] Error saving session log: {exc}")

    # =========================================================================
    # STT CALLBACK
    # =========================================================================

    def asr_callback(self, frames) -> None:
        """
        Called by the WAMP STT subscription on every speech frame.
        - Ignores non-final frames (partial transcripts).
        - Ignores all frames while the robot is speaking (prevents echo).
        - Records how long the user took to respond (latency tracking).
        """
        if self.robot_speaking:
            return   # Ignore the robot's own TTS output

        if frames["data"]["body"]["final"]:
            transcript = str(frames["data"]["body"]["text"]).strip()
            if transcript:
                print(f"[STT] '{transcript}'")
                self.user_input = transcript
                self.new_input_ready = True

                # Measure response latency: time from robot-finished to user-spoke
                if self.last_prompt_time > 0:
                    elapsed = time.time() - self.last_prompt_time
                    self.response_times.append(elapsed)
                    print(f"[PACING] Response time: {elapsed:.1f}s")
                    self.last_prompt_time = 0.0

    # =========================================================================
    # ROBOT SPEECH WRAPPER
    # =========================================================================

    @inlineCallbacks
    def _speak(self, text: str):
        """
        Speak text via TTS. Sets/clears the robot_speaking flag around the call.
        Catches TTS errors so a failed TTS call never crashes the session.
        """
        self.robot_speaking = True
        try:
            yield self.session.call("rie.dialogue.say_animated", text=text)
        except Exception as exc:
            err = f"TTS error: {exc}"
            print(f"[TTS] {err}")
            self.errors_encountered.append(err)
        finally:
            self.robot_speaking = False

    # =========================================================================
    # MAIN SESSION LOOP
    # =========================================================================

    @inlineCallbacks
    def start(self):
        """
        Full robot lifecycle:
          1. Initialise hardware (language, stand up, subscribe to STT)
          2. Deliver greeting (adapts to new vs returning participant)
          3. Dialogue loop:
               - idle polling with timeout check every 0.3 s
               - exit phrase  → farewell + break
               - time limit   → contextual or hardcoded farewell + break
               - unclear STT  → gentle clarification + re-open mic
               - normal turn  → LLM reply → speak → pause → re-open mic
          4. Cleanup: close mic, sit down, update profile, save log
        """

        # ── Hardware initialisation ───────────────────────────────────────────
        yield self.session.call("rie.dialogue.config.language", lang="en")
        yield self.session.call("rom.optional.behavior.play", name="BlocklyStand")
        yield self.session.subscribe(self.asr_callback, "rie.dialogue.stt.stream")

        # ── Greeting (three-tier: returning / new with name / unknown) ────────
        name = self.profile.get("preferred_name", "")
        summary = self.profile.get("last_session_summary", "")

        if name and summary:
            greeting = (
                f"Hello {name}! Great to see you again. "
                "Are you ready to start today?"
            )
        elif name:
            greeting = (
                f"Hello {name}! I am your robot assistant. "
                "How are you feeling today?"
            )
        else:
            greeting = (
                "Hello! I am your robot assistant. "
                "What is your name?"
            )

        self.log_turn("assistant", greeting)
        yield self.session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
        yield self._speak(greeting)
        self.last_prompt_time = time.time()
        yield self.session.call("rie.dialogue.stt.stream")

        # ── Dialogue loop ─────────────────────────────────────────────────────
        while True:

            # Poll every 300 ms while waiting for STT input.
            # The timeout check runs here too so the session ends even if the
            # user gives no input at all [r2 improvement].
            if not self.new_input_ready:
                yield self._sleep(0.3)

                if time.time() - self.session_start > MAX_DURATION:
                    self.exit_reason = "time_limit"
                    timeout_farewell = (
                        "Our time is up for today. "
                        "Thank you so much for talking with me!"
                    )
                    self.log_turn("assistant", timeout_farewell)
                    yield self._speak(timeout_farewell)
                    break

                continue

            # ── Atomically capture and reset STT state ────────────────────────
            current_input = self.user_input
            self.user_input = ""
            self.new_input_ready = False

            # ── Exit phrase ───────────────────────────────────────────────────
            if any(phrase in current_input.lower() for phrase in EXIT_PHRASES):
                self.exit_reason = "user_exit_phrase"
                farewell = "It was lovely talking to you. Take care and goodbye!"
                self.log_turn("user",      current_input)
                self.log_turn("assistant", farewell)
                yield self._speak(farewell)
                yield self.session.call("rom.optional.behavior.play",
                                        name="BlocklyWaveRightArm")
                break

            # ── Time limit (re-checked after input arrives) ───────────────────
            # Uses the LLM for a contextual farewell since we have user text.
            if time.time() - self.session_start > MAX_DURATION:
                self.exit_reason = "time_limit"
                speech, _ = self.get_llm_response(
                    CLOSING_PROMPT + current_input)
                self.log_turn("user",      current_input)
                self.log_turn("assistant", speech)
                yield self._speak(speech)
                break

            # ── Unclear STT input ─────────────────────────────────────────────
            if self.is_unclear_input(current_input):
                print(f"[INPUT] Unclear transcript: '{current_input}'")
                clarification = (
                    "Sorry, I missed that. "
                    "Could you say it again slowly?"
                )
                self.log_turn("user",      current_input)
                self.log_turn("assistant", clarification)
                yield self._speak(clarification)
                yield self._sleep(PAUSE_SLOW_BONUS)
                # Re-open mic — required after EVERY robot speech turn [fix]
                yield self.session.call("rie.dialogue.stt.stream")
                self.last_prompt_time = time.time()
                continue

            # ── Normal LLM turn ───────────────────────────────────────────────
            speech, pace = self.get_llm_response(current_input)

            # Log AFTER the LLM call: current_input is excluded from
            # conversation_history during the API call, preventing duplication.
            self.log_turn("user",      current_input)
            self.log_turn("assistant", speech)

            yield self._speak(speech)

            pause = self.calculate_pause(pace)
            yield self._sleep(pause)

            # Re-open mic for the next user turn
            yield self.session.call("rie.dialogue.stt.stream")
            self.last_prompt_time = time.time()

        # ── Cleanup ───────────────────────────────────────────────────────────
        yield self.session.call("rie.dialogue.stt.close")
        yield self.session.call("rom.optional.behavior.play", name="BlocklyCrouch")

        # Print full transcript to terminal
        print("\n── Session Transcript ───────────────────────────────")
        for entry in self.session_log:
            label = "ROBOT" if entry["speaker"] == "robot" else "USER"
            print(f"  [{label}] {entry['text']}")
        print("─────────────────────────────────────────────────────\n")

        # Generate and persist profile update
        print("[PROFILE] Analysing session for profile update...")
        updates = self._generate_profile_update()
        if updates:
            updated = merge_profile_updates(self.profile, updates)
            save_profile(self.participant_id, updated)
        else:
            print("[PROFILE] No updates generated (empty session or API failure).")

        self.save_session_log()
        self.session.leave()


# =============================================================================
# WAMP COMPONENT BINDING  +  ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Terminal setup runs BEFORE the async reactor starts so input() works.
    config = setup_session()

    # Module-level reference to the active controller so the KeyboardInterrupt
    # handler can trigger a clean save even if the reactor is stopped abruptly.
    # Uses a single-element list as a mutable container (simpler than nonlocal).
    _active: List[Optional[RobotController]] = [None]

    wamp = Component(
        transports=[{
            "url":         "ws://wamp.robotsindeklas.nl",
            "serializers": ["msgpack"],
            "max_retries": 0,
        }],
        realm=WAMP_REALM,
    )

    @inlineCallbacks
    def on_join(session, details):
        controller = RobotController(session, config)
        _active[0] = controller
        yield controller.start()

    wamp.on_join(on_join)

    try:
        run([wamp])
    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt — saving session before exit...")
        if _active[0] is not None:
            _active[0].exit_reason = "keyboard_interrupt"
            _active[0].save_session_log()
        print("[STOP] Done.")
