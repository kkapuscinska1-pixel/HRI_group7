"""
robot.py — Production-Ready Social Robot for Aphasia Speech Therapy.

Architecture (merged from three iterations + document 4 reliability requirements):

  ProfileManager     — static-method class; all profile I/O, merge logic
  LLMManager         — all OpenAI calls (sync); called via deferToThread
  RobotController    — WAMP session owner; FSM state, STT buffer, watchdog
  terminal_setup()   — blocking setup before reactor starts
  __main__           — WAMP component binding + KeyboardInterrupt handler

Key reliability fixes vs previous versions:
  [FREEZE FIX]   deferToThread  — OpenAI calls run in a worker thread; the
                                   Twisted reactor is NEVER blocked by LLM I/O.
  [FREEZE FIX]   WAMP timeout   — every session.call() has a hard timeout so a
                                   hung TTS/behaviour call can't stall the loop.
  [FREEZE FIX]   Watchdog       — LoopingCall detects states that haven't
                                   changed for too long and recovers.
  [STT FIX]      Buffer         — STT finals are accumulated for STT_BUFFER_DELAY
                                   seconds before being processed, merging split
                                   transcripts into one coherent utterance.
  [STT FIX]      Echo guard     — POST_SPEECH_MIC_DELAY prevents TTS audio from
                                   leaking back into the microphone.
  [STT FIX]      Stream restart — watchdog restarts a silently-dead STT stream.
  [TIMING FIX]   no time.sleep  — all waits use deferLater; reactor stays live.
  [TIMING FIX]   Exp. backoff   — LLM retries back off 1s / 3s in the thread.
  [CRASH FIX]    _log_saved     — idempotent save; safe on Ctrl+C or double call.
  [CRASH FIX]    Profile guard  — corrupt profile JSON falls back to empty; no crash.
  [HISTORY FIX]  Log after LLM  — user message is NOT in conversation_history
                                   during the API call; added only on success.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import enum
import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from autobahn.twisted.component import Component, run
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks
from twisted.internet.task import deferLater, LoopingCall


# =============================================================================
# CONSTANTS  (all tuneable in one place)
# =============================================================================

WAMP_REALM = "rie.69f352c326d8af1680827d4a"
LLM_MODEL = "gpt-4o"
LLM_TIMEOUT_SECS = 20.0    # OpenAI request hard timeout (in thread)
LLM_RETRY_DELAYS = (1.0, 3.0)  # wait before attempt 2, 3 (exp. backoff)

MAX_SESSION_DURATION = 60 * 5  # seconds before time-limit farewell

PROFILES_DIR = "profiles"
SESSIONS_DIR = "sessions"

# --- Pacing ---
PAUSE_BASE = 1.0    # always applied after robot speech
PAUSE_AUTO_SLOW_BONUS = 1.0    # added when avg response time > threshold
PAUSE_SLOW_BONUS = 1.5    # added when LLM signals pace="slow"
SLOW_RESPONSE_THRESHOLD = 4.0    # seconds: triggers auto-slow pacing
POST_SPEECH_MIC_DELAY = 0.8    # [NEW] wait after TTS before opening mic
#       prevents TTS echo leaking into STT

# --- STT reliability ---
STT_BUFFER_DELAY = 0.8    # [NEW] seconds to wait for more STT finals
#       before treating buffer as one utterance
STT_SILENCE_TIMEOUT = 25.0   # [NEW] restart STT stream if no activity
#       this long while it should be open

# --- WAMP calls ---
WAMP_CALL_TIMEOUT = 12.0   # [NEW] seconds before a WAMP call is cancelled

# --- Watchdog ---
WATCHDOG_INTERVAL = 5.0    # [NEW] how often the watchdog checks (seconds)
WATCHDOG_FREEZE_TIMEOUT = 45.0   # [NEW] how long a state may be unchanged
#       before watchdog triggers recovery

# --- Input ---
VALID_SHORT_WORDS = frozenset({"i", "a", "y", "n", "o"})
EXIT_PHRASES = frozenset({
    "goodbye", "bye", "quit", "exit",
    "stop", "that's all", "thats all", "no more",
})

# --- Fallbacks ---
LLM_FALLBACK_SPEECH = (
    "I am sorry, I lost my train of thought. "
    "Could you say that again?"
)
CLOSING_PROMPT = (
    "The conversation time is up. "
    "Briefly and warmly acknowledge what the user just said, "
    "then say a friendly goodbye. User said: "
)


# =============================================================================
# ROBOT STATE ENUM  [NEW]
# =============================================================================

class RobotState(enum.Enum):
    INIT = "init"
    GREETING = "greeting"
    LISTENING = "listening"   # STT open, waiting for user
    PROCESSING = "processing"  # LLM call in flight
    SPEAKING = "speaking"    # TTS active
    CLARIFYING = "clarifying"  # asking user to repeat
    CLOSING = "closing"     # farewell in progress
    DONE = "done"        # session over


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

_BASE_RULES = """\

COMMUNICATION RULES:
- Speak like a respectful adult companion, not a teacher or a child.
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
# PROFILE MANAGER  [from robot3 — static-method class, cleaner encapsulation]
# =============================================================================

class ProfileManager:
    """Handles all profile file I/O and data transformations. No instance state."""

    @staticmethod
    def ensure_dirs() -> None:
        os.makedirs(PROFILES_DIR, exist_ok=True)
        os.makedirs(SESSIONS_DIR, exist_ok=True)

    @staticmethod
    def make_participant_id(first: str, last: str) -> str:
        """'John Smith' → 'john_smith'"""
        raw = f"{first.strip().lower()}_{last.strip().lower()}"
        return re.sub(r"[^a-z0-9_]", "", raw)

    @staticmethod
    def empty_profile(pid: str) -> Dict[str, Any]:
        return {
            "participant_id":           pid,
            "preferred_name":           "",
            "preferred_question_style": "",
            "enjoyed_topics":           [],
            "avoided_topics":           [],
            # list of {"date": iso, "note": str}
            "session_notes":            [],
            "last_session_summary":     "",
            "suggested_next_scenario":  "",
        }

    @staticmethod
    def load(pid: str) -> Dict[str, Any]:
        """
        Load from disk. Returns blank profile on missing file or JSON corruption.
        Also backfills any missing list fields (handles older profile formats).
        """
        ProfileManager.ensure_dirs()
        path = os.path.join(PROFILES_DIR, f"{pid}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # [GUARD] Backfill list fields missing from older profiles
                for field in ("enjoyed_topics", "avoided_topics", "session_notes"):
                    if field not in data or not isinstance(data[field], list):
                        data[field] = []
                print(f"[PROFILE] Loaded profile for '{pid}'.")
                return data
            except (json.JSONDecodeError, OSError, KeyError) as exc:
                # [GUARD] Corrupt file: warn, fall through to blank profile
                print(
                    f"[PROFILE] Warning — corrupt profile ({exc}). Starting fresh.")
        else:
            print(
                f"[PROFILE] No profile found for '{pid}'. Creating new profile.")
        return ProfileManager.empty_profile(pid)

    @staticmethod
    def save(pid: str, data: Dict[str, Any]) -> None:
        """Write profile. Logs warning on failure rather than crashing."""
        ProfileManager.ensure_dirs()
        path = os.path.join(PROFILES_DIR, f"{pid}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[PROFILE] Saved → {path}")
        except OSError as exc:
            print(f"[PROFILE] Error saving: {exc}")

    @staticmethod
    def merge_updates(existing: Dict[str, Any],
                      updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely merge LLM-generated post-session data into the stored profile.
          - Topic lists: extend without duplicates.
          - session_notes: append a new {date, note} entry.
          - String fields: replace only when update value is non-empty.
        """
        for field in ("enjoyed_topics", "avoided_topics"):
            incoming = updates.get(field) or []
            if isinstance(incoming, list):
                current = existing.get(field, [])
                seen = set(current)
                existing[field] = current + \
                    [t for t in incoming if t not in seen]

        raw_note = updates.get("session_notes") or ""
        if raw_note and isinstance(raw_note, str):
            existing.setdefault("session_notes", []).append({
                "date": datetime.now().isoformat(),
                "note": raw_note,
            })

        for field in ("preferred_name", "preferred_question_style",
                      "last_session_summary", "suggested_next_scenario"):
            value = updates.get(field) or ""
            if value:
                existing[field] = value

        return existing

    @staticmethod
    def build_memory_block(profile: Dict[str, Any]) -> str:
        """
        Convert stored profile data into a plain-English paragraph for the
        system prompt — gives the LLM cross-session continuity.
        """
        p = profile
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
            + "\n".join(lines) + "\n"
        )


# =============================================================================
# LLM MANAGER  [from robot3, heavily enhanced]
# =============================================================================

class LLMManager:
    """
    Wraps all OpenAI API calls. All public methods are SYNCHRONOUS — they are
    always called via twisted.internet.threads.deferToThread so the Twisted
    reactor is never blocked by network I/O.
    """

    def __init__(self) -> None:
        self.client = OpenAI()

    # ------------------------------------------------------------------
    # Conversation turn
    # ------------------------------------------------------------------

    def _sync_get_response(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[str, str, Optional[str]]:
        """
        Synchronous LLM call with exponential-backoff retry.
        Returns (robot_speech, pace, error_or_None).
        Runs in a worker thread — time.sleep() is safe here.
        """
        attempts = 1 + len(LLM_RETRY_DELAYS)  # total attempts
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=LLM_TIMEOUT_SECS,
                )
                content = response.choices[0].message.content or ""
                parsed = json.loads(content)

                speech = parsed.get("robot_speech", "").strip()
                pace = parsed.get("pace", "normal").lower()

                if not speech:
                    raise ValueError("'robot_speech' empty in LLM JSON")
                if pace not in ("normal", "slow"):
                    pace = "normal"

                return speech, pace, None  # success

            except (APIError, APIConnectionError, APITimeoutError) as exc:
                err = f"API error attempt {attempt + 1}: {exc}"
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                err = f"Parse error attempt {attempt + 1}: {exc}"
            except Exception as exc:
                err = f"Unexpected error attempt {attempt + 1}: {exc}"

            print(f"[LLM] {err}")

            # Exponential backoff between attempts (safe in thread)
            if attempt < len(LLM_RETRY_DELAYS):
                delay = LLM_RETRY_DELAYS[attempt]
                print(f"[LLM] Retrying in {delay}s...")
                time.sleep(delay)

        return LLM_FALLBACK_SPEECH, "slow", err

    # ------------------------------------------------------------------
    # Post-session profile extraction
    # ------------------------------------------------------------------

    def _sync_extract_profile(
        self,
        conversation: List[Dict[str, str]],
        pid: str,
        scenario_name: str,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Synchronous post-session analysis call. Runs in a worker thread.
        Returns (updates_dict, error_or_None).
        """
        scenario_names = [sc["name"] for sc in SCENARIOS.values()]
        prompt = f"""Analyse this conversation and produce a personalisation profile update.

Participant: {pid}
Scenario: {scenario_name}

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
- session_notes: 1-3 short observations about communication style or engagement.
- last_session_summary: 1-2 sentences summarising what happened.
- suggested_next_scenario: choose ONE of {scenario_names} or leave empty.
"""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user",
                     "content": json.dumps(conversation, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                timeout=LLM_TIMEOUT_SECS,
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content), None
        except Exception as exc:
            err = f"Profile extraction failed: {exc}"
            print(f"[PROFILE] {err}")
            return {}, err


# =============================================================================
# ROBOT CONTROLLER
# =============================================================================

class RobotController:
    """
    Owns all session state and robot behaviour.
    Instantiated once per session inside the WAMP on_join callback.

    Public interface:
      start()            — @inlineCallbacks; runs full session lifecycle
      save_session_log() — idempotent; safe to call from KeyboardInterrupt handler
    """

    def __init__(self, session, config: Dict[str, Any]) -> None:
        self.session = session
        self.participant_id = config["participant_id"]
        self.profile = config["profile"]
        self.scenario = config["scenario"]
        self.llm = LLMManager()

        # Build system prompt once (doesn't change mid-session)
        self.system_prompt: str = (
            "You are a friendly, respectful social robot conversing with a "
            "person with aphasia.\n\n"
            f"SCENARIO: {self.scenario['prompt']}"
            + _BASE_RULES
            + ProfileManager.build_memory_block(self.profile)
        )

        # Conversation history fed to the LLM (role/content pairs)
        self.conversation_history: List[Dict[str, str]] = []
        # Timestamped transcript written to disk at session end
        self.session_log: List[Dict[str, Any]] = []

        # --- STT state ---
        self.robot_speaking:   bool = False
        self.user_input:       str = ""
        self.new_input_ready:  bool = False
        self._stt_buffer:      List[str] = []   # [NEW] accumulates finals
        self._stt_timer = None  # [NEW] IDelayedCall
        self._stt_open:        bool = False  # [NEW] track stream state
        self._stt_open_time:   float = 0.0   # [NEW] when stream was opened

        # --- Timing ---
        self.session_start:      float = time.time()
        self.session_start_iso:  str = datetime.now().isoformat()
        self._user_turn_start:   float = 0.0
        self.response_times:     List[float] = []
        self.slow_pace_count:    int = 0

        # --- FSM state ---  [NEW]
        self.state: RobotState = RobotState.INIT
        self._state_since:    float = time.time()

        # --- Watchdog ---  [NEW]
        self._watchdog: Optional[LoopingCall] = None

        # --- Session outcome ---
        self.exit_reason:         str = "normal_end"
        self.errors_encountered:  List[str] = []
        self._log_saved:          bool = False  # idempotent save guard

    # =========================================================================
    # STRUCTURED LOGGING  [NEW]
    # =========================================================================

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}][{self.state.value.upper()}] {msg}")

    def _log_error(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}][ERROR] {msg}")
        self.errors_encountered.append(f"{ts}: {msg}")

    # =========================================================================
    # FSM STATE MANAGEMENT  [NEW]
    # =========================================================================

    def _set_state(self, new_state: RobotState) -> None:
        if new_state != self.state:
            self._log(f"→ {new_state.value}")
            self.state = new_state
            self._state_since = time.time()

    # =========================================================================
    # WATCHDOG  [NEW]
    # =========================================================================

    def _start_watchdog(self) -> None:
        """Start a LoopingCall that periodically checks for frozen states."""
        self._watchdog = LoopingCall(self._watchdog_check)
        self._watchdog.start(WATCHDOG_INTERVAL, now=False)
        self._log("Watchdog started.")

    def _stop_watchdog(self) -> None:
        if self._watchdog and self._watchdog.running:
            self._watchdog.stop()

    def _watchdog_check(self) -> None:
        """
        Runs every WATCHDOG_INTERVAL seconds.
        Detects states that haven't changed for too long and recovers.
        """
        if self.state == RobotState.DONE:
            self._stop_watchdog()
            return

        elapsed = time.time() - self._state_since

        # LISTENING too long: STT stream may have silently died
        if self.state == RobotState.LISTENING:
            stt_idle = time.time() - self._stt_open_time
            if stt_idle > STT_SILENCE_TIMEOUT:
                self._log_error(
                    f"[WATCHDOG] STT silent for {stt_idle:.0f}s — restarting stream"
                )
                reactor.callLater(0, self._restart_stt)
                return

        # SPEAKING too long: TTS call may have hung
        if self.state == RobotState.SPEAKING and elapsed > WATCHDOG_FREEZE_TIMEOUT:
            self._log_error(
                f"[WATCHDOG] Stuck in SPEAKING for {elapsed:.0f}s — resetting"
            )
            self.robot_speaking = False
            self._set_state(RobotState.LISTENING)
            reactor.callLater(0, self._restart_stt)
            return

        # PROCESSING too long: LLM thread may be very slow — warn only
        if self.state == RobotState.PROCESSING and elapsed > WATCHDOG_FREEZE_TIMEOUT:
            self._log_error(
                f"[WATCHDOG] LLM processing for {elapsed:.0f}s — still waiting"
            )
            # We can't cancel the thread safely; just reset the timer to
            # avoid repeated warnings for the same slow call.
            self._state_since = time.time()

    # =========================================================================
    # WAMP CALL WRAPPER  [NEW — timeout on every robot call]
    # =========================================================================

    @inlineCallbacks
    def _wamp(self, method: str, *args, **kwargs):
        """
        Safe wrapper around session.call() with a hard timeout.
        Returns None on timeout or error rather than crashing.
        """
        d = self.session.call(method, *args, **kwargs)

        # Schedule cancellation after timeout
        def _cancel():
            if not d.called:
                self._log_error(f"[WAMP] Timeout: {method}")
                d.cancel()

        timeout_handle = reactor.callLater(WAMP_CALL_TIMEOUT, _cancel)

        try:
            result = yield d
            if timeout_handle.active():
                timeout_handle.cancel()
            return result
        except Exception as exc:
            if timeout_handle.active():
                timeout_handle.cancel()
            self._log_error(
                f"[WAMP] Error {method}: {type(exc).__name__}: {exc}")
            return None

    # =========================================================================
    # STT STREAM MANAGEMENT  [NEW — explicit open/close with buffer logic]
    # =========================================================================

    @inlineCallbacks
    def _open_stt(self) -> None:
        """
        Open the STT stream. Includes POST_SPEECH_MIC_DELAY to let TTS audio
        fully decay before the microphone becomes active.
        """
        # [NEW] brief pause so TTS audio doesn't leak into the microphone
        yield deferLater(reactor, POST_SPEECH_MIC_DELAY, lambda: None)
        result = yield self._wamp("rie.dialogue.stt.stream")
        if result is not None or True:   # proceed even if wamp returns None
            self._stt_open = True
            self._stt_open_time = time.time()
            self._log("STT stream opened.")

    @inlineCallbacks
    def _close_stt(self) -> None:
        self._stt_open = False
        self._cancel_stt_timer()
        yield self._wamp("rie.dialogue.stt.close")
        self._log("STT stream closed.")

    @inlineCallbacks
    def _restart_stt(self) -> None:
        """Close and reopen the STT stream (recovery from silent stream)."""
        self._log("Restarting STT stream...")
        self._stt_open = False
        self._cancel_stt_timer()
        yield self._wamp("rie.dialogue.stt.close")
        yield deferLater(reactor, 0.5, lambda: None)
        yield self._open_stt()

    def _cancel_stt_timer(self) -> None:
        """Cancel any pending STT buffer flush."""
        if self._stt_timer is not None:
            try:
                if self._stt_timer.active():
                    self._stt_timer.cancel()
            except Exception:
                pass
            self._stt_timer = None

    # =========================================================================
    # STT CALLBACK  [NEW — buffered; accumulates split transcripts]
    # =========================================================================

    def asr_callback(self, frames) -> None:
        """
        Called on every STT frame.

        Problem solved: STT often delivers multiple 'final' results for a
        single utterance (e.g., "I like" then "music" as separate finals).
        Solution: each final is buffered. A reactor.callLater timer resets on
        each new final. Only when STT_BUFFER_DELAY passes with no new finals
        does _finalize_stt() process the accumulated text as one utterance.
        """
        if self.robot_speaking:
            return  # Ignore the robot's own TTS output

        try:
            body = frames["data"]["body"]
        except (KeyError, TypeError):
            return

        if not body.get("final"):
            return

        text = str(body.get("text", "")).strip()
        if not text:
            return

        self._log(f"[STT] Final fragment: '{text}'")
        self._stt_buffer.append(text)
        self._stt_open_time = time.time()  # reset silence watchdog

        # [NEW] Cancel pending flush and restart the buffer delay timer
        self._cancel_stt_timer()
        self._stt_timer = reactor.callLater(
            STT_BUFFER_DELAY, self._finalize_stt)

    def _finalize_stt(self) -> None:
        """
        Called after STT_BUFFER_DELAY with no new finals.
        Joins buffered fragments and triggers the conversation loop.
        """
        self._stt_timer = None
        full_text = " ".join(self._stt_buffer).strip()
        self._stt_buffer.clear()

        if not full_text:
            return

        # Only accept input when genuinely listening
        if self.state not in (RobotState.LISTENING, RobotState.CLARIFYING):
            self._log(
                f"[STT] Discarded (state={self.state.value}): '{full_text}'")
            return

        self._log(f"[STT] Finalized utterance: '{full_text}'")

        # Record how long the user took to respond
        if self._user_turn_start > 0:
            elapsed = time.time() - self._user_turn_start
            self.response_times.append(elapsed)
            self._log(f"[PACING] Response time: {elapsed:.1f}s")
            self._user_turn_start = 0.0

        self.user_input = full_text
        self.new_input_ready = True

    # =========================================================================
    # ROBOT SPEECH WRAPPER
    # =========================================================================

    @inlineCallbacks
    def _speak(self, text: str) -> None:
        """
        Speak via TTS. Sets state = SPEAKING; resets on completion/error.
        Uses _wamp() so a hung TTS call times out rather than freezing.
        """
        self._set_state(RobotState.SPEAKING)
        self.robot_speaking = True
        self._log(f"[TTS] '{text[:80]}{'...' if len(text) > 80 else ''}'")
        try:
            yield self._wamp("rie.dialogue.say_animated", text=text)
        finally:
            self.robot_speaking = False

    # =========================================================================
    # PACING
    # =========================================================================

    def _calculate_pause(self, pace: str) -> float:
        """
        Additive pause after robot speech:
          base + auto_slow_bonus (avg response > threshold) + slow_bonus (LLM signal)
        """
        pause = PAUSE_BASE

        if self.response_times:
            avg = sum(self.response_times[-5:]) / len(self.response_times[-5:])
            if avg > SLOW_RESPONSE_THRESHOLD:
                self._log(f"[PACING] Auto-slow active (avg {avg:.1f}s)")
                pause += PAUSE_AUTO_SLOW_BONUS

        if pace == "slow":
            pause += PAUSE_SLOW_BONUS
            self.slow_pace_count += 1

        return pause

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    @staticmethod
    def _is_unclear(text: str) -> bool:
        """
        True when transcript is noise or too short to process.
        Single valid words (i/a/y/n/o) are allowed — common in aphasia.
        """
        cleaned = text.strip()
        if not cleaned:
            return True
        if len(cleaned) == 1 and cleaned.lower() not in VALID_SHORT_WORDS:
            return True
        return False

    # =========================================================================
    # TURN LOGGING
    # =========================================================================

    def _log_turn(self, role: str, text: str) -> None:
        """
        Append to both conversation_history (LLM input) and session_log (disk).
        Only "user" / "assistant" go into conversation_history.
        """
        if role in ("user", "assistant"):
            self.conversation_history.append({"role": role, "content": text})

        self.session_log.append({
            "speaker":   "robot" if role == "assistant" else role,
            "text":      text,
            "timestamp": datetime.now().isoformat(),
        })

    # =========================================================================
    # SESSION LOG  (idempotent)
    # =========================================================================

    def save_session_log(self) -> None:
        """Write session JSON to disk. Idempotent — safe on Ctrl+C."""
        if self._log_saved:
            return
        self._log_saved = True

        ts = datetime.now().isoformat().replace(":", "-")
        filename = os.path.join(
            SESSIONS_DIR, f"{self.participant_id}_{ts}.json")

        avg_resp = (
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
                "avg_response_time_seconds": avg_resp,
                "total_user_turns":          len(self.response_times),
                "slow_pace_count":           self.slow_pace_count,
            },
            "errors_encountered": self.errors_encountered,
            "conversation":       self.session_log,
            "questionnaire":      {},  # filled externally after session
        }
        try:
            ProfileManager.ensure_dirs()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[LOG] Session saved → {filename}")
        except OSError as exc:
            print(f"[LOG] Error saving session log: {exc}")

    # =========================================================================
    # MAIN SESSION LOOP
    # =========================================================================

    @inlineCallbacks
    def start(self) -> None:
        """
        Full robot lifecycle:
          1.  Initialise hardware
          2.  Start watchdog
          3.  Deliver greeting
          4.  Dialogue loop:
                - Poll every 0.3 s (reactor-safe deferLater)
                - Idle timeout check
                - Exit phrase → farewell
                - Time limit → contextual farewell
                - Unclear input → clarification
                - Normal turn → LLM (in thread) → speak → pause → reopen mic
          5.  Cleanup: close mic, sit, update profile, save log, leave
        """

        # ── Hardware init ─────────────────────────────────────────────────────
        self._set_state(RobotState.INIT)
        yield self._wamp("rie.dialogue.config.language", lang="en")
        yield self._wamp("rom.optional.behavior.play", name="BlocklyStand")
        yield self.session.subscribe(self.asr_callback, "rie.dialogue.stt.stream")

        self._start_watchdog()

        # ── Greeting ─────────────────────────────────────────────────────────
        self._set_state(RobotState.GREETING)
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

        self._log_turn("assistant", greeting)
        yield self._wamp("rom.optional.behavior.play", name="BlocklyWaveRightArm")
        yield self._speak(greeting)
        yield self._open_stt()
        self._set_state(RobotState.LISTENING)
        self._user_turn_start = time.time()

        # ── Dialogue loop ─────────────────────────────────────────────────────
        while True:

            # Poll while waiting for STT — reactor stays free via deferLater
            if not self.new_input_ready:
                yield deferLater(reactor, 0.3, lambda: None)

                # Idle session timeout (no input at all)
                if time.time() - self.session_start > MAX_SESSION_DURATION:
                    self._set_state(RobotState.CLOSING)
                    self.exit_reason = "time_limit"
                    timeout_farewell = (
                        "Our time is up for today. "
                        "Thank you so much for talking with me!"
                    )
                    self._log_turn("assistant", timeout_farewell)
                    yield self._speak(timeout_farewell)
                    break

                continue

            # ── Atomically capture STT result ─────────────────────────────────
            current_input = self.user_input
            self.user_input = ""
            self.new_input_ready = False

            # ── Exit phrase ───────────────────────────────────────────────────
            if any(p in current_input.lower() for p in EXIT_PHRASES):
                self._set_state(RobotState.CLOSING)
                self.exit_reason = "user_exit_phrase"
                farewell = "It was lovely talking to you. Take care and goodbye!"
                self._log_turn("user",      current_input)
                self._log_turn("assistant", farewell)
                yield self._speak(farewell)
                yield self._wamp("rom.optional.behavior.play", name="BlocklyWaveRightArm")
                break

            # ── Time limit (re-checked when input arrives) ────────────────────
            if time.time() - self.session_start > MAX_SESSION_DURATION:
                self._set_state(RobotState.CLOSING)
                self.exit_reason = "time_limit"
                closing_text = CLOSING_PROMPT + current_input
                messages = (
                    [{"role": "system", "content": self.system_prompt}]
                    + self.conversation_history
                    + [{"role": "user", "content": closing_text}]
                )
                # [NEW] LLM in thread — reactor stays live during call
                speech, pace, err = yield threads.deferToThread(
                    self.llm._sync_get_response, messages
                )
                if err:
                    self._log_error(err)
                self._log_turn("user",      current_input)
                self._log_turn("assistant", speech)
                yield self._speak(speech)
                break

            # ── Unclear STT input ─────────────────────────────────────────────
            if self._is_unclear(current_input):
                self._log(f"[INPUT] Unclear: '{current_input}'")
                self._set_state(RobotState.CLARIFYING)
                clarification = (
                    "Sorry, I missed that. "
                    "Could you say it again slowly?"
                )
                self._log_turn("user",      current_input)
                self._log_turn("assistant", clarification)
                yield self._speak(clarification)
                yield deferLater(reactor, PAUSE_SLOW_BONUS, lambda: None)
                # [FIX] STT must be reopened after EVERY robot speech turn
                yield self._open_stt()
                self._set_state(RobotState.LISTENING)
                self._user_turn_start = time.time()
                continue

            # ── Normal LLM turn ───────────────────────────────────────────────
            self._set_state(RobotState.PROCESSING)

            # Build messages with current input NOT yet in history
            # [FIX] user message excluded during API call — no duplication
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + self.conversation_history
                + [{"role": "user", "content": current_input}]
            )

            t0 = time.time()
            # [NEW] deferToThread: OpenAI runs in worker thread
            speech, pace, err = yield threads.deferToThread(
                self.llm._sync_get_response, messages
            )
            self._log(f"[LLM] Responded in {time.time()-t0:.1f}s")

            if err:
                self._log_error(err)

            # Log AFTER successful (or fallback) LLM response
            self._log_turn("user",      current_input)
            self._log_turn("assistant", speech)

            yield self._speak(speech)

            pause = self._calculate_pause(pace)
            yield deferLater(reactor, pause, lambda: None)

            yield self._open_stt()
            self._set_state(RobotState.LISTENING)
            self._user_turn_start = time.time()

        # ── Cleanup ───────────────────────────────────────────────────────────
        self._set_state(RobotState.DONE)
        self._stop_watchdog()
        self._cancel_stt_timer()

        yield self._close_stt()
        yield self._wamp("rom.optional.behavior.play", name="BlocklyCrouch")

        # Print transcript to terminal
        print("\n── Session Transcript ─────────────────────────────────────")
        for entry in self.session_log:
            label = "ROBOT" if entry["speaker"] == "robot" else "USER"
            print(f"  [{label}] {entry['text']}")
        print("────────────────────────────────────────────────────────────\n")

        # Post-session: generate profile update in thread
        print("[PROFILE] Analysing session...")
        updates, err = yield threads.deferToThread(
            self.llm._sync_extract_profile,
            self.conversation_history,
            self.participant_id,
            self.scenario["name"],
        )
        if err:
            self._log_error(err)
        if updates:
            updated = ProfileManager.merge_updates(self.profile, updates)
            ProfileManager.save(self.participant_id, updated)
        else:
            print("[PROFILE] No updates generated.")

        self.save_session_log()
        self.session.leave()


# =============================================================================
# TERMINAL SETUP  (blocking — runs BEFORE the Twisted reactor starts)
# =============================================================================

def terminal_setup() -> Dict[str, Any]:
    """
    Collect participant name and scenario selection from the terminal.
    Returns a config dict passed to RobotController.__init__.
    """
    ProfileManager.ensure_dirs()

    print("\n" + "=" * 56)
    print("  APHASIA ROBOT SESSION SETUP")
    print("=" * 56)

    # ── Name ─────────────────────────────────────────────────────────────────
    while True:
        raw = input("\nEnter participant first and last name: ").strip()
        parts = raw.split()
        if len(parts) >= 2:
            break
        print("  Please enter both a first name and a last name.")

    pid = ProfileManager.make_participant_id(parts[0], parts[-1])
    profile = ProfileManager.load(pid)

    # Set preferred_name only when the profile doesn't already have one
    if not profile.get("preferred_name"):
        profile["preferred_name"] = parts[0].capitalize()

    # Surface reminders from the previous session
    if profile.get("last_session_summary"):
        print(f"\n  Last session   : {profile['last_session_summary']}")
    if profile.get("suggested_next_scenario"):
        print(f"  Suggested today: {profile['suggested_next_scenario']}")

    # ── Scenario ─────────────────────────────────────────────────────────────
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

    return {"participant_id": pid, "profile": profile, "scenario": scenario}


# =============================================================================
# WAMP COMPONENT BINDING + ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # Terminal setup runs BEFORE the async reactor starts so input() works.
    config = terminal_setup()

    # Module-level reference to the active controller so the Ctrl+C handler
    # can trigger a clean session save. Single-element list = mutable container.
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
            _active[0]._stop_watchdog()
            _active[0].save_session_log()
        print("[STOP] Done.")
