"""
robot.py — Production-Ready Social Robot for Aphasia Speech Therapy.

Architecture (merged from three iterations + document 4 reliability requirements):

  ProfileManager     — static-method class; all profile I/O, merge logic
  LLMManager         — all OpenAI calls (sync); called via deferToThread
  RobotController    — WAMP session owner; FSM state, STT buffer, watchdog
  terminal_setup()   — blocking setup before reactor starts
  __main__           — WAMP component binding + KeyboardInterrupt handler
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

WAMP_REALM = "rie.6a2028558a2cba4f82b851d2"
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
POST_SPEECH_MIC_DELAY = 0.8    # wait after TTS before opening mic

# --- STT reliability ---
STT_BUFFER_DELAY = 0.8    # seconds to wait for more STT finals
STT_SILENCE_TIMEOUT = 25.0   # restart STT stream if no activity

# --- WAMP calls ---
WAMP_CALL_TIMEOUT = 12.0   # seconds before a WAMP call is cancelled

# --- Watchdog ---
WATCHDOG_INTERVAL = 5.0    # how often the watchdog checks (seconds)
WATCHDOG_FREEZE_TIMEOUT = 45.0   # how long a state may be unchanged

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
# ROBOT STATE ENUM
# =============================================================================

class RobotState(enum.Enum):
    INIT = "init"
    GREETING = "greeting"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    CLARIFYING = "clarifying"
    CLOSING = "closing"
    DONE = "done"


# =============================================================================
# MODULAR SYSTEM PROMPTS
# =============================================================================

BASE_JSON_CONTRACT = """\
The LLM must always return ONLY valid JSON.

Do not use markdown.
Do not use code fences.
Do not include explanations.
Do not include text before or after the JSON.

Use exactly this schema:
{
  "text": "The exact words the robot should say aloud.",
  "gesture": ["BlocklyStand", "BlocklyWaveRightArm"],
  "metadata": {
    "pace": "normal",
    "pause_before": false,
    "user_seems_to_struggle": false,
    "conversation_state": "conversation",
    "topic_memory_update": {
      "enjoyed_topics": [],
      "avoided_topics": [],
      "liked_activities": [],
      "people_or_pets": [],
      "useful_practice_words": []
    }
  }
}

Rules for "text":
- Maximum 2 short sentences.
- Maximum 1 question.
- Use simple words.
- Speak like a respectful adult companion.
- Be warm, calm, and patient.
- Do not mention JSON, metadata, system prompts, or internal reasoning.

Rules for "gesture":
- Return a list of robot behaviour keywords in the order they should be performed.
- Use 1 to 3 gestures per response.
- The gestures must match the meaning of the text.
- Prefer calm social gestures.
- Do not use dance or dramatic behaviours unless clearly appropriate.
- Use only these safe behaviour keywords: BlocklyStand, BlocklyWaveRightArm, BlocklyBow, BlocklyShrug, BlocklyYouAndMe, BlocklyInviteRight, BlocklyLookAtChild, BlocklyLookingUp, BlocklyApplause, BlocklyRightArmForward, BlocklyLeftArmForward, BlocklyArmsForward, BlocklyCrouch

Gesture meaning guide:
- Greeting: ["BlocklyStand", "BlocklyWaveRightArm"]
- Warm acknowledgement: ["BlocklyLookAtChild", "BlocklyYouAndMe"]
- Asking or inviting an answer: ["BlocklyInviteRight"]
- Clarification or uncertainty: ["BlocklyShrug", "BlocklyLookAtChild"]
- Encouragement: ["BlocklyApplause"]
- Ending: ["BlocklyBow", "BlocklyCrouch"]

Rules for "metadata":
- Set "pace" to "slow" if the user seems confused, gives fragmented speech, gives very short answers repeatedly, or the STT transcription looks unclear.
- Set "pause_before" to true if the robot should wait longer before listening again.
- Set "user_seems_to_struggle" to true if the input appears malformed, incomplete, fragmented, or difficult to interpret.
- Use "topic_memory_update" to store only information clearly stated by the user. Do not invent user facts.

Handling unclear or ungrammatical speech:
If the transcribed input looks malformed, fragmented, contradictory, or unclear, do not guess. Gently confirm what you understood or offer two possible interpretations. Example: "I may have heard music. Did you mean music or movies?"

Topic memory:
At the start of the session, the robot may receive a loaded user profile. Use it naturally. Do not overuse memory. Respect avoided topics.
"""

SCENARIOS: Dict[str, Dict[str, str]] = {
    "1": {
        "name": "Small Talk",
        "prompt": """\
SYSTEM PROMPT 1: SMALL TALK / INTRODUCTORY CONVERSATION
You are a friendly and respectful social robot having an introductory getting-to-know-you conversation with a person with aphasia.

CONTEXT
Your goal is to make the user feel comfortable and understood while collecting basic information that can help personalize future interactions. You are not a doctor, therapist, or diagnostic tool. Do not diagnose the user. 
Use the profile gently and naturally. Do not mention that you are reading a profile.

MAIN CONVERSATION RULES
- Speak like a respectful adult companion, not like a teacher or a child.
- Ask only one question at a time. Prefer yes/no questions or simple choice questions.
- If the user's answer is unclear, gently confirm what you understood.
- If the user seems stuck, offer two simple choices.
- Keep the conversation focused on safe personal topics: name, hobbies, music, food, family/pets, daily routines.
- After around 5 user turns, ask whether they want to continue.

TURN-TAKING BEHAVIOUR
- If the user gives a short answer, respond warmly and ask one easy follow-up question.
- If the user gives no answer or the STT result is unclear, say it is okay and ask a simpler question.
"""
    },
    "2": {
        "name": "Therapy Practice Session",
        "prompt": """\
SYSTEM PROMPT 2: THERAPY PRACTICE SESSION
You are a friendly and respectful social robot supporting a simple language-practice session with a person with aphasia.

CONTEXT
Your goal is to help the user practise everyday words and short phrases in a calm, low-pressure way. You are not a doctor, therapist, or diagnostic tool. 
Use the profile to choose motivating practice words (e.g., if they like music, practice words like "song", "radio").

MAIN SESSION GOALS
- Practise simple, useful words or short everyday phrases. Keep the task easy and positive.
- Ask the user to repeat, choose, name, or complete simple phrases.
- If the user struggles, make the task easier. If the user succeeds, give calm encouragement. Do not correct harshly.
- End or switch task if the user appears tired or asks to stop.

THERAPY PRACTICE STYLE
- Use one small task at a time (naming, choice, phrase completion, yes/no practice, repetition).
- Offer two choices if the user is stuck.
- Use "pace": "slow" and "pause_before": true when the user gives fragmented speech or seems stuck.
"""
    },
    "3": {
        "name": "Social Roleplay",
        "prompt": """\
SYSTEM PROMPT 3: SOCIAL ROLEPLAY SESSION
You are a friendly and respectful social robot practising a simple social roleplay with a person with aphasia.

CONTEXT
Your goal is to help the user practise everyday conversation situations in a safe and supportive way. 
Use the profile to choose relevant roleplay situations (e.g., ordering coffee, greeting a friend, asking for help).

MAIN SESSION RULES
- Clearly introduce the roleplay (e.g., "Let's practise ordering coffee.").
- Keep the situation realistic and simple. Stay in character, but keep the interaction easy.
- If the user gets stuck, offer two possible replies.
- After a few turns, summarize what went well in one short sentence. Do not judge the user's speech.
- Use "pace": "slow" and "pause_before": true when the user appears stuck or STT text is unclear.
"""
    }
}


# =============================================================================
# PROFILE MANAGER
# =============================================================================

class ProfileManager:
    """Handles all profile file I/O and data transformations."""

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
            "participant_id": pid,
            "preferred_name": None,
            "enjoyed_topics": [],
            "liked_activities": [],
            "people_or_pets": [],
            "communication_preferences": {
                "question_type": None,
                "needs_extra_time": None,
                "helpful_supports": []
            },
            "topics_to_avoid": [],
            "future_conversation_suggestions": [],
            "session_notes": {
                "what_went_well": [],
                "what_was_hard": [],
                "support_that_helped": [],
                "possible_stt_or_transcription_issues": []
            },
            "suggested_next_scenario": {
                "scenario": None,
                "reason": None
            }
        }

    @staticmethod
    def load(pid: str) -> Dict[str, Any]:
        """Load from disk. Returns blank profile on missing file or JSON corruption."""
        ProfileManager.ensure_dirs()
        path = os.path.join(PROFILES_DIR, f"{pid}.json")

        default_profile = ProfileManager.empty_profile(pid)

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Merge loaded data into the empty profile structure to guarantee all keys exist
                # (Handles migrating old profiles to the new schema automatically)
                for key, default_val in default_profile.items():
                    if key not in data:
                        data[key] = default_val

                print(f"[PROFILE] Loaded profile for '{pid}'.")
                return data
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"[PROFILE] Warning — corrupt profile ({exc}). Starting fresh.")
        else:
            print(
                f"[PROFILE] No profile found for '{pid}'. Creating new profile.")

        return default_profile

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
    def merge_updates(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge LLM-generated post-session data into the stored profile."""

        # Merge lists without duplicates
        for field in ("enjoyed_topics", "avoided_topics", "liked_activities", "people_or_pets", "future_conversation_suggestions"):
            incoming = updates.get(field) or []
            if isinstance(incoming, list):
                current = existing.get(field, [])
                seen = set(current)
                existing[field] = current + \
                    [t for t in incoming if t not in seen]

        # Merge dicts (communication_preferences, session_notes, suggested_next_scenario)
        for dict_field in ("communication_preferences", "session_notes", "suggested_next_scenario"):
            incoming = updates.get(dict_field) or {}
            if isinstance(incoming, dict):
                current = existing.get(dict_field, {})
                for k, v in incoming.items():
                    # Only overwrite if LLM provided a non-empty/non-null value, or append if it's a list
                    if isinstance(v, list):
                        current_list = current.get(k, [])
                        seen = set(current_list)
                        current[k] = current_list + \
                            [i for i in v if i not in seen]
                    elif v:
                        current[k] = v
                existing[dict_field] = current

        # Merge string/null overrides
        if updates.get("preferred_name"):
            existing["preferred_name"] = updates["preferred_name"]

        return existing

    @staticmethod
    def build_memory_block(profile: Dict[str, Any]) -> str:
        """Convert stored profile data into a plain-English paragraph for the system prompt."""
        p = profile
        lines: List[str] = []
        if p.get("preferred_name"):
            lines.append(
                f"The user's preferred name is {p['preferred_name']}.")

        comms = p.get("communication_preferences", {})
        if comms.get("question_type"):
            lines.append(f"They prefer {comms['question_type']} questions.")
        if comms.get("needs_extra_time"):
            lines.append("They need extra time to process questions.")

        if p.get("enjoyed_topics"):
            lines.append(
                f"Topics they enjoy: {', '.join(p['enjoyed_topics'])}.")
        if p.get("topics_to_avoid"):
            lines.append(
                f"Topics to avoid: {', '.join(p['topics_to_avoid'])}.")

        next_scen = p.get("suggested_next_scenario", {})
        if next_scen.get("scenario"):
            lines.append(
                f"Suggested focus for today: {next_scen['scenario']} ({next_scen.get('reason', '')})")

        if not lines:
            return ""
        return "\nPERSONALIZATION MEMORY (from previous sessions):\n" + "\n".join(lines) + "\n"


# =============================================================================
# LLM MANAGER
# =============================================================================

class LLMManager:
    """Wraps all OpenAI API calls. Runs synchronously in Twisted worker threads."""

    def __init__(self) -> None:
        self.client = OpenAI()

    def _sync_get_response(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, List[str], Dict[str, Any], Optional[str]]:
        """
        Synchronous LLM call with exponential-backoff retry.
        Returns (robot_speech, gestures_list, metadata_dict, error_or_None).
        """
        attempts = 1 + len(LLM_RETRY_DELAYS)
        for attempt in range(attempts):
            try:
                response = self.client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=LLM_TIMEOUT_SECS,
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)

                speech = parsed.get("text", "").strip()

                # Handle gesture list natively
                gestures = parsed.get("gesture", ["BlocklyWaveRightArm"])
                if isinstance(gestures, str):
                    gestures = [gestures]
                if not isinstance(gestures, list) or not gestures:
                    gestures = ["BlocklyWaveRightArm"]

                metadata = parsed.get("metadata", {})

                if not speech:
                    raise ValueError("'text' empty in LLM JSON")

                return speech, gestures, metadata, None

            except (APIError, APIConnectionError, APITimeoutError) as exc:
                err = f"API error attempt {attempt + 1}: {exc}"
            except (json.JSONDecodeError, ValueError, KeyError) as exc:
                err = f"Parse error attempt {attempt + 1}: {exc}"
            except Exception as exc:
                err = f"Unexpected error attempt {attempt + 1}: {exc}"

            print(f"[LLM] {err}")

            if attempt < len(LLM_RETRY_DELAYS):
                delay = LLM_RETRY_DELAYS[attempt]
                print(f"[LLM] Retrying in {delay}s...")
                time.sleep(delay)

        return LLM_FALLBACK_SPEECH, ["BlocklyWaveRightArm"], {"pace": "slow"}, err

    def _sync_extract_profile(
        self, conversation: List[Dict[str, str]]
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Extracts nested profile updates post-session."""
        profile_prompt = """\
Analyze the conversation and create a personalization profile for future robot interactions with a person with aphasia.
Return ONLY valid JSON.
Do not use markdown.
Do not use code fences.
Do not include explanations.
Do not include any text before or after the JSON.
Use this exact schema:
{
  "preferred_name": null,
  "enjoyed_topics": [],
  "liked_activities": [],
  "people_or_pets": [],
  "communication_preferences": {
    "question_type": null,
    "needs_extra_time": null,
    "helpful_supports": []
  },
  "topics_to_avoid": [],
  "future_conversation_suggestions": [],
  "session_notes": {
    "what_went_well": [],
    "what_was_hard": [],
    "support_that_helped": [],
    "possible_stt_or_transcription_issues": []
  },
  "suggested_next_scenario": {
    "scenario": null,
    "reason": null
  }
}
Rules:
- Use null if unknown. Use [] if none found.
- Do not invent facts. Base everything only on the conversation.
- Do not diagnose the user. Do not label the user clinically.
- "question_type" must be one of: "yes_no", "choice", "open", "mixed", or null.
- "needs_extra_time" must be true, false, or null.
- "helpful_supports" can include: "repeat", "rephrase", "write_keywords", "two_choices", "yes_no_questions", "slower_pace".
- "suggested_next_scenario.scenario" must be one of: "small_talk", "therapy_practice", "social_roleplay", or null.
- Choose "small_talk" if the user seemed comfortable and shared personal interests.
- Choose "therapy_practice" if the user struggled with specific words or short phrases.
- Choose "social_roleplay" if the user would benefit from practising everyday situations.
"""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": profile_prompt},
                    {"role": "user", "content": json.dumps(
                        conversation, ensure_ascii=False)}
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
    """Owns all session state and robot behaviour."""

    def __init__(self, session, config: Dict[str, Any]) -> None:
        self.session = session
        self.participant_id = config["participant_id"]
        self.profile = config["profile"]
        self.scenario = config["scenario"]
        self.llm = LLMManager()

        # Assemble the modular prompts
        self.system_prompt: str = (
            BASE_JSON_CONTRACT + "\n\n" +
            self.scenario["prompt"] + "\n\n" +
            ProfileManager.build_memory_block(self.profile)
        )

        self.conversation_history: List[Dict[str, str]] = []
        self.session_log: List[Dict[str, Any]] = []

        # --- STT state ---
        self.robot_speaking:   bool = False
        self.user_input:       str = ""
        self.new_input_ready:  bool = False
        self._stt_buffer:      List[str] = []
        self._stt_timer = None
        self._stt_open:        bool = False
        self._stt_open_time:   float = 0.0

        # --- Timing ---
        self.session_start:      float = time.time()
        self.session_start_iso:  str = datetime.now().isoformat()
        self._user_turn_start:   float = 0.0
        self.response_times:     List[float] = []
        self.slow_pace_count:    int = 0

        # --- FSM state ---
        self.state: RobotState = RobotState.INIT
        self._state_since:    float = time.time()

        # --- Watchdog ---
        self._watchdog: Optional[LoopingCall] = None

        # --- Session outcome ---
        self.exit_reason:         str = "normal_end"
        self.errors_encountered:  List[str] = []
        self._log_saved:          bool = False

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}][{self.state.value.upper()}] {msg}")

    def _log_error(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{ts}][ERROR] {msg}")
        self.errors_encountered.append(f"{ts}: {msg}")

    def _set_state(self, new_state: RobotState) -> None:
        if new_state != self.state:
            self._log(f"→ {new_state.value}")
            self.state = new_state
            self._state_since = time.time()

    def _start_watchdog(self) -> None:
        self._watchdog = LoopingCall(self._watchdog_check)
        self._watchdog.start(WATCHDOG_INTERVAL, now=False)
        self._log("Watchdog started.")

    def _stop_watchdog(self) -> None:
        if self._watchdog and self._watchdog.running:
            self._watchdog.stop()

    def _watchdog_check(self) -> None:
        if self.state == RobotState.DONE:
            self._stop_watchdog()
            return

        elapsed = time.time() - self._state_since

        if self.state == RobotState.LISTENING:
            stt_idle = time.time() - self._stt_open_time
            if stt_idle > STT_SILENCE_TIMEOUT:
                self._log_error(
                    f"[WATCHDOG] STT silent for {stt_idle:.0f}s — restarting stream")
                reactor.callLater(0, self._restart_stt)
                return

        if self.state == RobotState.SPEAKING and elapsed > WATCHDOG_FREEZE_TIMEOUT:
            self._log_error(
                f"[WATCHDOG] Stuck in SPEAKING for {elapsed:.0f}s — resetting")
            self.robot_speaking = False
            self._set_state(RobotState.LISTENING)
            reactor.callLater(0, self._restart_stt)
            return

        if self.state == RobotState.PROCESSING and elapsed > WATCHDOG_FREEZE_TIMEOUT:
            self._log_error(
                f"[WATCHDOG] LLM processing for {elapsed:.0f}s — still waiting")
            self._state_since = time.time()

    @inlineCallbacks
    def _wamp(self, method: str, *args, **kwargs):
        d = self.session.call(method, *args, **kwargs)

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

    @inlineCallbacks
    def _open_stt(self) -> None:
        yield deferLater(reactor, POST_SPEECH_MIC_DELAY, lambda: None)
        result = yield self._wamp("rie.dialogue.stt.stream")
        if result is not None or True:
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
        self._log("Restarting STT stream...")
        self._stt_open = False
        self._cancel_stt_timer()
        yield self._wamp("rie.dialogue.stt.close")
        yield deferLater(reactor, 0.5, lambda: None)
        yield self._open_stt()

    def _cancel_stt_timer(self) -> None:
        if self._stt_timer is not None:
            try:
                if self._stt_timer.active():
                    self._stt_timer.cancel()
            except Exception:
                pass
            self._stt_timer = None

    def asr_callback(self, frames) -> None:
        if self.robot_speaking:
            return

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
        self._stt_open_time = time.time()

        self._cancel_stt_timer()
        self._stt_timer = reactor.callLater(
            STT_BUFFER_DELAY, self._finalize_stt)

    def _finalize_stt(self) -> None:
        self._stt_timer = None
        full_text = " ".join(self._stt_buffer).strip()
        self._stt_buffer.clear()

        if not full_text:
            return

        if self.state not in (RobotState.LISTENING, RobotState.CLARIFYING):
            self._log(
                f"[STT] Discarded (state={self.state.value}): '{full_text}'")
            return

        self._log(f"[STT] Finalized utterance: '{full_text}'")

        if self._user_turn_start > 0:
            elapsed = time.time() - self._user_turn_start
            self.response_times.append(elapsed)
            self._log(f"[PACING] Response time: {elapsed:.1f}s")
            self._user_turn_start = 0.0

        self.user_input = full_text
        self.new_input_ready = True

    @inlineCallbacks
    def _speak(self, text: str, gestures: List[str] = None) -> None:
        self._set_state(RobotState.SPEAKING)
        self.robot_speaking = True
        self._log(f"[TTS] '{text[:80]}{'...' if len(text) > 80 else ''}'")

        if not gestures:
            gestures = ["BlocklyWaveRightArm"]

        try:
            # Play first gesture alongside speech
            movement = self._wamp(
                "rom.optional.behavior.play", name=gestures[0])
            yield self._wamp("rie.dialogue.say", text=text, lang="en")
            yield movement

            # Queue up any remaining gestures requested by the LLM
            for g in gestures[1:]:
                yield self._wamp("rom.optional.behavior.play", name=g)

        finally:
            self.robot_speaking = False

    def _calculate_pause(self, pace: str) -> float:
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

    @staticmethod
    def _is_unclear(text: str) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return True
        if len(cleaned) == 1 and cleaned.lower() not in VALID_SHORT_WORDS:
            return True
        return False

    def _log_turn(self, role: str, text: str) -> None:
        if role in ("user", "assistant"):
            self.conversation_history.append({"role": role, "content": text})

        self.session_log.append({
            "speaker":   "robot" if role == "assistant" else role,
            "text":      text,
            "timestamp": datetime.now().isoformat(),
        })

    def save_session_log(self) -> None:
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
            "questionnaire":      {},
        }
        try:
            ProfileManager.ensure_dirs()
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[LOG] Session saved → {filename}")
        except OSError as exc:
            print(f"[LOG] Error saving session log: {exc}")

    @inlineCallbacks
    def start(self) -> None:
        # ── Hardware init ─────────────────────────────────────────────────────
        self._set_state(RobotState.INIT)
        yield self._wamp("rie.dialogue.config.language", lang="en")
        yield self._wamp("rom.optional.behavior.play", name="BlocklyStand")
        yield self.session.subscribe(self.asr_callback, "rie.dialogue.stt.stream")

        self._start_watchdog()

        # ── Greeting ─────────────────────────────────────────────────────────
        self._set_state(RobotState.GREETING)
        name = self.profile.get("preferred_name", "")

        # Access nested summary from the new profile structure
        next_scen_dict = self.profile.get("suggested_next_scenario", {})
        suggested = next_scen_dict.get("scenario", "")

        if name and suggested:
            greeting = f"Hello {name}! Great to see you again. Are you ready to start today?"
        elif name:
            greeting = f"Hello {name}! I am your robot assistant. How are you feeling today?"
        else:
            greeting = "Hello! I am your robot assistant. What is your name?"

        self._log_turn("assistant", greeting)
        yield self._speak(greeting, gestures=["BlocklyWaveRightArm"])
        yield self._open_stt()
        self._set_state(RobotState.LISTENING)
        self._user_turn_start = time.time()

        # ── Dialogue loop ─────────────────────────────────────────────────────
        while True:

            if not self.new_input_ready:
                yield deferLater(reactor, 0.3, lambda: None)

                if time.time() - self.session_start > MAX_SESSION_DURATION:
                    self._set_state(RobotState.CLOSING)
                    self.exit_reason = "time_limit"
                    timeout_farewell = "Our time is up for today. Thank you so much for talking with me!"
                    self._log_turn("assistant", timeout_farewell)
                    yield self._speak(timeout_farewell, gestures=["BlocklyWaveRightArm"])
                    break

                continue

            current_input = self.user_input
            self.user_input = ""
            self.new_input_ready = False

            if any(p in current_input.lower() for p in EXIT_PHRASES):
                self._set_state(RobotState.CLOSING)
                self.exit_reason = "user_exit_phrase"
                farewell = "It was lovely talking to you. Take care and goodbye!"
                self._log_turn("user",      current_input)
                self._log_turn("assistant", farewell)
                yield self._speak(farewell, gestures=["BlocklyWaveRightArm"])
                break

            if time.time() - self.session_start > MAX_SESSION_DURATION:
                self._set_state(RobotState.CLOSING)
                self.exit_reason = "time_limit"
                closing_text = CLOSING_PROMPT + current_input
                messages = (
                    [{"role": "system", "content": self.system_prompt}]
                    + self.conversation_history
                    + [{"role": "user", "content": closing_text}]
                )
                speech, gestures, metadata, err = yield threads.deferToThread(
                    self.llm._sync_get_response, messages
                )
                if err:
                    self._log_error(err)
                self._log_turn("user",      current_input)
                self._log_turn("assistant", speech)
                yield self._speak(speech, gestures=gestures)
                break

            if self._is_unclear(current_input):
                self._log(f"[INPUT] Unclear: '{current_input}'")
                self._set_state(RobotState.CLARIFYING)
                clarification = "Sorry, I missed that. Could you say it again slowly?"
                self._log_turn("user",      current_input)
                self._log_turn("assistant", clarification)

                yield deferLater(reactor, PAUSE_SLOW_BONUS, lambda: None)
                yield self._speak(clarification, gestures=["BlocklyShrug", "BlocklyLookAtChild"])
                yield self._open_stt()
                self._set_state(RobotState.LISTENING)
                self._user_turn_start = time.time()
                continue

            self._set_state(RobotState.PROCESSING)

            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + self.conversation_history
                + [{"role": "user", "content": current_input}]
            )

            t0 = time.time()
            speech, gestures, metadata, err = yield threads.deferToThread(
                self.llm._sync_get_response, messages
            )
            self._log(f"[LLM] Responded in {time.time()-t0:.1f}s")

            if err:
                self._log_error(err)

            self._log_turn("user",      current_input)
            self._log_turn("assistant", speech)

            # Apply LLM metadata for dynamic pacing
            pace = metadata.get("pace", "normal")
            pause_before = metadata.get("pause_before", False)

            if pause_before:
                self._log("[PACING] LLM requested pause_before")
                yield deferLater(reactor, PAUSE_SLOW_BONUS, lambda: None)

            yield self._speak(speech, gestures=gestures)

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

        print("\n── Session Transcript ─────────────────────────────────────")
        for entry in self.session_log:
            label = "ROBOT" if entry["speaker"] == "robot" else "USER"
            print(f"  [{label}] {entry['text']}")
        print("────────────────────────────────────────────────────────────\n")

        print("[PROFILE] Analysing session...")
        updates, err = yield threads.deferToThread(
            self.llm._sync_extract_profile,
            self.conversation_history,
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
# TERMINAL SETUP
# =============================================================================

def terminal_setup() -> Dict[str, Any]:
    ProfileManager.ensure_dirs()

    print("\n" + "=" * 56)
    print("  APHASIA ROBOT SESSION SETUP")
    print("=" * 56)

    while True:
        raw = input("\nEnter participant first and last name: ").strip()
        parts = raw.split()
        if len(parts) >= 2:
            break
        print("  Please enter both a first name and a last name.")

    pid = ProfileManager.make_participant_id(parts[0], parts[-1])
    profile = ProfileManager.load(pid)

    if not profile.get("preferred_name"):
        profile["preferred_name"] = parts[0].capitalize()

    # Read nested properties from updated profile format for setup hints
    next_scen = profile.get("suggested_next_scenario", {})
    if next_scen.get("scenario"):
        print(
            f"\n  Suggested today: {next_scen['scenario']} ({next_scen.get('reason', '')})")

    notes = profile.get("session_notes", {})
    if notes.get("what_was_hard"):
        print(f"  Previously hard: {', '.join(notes['what_was_hard'])}")

    print("\nSelect a scenario:")
    for key, sc in SCENARIOS.items():
        print(f"  [{key}] {sc['name']}")

    while True:
        raw_choice = input(
            "\nEnter scenario number (default 1): ").strip() or "1"
        if raw_choice in SCENARIOS:
            break
        print(f"  Please enter a number between 1 and {len(SCENARIOS)}.")

    scenario = SCENARIOS[raw_choice]
    print(f"\n  Participant : {pid}")
    print(f"  Scenario    : {scenario['name']}")
    print("=" * 56 + "\n")

    return {"participant_id": pid, "profile": profile, "scenario": scenario}


# =============================================================================
# WAMP COMPONENT BINDING + ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    config = terminal_setup()
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
