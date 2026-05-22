"""
robot.py – Production-Ready Social Robot for Aphasia Speech Therapy.

Features:
  - Class-based state management (no globals).
  - Robust LLM handling with retries, timeouts, and JSON-enforced outputs.
  - Dynamic pacing (tracks user response time and LLM pace signals).
  - Clarification middleware for unintelligible STT inputs.
  - Persistent profile memory injected across sessions.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from autobahn.twisted.component import Component, run
from autobahn.twisted.util import sleep
from twisted.internet.defer import inlineCallbacks

import openai
from openai import OpenAI

# =============================================================================
# 1. CONFIG & CONSTANTS
# =============================================================================

MAX_DURATION = 60 * 5        # Maximum session length in seconds
WAMP_REALM = "rie.69f352c326d8af1680827d4a"
LLM_MODEL = "gpt-4o"

PROFILES_DIR = "profiles"
SESSIONS_DIR = "sessions"

PACE_NORMAL_SLEEP = 1.0           # base pause (seconds) before reopening mic
PACE_SLOW_SLEEP = 2.5           # extended pause for struggling users
# if avg user response time > this, auto-slow
SLOW_RESPONSE_THRESHOLD = 6.0
MIN_WORD_COUNT = 1

EXIT_PHRASES = (
    "goodbye", "bye", "quit", "exit",
    "stop", "that's all", "thats all", "no more",
)

# =============================================================================
# 2. SCENARIOS & PROMPTS
# =============================================================================

SCENARIO_MENU = {
    1: "Small Talk / Getting to Know You",
    2: "Asking for Help",
    3: "Daily Routine",
    4: "Describing Feelings",
    5: "Social Roleplay",
}

_BASE_RULES = """
COMMUNICATION RULES:
- Speak like a respectful adult companion, not a teacher or a child.
- Use short, simple sentences (maximum 2 sentences per turn).
- Ask only ONE question per turn.
- Prefer yes/no or simple choice questions.
- If the user's answer is unclear, gently confirm what you understood.
- If the user seems stuck, offer two simple choices.
- Do not interrupt or finish the user's sentences.
- Avoid jargon, complex wordplay, or emotionally sensitive topics.

RESPONSE FORMAT:
You MUST reply with a valid JSON object.
Required format:
{"robot_speech": "Your spoken text here.", "pace": "normal"}

"pace" must be either "normal" or "slow". Use "slow" when the user seems to struggle, gave a short answer, or appears to need extra time.
"""

SCENARIO_PROMPTS = {
    1: "You are a friendly social robot having a getting-to-know-you conversation with a person with aphasia. Learn their name, interests, hobbies, pets, and daily life.\n" + _BASE_RULES,
    2: "You are a friendly social robot helping a person with aphasia practice asking for help. Role-play everyday situations: asking a shop assistant, asking a friend, etc.\n" + _BASE_RULES,
    3: "You are a friendly social robot helping a person with aphasia talk about their daily routine. Ask about mornings, meals, activities, and bedtime. Keep questions concrete.\n" + _BASE_RULES,
    4: "You are a friendly social robot helping a person with aphasia practice describing feelings. Use simple emotion words: happy, sad, tired, excited, worried, calm.\n" + _BASE_RULES,
    5: "You are a friendly social robot helping a person with aphasia practice social interactions. Role-play situations like greeting someone, ordering food, or talking to a neighbour.\n" + _BASE_RULES,
}

CLOSING_PROMPT = (
    "The conversation time is up. Briefly respond to what the user just said, "
    "then say a warm farewell. User said: "
)

# =============================================================================
# 3. PROFILE MANAGER
# =============================================================================


class ProfileManager:
    @staticmethod
    def ensure_dirs():
        os.makedirs(PROFILES_DIR, exist_ok=True)
        os.makedirs(SESSIONS_DIR, exist_ok=True)

    @staticmethod
    def make_participant_id(first: str, last: str) -> str:
        return re.sub(r'[^a-z0-9_]', '', f"{first.strip().lower()}_{last.strip().lower()}")

    @staticmethod
    def get_empty_profile(pid: str) -> Dict[str, Any]:
        return {
            "participant_id": pid,
            "preferred_name": "",
            "preferred_question_style": "",
            "enjoyed_topics": [],
            "avoided_topics": [],
            "session_notes": [],
            "last_session_summary": "",
            "suggested_next_scenario": "",
        }

    @staticmethod
    def load_profile(pid: str) -> Dict[str, Any]:
        ProfileManager.ensure_dirs()
        path = os.path.join(PROFILES_DIR, f"{pid}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure list fields exist even in older profiles
            for field in ["enjoyed_topics", "avoided_topics", "session_notes"]:
                if field not in data:
                    data[field] = []
            print(f"[PROFILE] Loaded existing profile for '{pid}'.")
            return data
        print(f"[PROFILE] No profile found for '{pid}'. Starting fresh.")
        return ProfileManager.get_empty_profile(pid)

    @staticmethod
    def save_profile(pid: str, data: Dict[str, Any]) -> None:
        ProfileManager.ensure_dirs()
        path = os.path.join(PROFILES_DIR, f"{pid}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[PROFILE] Profile saved → {path}")

    @staticmethod
    def build_memory_block(profile: Dict[str, Any]) -> str:
        lines = []
        if profile.get("preferred_name"):
            lines.append(
                f"The user's preferred name is {profile['preferred_name']}.")
        if profile.get("preferred_question_style"):
            lines.append(
                f"They prefer {profile['preferred_question_style']} questions.")
        if profile.get("enjoyed_topics"):
            lines.append(
                f"Topics they enjoy: {', '.join(profile['enjoyed_topics'])}.")
        if profile.get("avoided_topics"):
            lines.append(
                f"Topics to avoid: {', '.join(profile['avoided_topics'])}.")
        if profile.get("last_session_summary"):
            lines.append(
                f"Last session summary: {profile['last_session_summary']}")

        if not lines:
            return ""
        return "\nPERSONALIZATION MEMORY (from previous sessions):\n" + "\n".join(lines) + "\n"

    @staticmethod
    def merge_updates(existing: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        list_fields = ["enjoyed_topics", "avoided_topics"]
        str_fields = ["preferred_name", "preferred_question_style",
                      "last_session_summary", "suggested_next_scenario"]

        for field in list_fields:
            new_items = updates.get(field) or []
            if isinstance(new_items, list):
                combined = existing.get(field, []) + new_items
                seen = set()
                existing[field] = [x for x in combined if not (
                    x in seen or seen.add(x))]

        for field in str_fields:
            value = updates.get(field) or ""
            if value:
                existing[field] = value

        # Handle session notes appending
        note = updates.get("session_notes")
        if note and isinstance(note, str):
            if "session_notes" not in existing:
                existing["session_notes"] = []
            existing["session_notes"].append({
                "date": datetime.now().isoformat(),
                "note": note
            })

        return existing


# =============================================================================
# 4. LLM MANAGER
# =============================================================================

class LLMManager:
    def __init__(self):
        self.client = OpenAI()

    def get_chat_response(self, messages: List[Dict[str, str]], retry: bool = False) -> Tuple[str, str]:
        """Calls OpenAI explicitly requesting JSON. Returns (speech, pace)."""
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=15.0
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            speech = data.get("robot_speech", "").strip()
            pace = data.get("pace", "normal").lower()

            if not speech:
                raise ValueError("robot_speech key missing or empty in JSON.")

            return speech, pace

        except (openai.APIConnectionError, openai.APITimeoutError, openai.APIError, ValueError, json.JSONDecodeError) as e:
            print(f"[LLM Error] {str(e)}")
            if not retry:
                print("[LLM] Retrying once...")
                time.sleep(1)
                return self.get_chat_response(messages, retry=True)

            print("[LLM] Fallback triggered.")
            return "Sorry, I lost my train of thought. Could you say that again?", "slow"

    def extract_profile_updates(self, conversation: List[Dict[str, str]], pid: str, scenario_id: int) -> Dict[str, Any]:
        """Analyzes session transcript to extract profile learnings."""
        scenario_label = SCENARIO_MENU.get(scenario_id, 'Unknown')
        prompt = f"""
        Analyse this conversation with participant '{pid}' (Scenario: {scenario_label}).
        Produce a JSON update based ONLY on facts revealed in the transcript.
        
        Required Schema (return JSON only):
        {{
          "preferred_name": "extracted name or null",
          "preferred_question_style": "brief observation on what question format worked best",
          "enjoyed_topics": ["topic1", "topic2"],
          "avoided_topics": ["topic1"],
          "session_notes": "1-2 sentence summary of communication challenges and strengths",
          "last_session_summary": "1-2 sentence summary of what was discussed",
          "suggested_next_scenario": "idea for next session based on today"
        }}
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(
                conversation, ensure_ascii=False)}
        ]

        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=20.0
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[PROFILE UPDATE ERROR] {e}")
            return {}


# =============================================================================
# 5. ROBOT CONTROLLER (WAMP STATE MACHINE)
# =============================================================================

class RobotController:
    def __init__(self, session, config: Dict[str, Any]):
        self.session = session
        self.config = config
        self.pid = config["participant_id"]
        self.profile = config["profile"]
        self.scenario_id = config["scenario_id"]

        self.llm = LLMManager()

        # State tracking
        self.conversation: List[Dict[str, str]] = []
        self.raw_log: List[Dict[str, Any]] = []
        self.response_times: List[float] = []

        self.user_input = ""
        self.new_input_ready = False
        self.robot_speaking = False

        self.session_start_time = time.time()
        self.user_turn_start: Optional[float] = None
        self.exit_reason = "normal_end"

        # Build System Prompt
        base_prompt = SCENARIO_PROMPTS.get(
            self.scenario_id, SCENARIO_PROMPTS[1])
        memory_block = ProfileManager.build_memory_block(self.profile)
        self.system_prompt = base_prompt + "\n" + memory_block

    def is_unclear_input(self, text: str) -> bool:
        cleaned = text.strip()
        if not cleaned or len(cleaned) <= 2:
            return True
        words = cleaned.split()
        return len(words) < MIN_WORD_COUNT

    def calculate_sleep_duration(self, pace_flag: str) -> float:
        """Determines pause length before reopening STT stream."""
        if pace_flag == "slow":
            return PACE_SLOW_SLEEP

        recent = self.response_times[-5:]
        if len(recent) >= 3 and (sum(recent) / len(recent)) > SLOW_RESPONSE_THRESHOLD:
            print("[PACING] Auto-slow triggered due to long user response times.")
            return PACE_SLOW_SLEEP

        return PACE_NORMAL_SLEEP

    def log_turn(self, role: str, text: str):
        if role != "system":
            self.conversation.append({"role": role, "content": text})

        speaker = "robot" if role == "assistant" else "user"
        self.raw_log.append({
            "speaker": speaker,
            "text": text,
            "timestamp": datetime.now().isoformat()
        })
        print(f"[{speaker.upper()}] {text}")

    def save_session(self):
        duration = round(time.time() - self.session_start_time, 2)
        ts = datetime.now().isoformat().replace(":", "-")
        filename = os.path.join(SESSIONS_DIR, f"{self.pid}_{ts}.json")

        log_data = {
            "participant_id": self.pid,
            "scenario": SCENARIO_MENU.get(self.scenario_id, "Unknown"),
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "exit_reason": self.exit_reason,
            "conversation": self.raw_log,
            "questionnaire": {}
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"[LOG] Session saved → {filename}")

    # --- Callbacks ---
    def asr_callback(self, frames):
        if self.robot_speaking:
            return

        if frames["data"]["body"]["final"]:
            transcript = str(frames["data"]["body"]["text"]).strip()
            if transcript:
                self.user_input = transcript
                self.new_input_ready = True

                if self.user_turn_start:
                    elapsed = time.time() - self.user_turn_start
                    self.response_times.append(elapsed)
                    print(f"[PACING] Response time: {elapsed:.1f}s")
                    self.user_turn_start = None

    # --- Main WAMP Loop ---
    @inlineCallbacks
    def start(self):
        yield self.session.call("rie.dialogue.config.language", lang="en")
        yield self.session.call("rom.optional.behavior.play", name="BlocklyStand")
        yield self.session.subscribe(self.asr_callback, "rie.dialogue.stt.stream")

        # Greeting
        preferred = self.profile.get("preferred_name", "")
        greeting = f"Hello {preferred}! Great to see you again. Are you ready to start?" if preferred else "Hello! I am your robot assistant. What is your name?"

        self.log_turn("assistant", greeting)

        self.robot_speaking = True
        yield self.session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
        yield self.session.call("rie.dialogue.say_animated", text=greeting)
        self.robot_speaking = False

        yield self.session.call("rie.dialogue.stt.stream")
        self.user_turn_start = time.time()

        # Conversation Loop
        while True:
            if not self.new_input_ready:
                yield sleep(0.3)

                if time.time() - self.session_start_time > MAX_DURATION:
                    self.exit_reason = "time_limit"

                    messages = [{"role": "system", "content": self.system_prompt}] + \
                        self.conversation + \
                        [{"role": "user", "content": CLOSING_PROMPT}]
                    speech, _ = self.llm.get_chat_response(messages)

                    self.log_turn("assistant", speech)
                    self.robot_speaking = True
                    yield self.session.call("rie.dialogue.say_animated", text=speech)
                    self.robot_speaking = False
                    break
                continue

            current_input = self.user_input
            self.user_input = ""
            self.new_input_ready = False

            # 1. Exit Phrase Check
            if any(phrase in current_input.lower() for phrase in EXIT_PHRASES):
                self.exit_reason = "user_exit_phrase"
                farewell = "It was lovely talking to you. Take care and goodbye!"
                self.log_turn("user", current_input)
                self.log_turn("assistant", farewell)

                self.robot_speaking = True
                yield self.session.call("rie.dialogue.say_animated", text=farewell)
                yield self.session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
                self.robot_speaking = False
                break

            # 2. Unclear Input Check
            if self.is_unclear_input(current_input):
                print(f"[INPUT] Unclear/Short: '{current_input}'")
                clarification = "I did not quite catch that. Could you say it again?"
                self.log_turn("user", current_input)
                self.log_turn("assistant", clarification)

                self.robot_speaking = True
                yield self.session.call("rie.dialogue.say_animated", text=clarification)
                self.robot_speaking = False

                yield sleep(PACE_SLOW_SLEEP)
                yield self.session.call("rie.dialogue.stt.stream")
                self.user_turn_start = time.time()
                continue

            # 3. Normal LLM Turn
            self.log_turn("user", current_input)

            messages = [
                {"role": "system", "content": self.system_prompt}] + self.conversation
            speech, pace = self.llm.get_chat_response(messages)

            self.log_turn("assistant", speech)

            self.robot_speaking = True
            yield self.session.call("rie.dialogue.say_animated", text=speech)
            self.robot_speaking = False

            pause = self.calculate_sleep_duration(pace)
            yield sleep(pause)

            yield self.session.call("rie.dialogue.stt.stream")
            self.user_turn_start = time.time()

        # Shutdown & Cleanup
        yield self.session.call("rie.dialogue.stt.close")
        yield self.session.call("rom.optional.behavior.play", name="BlocklyCrouch")

        # Post-session Tasks
        print("\n[SYSTEM] Analyzing session for profile updates...")
        updates = self.llm.extract_profile_updates(
            self.conversation, self.pid, self.scenario_id)
        if updates:
            updated_profile = ProfileManager.merge_updates(
                self.profile, updates)
            ProfileManager.save_profile(self.pid, updated_profile)

        self.save_session()
        self.session.leave()


# =============================================================================
# 6. TERMINAL SETUP & ENTRY POINT
# =============================================================================

def terminal_setup() -> Dict[str, Any]:
    """Runs blockingly before WAMP reactor to configure the session."""
    ProfileManager.ensure_dirs()

    print("\n" + "=" * 52)
    print("   ROBOT THERAPY SESSION SETUP")
    print("=" * 52)

    while True:
        raw = input("\nEnter first name and last name: ").strip()
        parts = raw.split()
        if len(parts) >= 2:
            pid = ProfileManager.make_participant_id(parts[0], parts[-1])
            break
        print("  Please enter both a first name and a last name.")

    profile = ProfileManager.load_profile(pid)

    if profile.get("last_session_summary"):
        print(f"\n  Last session : {profile['last_session_summary']}")
    if profile.get("suggested_next_scenario"):
        print(f"  Suggested    : {profile['suggested_next_scenario']}")

    print("\nSelect a scenario:")
    for key, label in SCENARIO_MENU.items():
        print(f"  {key}. {label}")

    while True:
        try:
            choice = int(input("\nEnter number (1-5): ").strip())
            if choice in SCENARIO_MENU:
                break
            print("  Please enter a valid number.")
        except ValueError:
            print("  Please enter a number.")

    print(f"\n  Participant : {pid}")
    print(f"  Scenario    : {SCENARIO_MENU[choice]}")
    print("=" * 52 + "\n")

    return {
        "participant_id": pid,
        "profile": profile,
        "scenario_id": choice
    }


if __name__ == "__main__":
    session_config = terminal_setup()

    wamp = Component(
        transports=[{
            "url": "ws://wamp.robotsindeklas.nl",
            "serializers": ["msgpack"],
            "max_retries": 0,
        }],
        realm=WAMP_REALM,
    )

    @inlineCallbacks
    def on_join(session, details):
        controller = RobotController(session, session_config)
        yield controller.start()

    wamp.on_join(on_join)

    try:
        run([wamp])
    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt – session ended.")
