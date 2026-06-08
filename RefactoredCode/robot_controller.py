import enum
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from twisted.internet import reactor, threads
from twisted.internet.defer import inlineCallbacks
from twisted.internet.task import deferLater, LoopingCall

from config import *
from prompts import BASE_JSON_CONTRACT, CLOSING_PROMPT
from profile_manager import ProfileManager
from llm_manager import LLMManager


class RobotState(enum.Enum):
    INIT = "init"
    GREETING = "greeting"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    CLARIFYING = "clarifying"
    CLOSING = "closing"
    DONE = "done"


class RobotController:
    """Owns all session state and robot behaviour."""

    def __init__(self, session, config: Dict[str, Any]) -> None:
        self.session = session
        self.participant_id = config["participant_id"]
        self.profile = config["profile"]
        self.scenario = config["scenario"]
        self.llm = LLMManager()

        self.system_prompt: str = (
            BASE_JSON_CONTRACT + "\n\n" +
            self.scenario["prompt"] + "\n\n" +
            ProfileManager.build_memory_block(self.profile)
        )

        self.conversation_history: List[Dict[str, str]] = []
        self.session_log: List[Dict[str, Any]] = []

        # State vars
        self.robot_speaking: bool = False
        self.user_input: str = ""
        self.new_input_ready: bool = False
        self._stt_buffer: List[str] = []
        self._stt_timer = None
        self._stt_open: bool = False
        self._stt_open_time: float = 0.0

        # Timers
        self.session_start: float = time.time()
        self.session_start_iso: str = datetime.now().isoformat()
        self._user_turn_start: float = 0.0
        self.response_times: List[float] = []
        self.slow_pace_count: int = 0

        # FSM and Watchdog
        self.state: RobotState = RobotState.INIT
        self._state_since: float = time.time()
        self._watchdog: Optional[LoopingCall] = None

        self.exit_reason: str = "normal_end"
        self.errors_encountered: List[str] = []
        self._log_saved: bool = False

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
        yield self._wamp("rie.dialogue.stt.stream")
        # Assume successful stream opening, update local state
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
            movement = self._wamp(
                "rom.optional.behavior.play", name=gestures[0])
            yield self._wamp("rie.dialogue.say", text=text, lang="en")
            yield movement

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
            "speaker": "robot" if role == "assistant" else role,
            "text": text,
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
            "participant_id": self.participant_id,
            "scenario": self.scenario["name"],
            "timestamp": self.session_start_iso,
            "duration_seconds": round(time.time() - self.session_start, 2),
            "exit_reason": self.exit_reason,
            "pacing_metadata": {
                "avg_response_time_seconds": avg_resp,
                "total_user_turns": len(self.response_times),
                "slow_pace_count": self.slow_pace_count,
            },
            "errors_encountered": self.errors_encountered,
            "conversation": self.session_log,
            "questionnaire": {},
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
        self._set_state(RobotState.INIT)
        yield self._wamp("rie.dialogue.config.language", lang="en")
        yield self._wamp("rom.optional.behavior.play", name="BlocklyStand")
        yield self.session.subscribe(self.asr_callback, "rie.dialogue.stt.stream")

        self._start_watchdog()

        self._set_state(RobotState.GREETING)
        name = self.profile.get("preferred_name", "")
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
                self._log_turn("user", current_input)
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
                    self.llm.sync_get_response, messages
                )
                if err:
                    self._log_error(err)
                self._log_turn("user", current_input)
                self._log_turn("assistant", speech)
                yield self._speak(speech, gestures=gestures)
                break

            if self._is_unclear(current_input):
                self._log(f"[INPUT] Unclear: '{current_input}'")
                self._set_state(RobotState.CLARIFYING)
                clarification = "Sorry, I missed that. Could you say it again slowly?"
                self._log_turn("user", current_input)
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
                self.llm.sync_get_response, messages
            )
            self._log(f"[LLM] Responded in {time.time()-t0:.1f}s")

            if err:
                self._log_error(err)

            self._log_turn("user", current_input)
            self._log_turn("assistant", speech)

            pace = metadata.get("pace", "normal")
            if metadata.get("pause_before", False):
                self._log("[PACING] LLM requested pause_before")
                yield deferLater(reactor, PAUSE_SLOW_BONUS, lambda: None)

            yield self._speak(speech, gestures=gestures)

            pause = self._calculate_pause(pace)
            yield deferLater(reactor, pause, lambda: None)

            yield self._open_stt()
            self._set_state(RobotState.LISTENING)
            self._user_turn_start = time.time()

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
            self.llm.sync_extract_profile,
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
