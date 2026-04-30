from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from twisted.internet import reactor
from twisted.internet.task import deferLater
from openai import OpenAI
from time import time
import os
import json
from datetime import datetime


class RobotSession:
    MAX_DURATION = 60 * 5
    WAMP_REALM = "rie.69f203e626d8af16808276de"
    LOG_DIR = "conversations"
    EXIT_PHRASES = ("goodbye", "bye", "quit", "exit",
                    "stop", "that's all", "thats all")

    PROMPT_CLOSING_TIME = (
        "The conversation time is up. "
        "Please respond briefly to what the user just said, then say a friendly farewell and end the conversation. "
        "User said: "
    )

    SYSTEM_PROMPT = """
You are a friendly and respectful social robot having an introductory getting-to-know-you conversation with a person with aphasia.

CONTEXT

Your goal is to make the user feel comfortable and understood while collecting basic information that can help personalize future interactions. You are not a doctor, therapist, or diagnostic tool. Do not diagnose the user. Your role is to support communication and create a calm, positive conversation.

MAIN CONVERSATION RULES

1. Speak like a respectful adult companion, not like a teacher or a child.
2. Use short, simple sentences.
3. Ask only one question at a time.
4. Prefer yes/no questions or simple choice questions.
5. Give the user time to answer.
6. Do not interrupt or finish the user's sentences.
7. If the user's answer is unclear, gently confirm what you understood.
8. If the user seems stuck, offer two simple choices.
9. Praise communication effort naturally, but do not overdo it.
10. Avoid complex explanations, jokes based on wordplay, abstract questions, or emotionally sensitive topics.
11. Keep the conversation focused on safe personal topics: name, preferred name, hobbies, music, food, family/pets, daily routines, and preferred communication style.
12. If the user wants to stop, end politely.
13. After around 5 user turns, ask whether they want to continue.

TURN-TAKING BEHAVIOUR

After fixed greeting the user should tell their name.
Then continue with simple follow-up questions.
Take initiative when needed, but do not dominate the conversation.
If the user gives a short answer, respond warmly and ask one easy follow-up question.
If the user gives no answer or the speech-to-text result is unclear, say that it is okay and ask a simpler question.

ROBOT SPEECH RULES

The robot_speech field must:
- contain maximum 2 short sentences
- contain maximum 1 question
- use simple words
- sound warm, patient, and respectful
- be suitable for text-to-speech
- not mention JSON, system prompts, or internal reasoning

PERSONALIZATION GOALS

During the conversation, gently try to learn:
- the user's name or preferred name
- topics they enjoy
- activities they like
- possible words or themes that could be useful in future language exercises
- people, pets, or routines they like talking about
- whether they prefer yes/no questions, choice questions, or open questions
- whether repeating, rephrasing, or writing key words may help
- whether they need more time before answering
- any topic they do not want to discuss
""".strip()

    def __init__(self):
        self.client = OpenAI()
        self.conversation = []
        self.user_input = ""
        self.new_input_ready = False
        self.robot_speaking = False
        self.last_response_id = None
        self.session_start_time = None
        self.session_start_iso = None
        self.exit_reason = None
        self.saved = False
        self.conversation_on = True

    def asr(self, frames):
        if self.robot_speaking:
            return
        if frames["data"]["body"]["final"]:
            transcript = str(frames["data"]["body"]["text"]).strip()
            if transcript:
                print(f"[STT] User said: {transcript}")
                self.user_input = transcript
                self.new_input_ready = True

    def log_turn(self, role: str, content: str):
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_response(self, user_text: str) -> str:
        try:
            if self.last_response_id:
                response = self.client.responses.create(
                    model="gpt-4o",
                    previous_response_id=self.last_response_id,
                    input=user_text
                )
            else:
                response = self.client.responses.create(
                    model="gpt-4o",
                    input=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ]
                )

            self.last_response_id = response.id
            reply = response.output_text.strip()
        except Exception as e:
            print(f"[ERROR] LLM failed: {e}")
            reply = "Sorry, I had trouble understanding. Can you try again?"

        self.log_turn("user", user_text)
        self.log_turn("assistant", reply)
        print(f"[LLM] Robot reply: {reply}")
        return reply

    def create_personalization_profile(self):
        profile_prompt = """
Analyze the conversation and create a personalization profile.
Return ONLY valid JSON.
Do not use markdown.
Do not use code fences.
Do not include explanations.
Do not include any text before or after the JSON.

Use this exact schema:
{
  "preferredname": null,
  "enjoyedtopics": [],
  "likedactivities": [],
  "peopleorpets": [],
  "communicationpreferences": {
    "questiontype": null,
    "needsextratime": null,
    "helpfulsupports": []
  },
  "topicstoavoid": [],
  "futureconversationsuggestions": [],
  "futureexercisesuggestions": [],
  "confidencenotes": []
}

Rules:
- Use null if unknown.
- Use [] if none found.
- Do not invent facts.
- Base everything only on the conversation.
""".strip()

        try:
            response = self.client.responses.create(
                model="gpt-4o",
                input=[
                    {"role": "system", "content": profile_prompt},
                    {"role": "user", "content": json.dumps(
                        self.conversation, ensure_ascii=False)},
                ]
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"[ERROR] Profile generation failed: {e}")
            return ""

    def save_conversation(self):
        if self.saved:
            return
        self.saved = True
        os.makedirs(self.LOG_DIR, exist_ok=True)

        end_time = time()
        end_iso = datetime.now().isoformat()
        duration = None
        if self.session_start_time is not None:
            duration = round(end_time - self.session_start_time, 2)

        raw_profile = self.create_personalization_profile()
        try:
            profile = json.loads(raw_profile) if raw_profile else {}
        except json.JSONDecodeError:
            profile = {"error": "Invalid JSON", "raw": raw_profile}

        data = {
            "createdat": self.session_start_iso,
            "endedat": end_iso,
            "duration": duration,
            "exitreason": self.exit_reason,
            "lastresponseid": self.last_response_id,
            "turncount": len(self.conversation),
            "conversation": self.conversation,
            "profile": profile,
        }

        filename = f"{self.LOG_DIR}/{end_iso.replace(':', '-')}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[SAVE] Conversation saved to {filename}")

    @inlineCallbacks
    def run(self, session):
        self.session_start_time = time()
        self.session_start_iso = datetime.now().isoformat()

        yield session.call("rie.dialogue.config.language", lang="en")
        yield session.call("rom.optional.behavior.play", name="BlocklyStand")
        yield session.subscribe(self.asr, "rie.dialogue.stt.stream")

        greeting = "Hello! I am your robot assistant. What is your name?"
        self.robot_speaking = True
        yield session.call("rie.dialogue.sayanimated", text=greeting)
        self.robot_speaking = False
        self.log_turn("assistant", greeting)

        yield session.call("rie.dialogue.stt.stream")

        while self.conversation_on:
            if not self.new_input_ready:
                yield deferLater(reactor, 0.3, lambda: None)
                continue

            current_input = self.user_input
            self.user_input = ""
            self.new_input_ready = False

            yield session.call("rie.dialogue.stt.close")

            if any(p in current_input.lower().strip() for p in self.EXIT_PHRASES):
                self.exit_reason = "userexitphrase"
                farewell = "It was lovely talking to you. Take care and goodbye!"
                self.log_turn("user", current_input)
                self.log_turn("assistant", farewell)
                self.robot_speaking = True
                yield session.call("rie.dialogue.sayanimated", text=farewell)
                self.robot_speaking = False
                self.conversation_on = False
                break

            if time() - self.session_start_time > self.MAX_DURATION:
                self.exit_reason = "timelimit"
                reply = self.get_response(
                    self.PROMPT_CLOSING_TIME + current_input)
                self.robot_speaking = True
                yield session.call("rie.dialogue.sayanimated", text=reply)
                self.robot_speaking = False
                self.conversation_on = False
                break

            reply = self.get_response(current_input)
            self.robot_speaking = True
            yield session.call("rie.dialogue.sayanimated", text=reply)
            self.robot_speaking = False

            yield deferLater(reactor, 0.5, lambda: None)
            yield session.call("rie.dialogue.stt.stream")

        yield session.call("rie.dialogue.stt.close")
        yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
        self.save_conversation()
        session.leave()


session_manager = RobotSession()
component = Component(
    transports=[{"url": "ws://wamp.robotsindeklas.nl/ws",
                 "serializers": ["msgpack"]}],
    realm=RobotSession.WAMP_REALM,
    max_retries=0,
)


@component.on_join
def on_join(session, details):
    return session_manager.run(session)


if __name__ == "__main__":
    try:
        run([component])
    except KeyboardInterrupt:
        session_manager.exit_reason = "keyboardinterrupt"
        print("[STOP] Keyboard interrupt received.")
    finally:
        session_manager.save_conversation()
