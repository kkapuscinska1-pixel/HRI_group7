from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from openai import OpenAI
from time import time
import os
import json
from datetime import datetime

MAX_DURATION = 60 * 5          # max conversation length in seconds
WAMP_REALM = "rie.69f352c326d8af1680827d4a"
TITLE = "Recording_2"
LOG_DIR = "conversations"

# phrases that signal the user wants to stop the conversation
EXIT_PHRASES = ("goodbye", "bye", "quit", "exit",
                "stop", "that's all", "thats all")


# SYSTEM_PROMPT = """
# You are a warm, patient, and encouraging robot assistant helping a Person with
# Aphasia (PwA) in a short introductory conversation. Your goal is to learn a
# few basic facts about the person so that future interactions can be personalised.

# Rules you must follow at all times:
# 1. Keep every response SHORT,  no more than two sentences.
# 2. Ask ONLY ONE question per turn. Never combine multiple questions.
# 3. Use SIMPLE vocabulary and SHORT words. Avoid jargon.
# 4. If the user's answer is unclear or very brief, gently ask ONE follow-up
#    question to clarify, do not assume.
# 5. Be encouraging: acknowledge what the user said before moving on.
# 6. Topics to cover (in order, one at a time):
#      a. The user's first name.
#      b. How the user is feeling today (offer a simple choice if needed).
#      c. A hobby or activity the user enjoys.
#      d. Whether the user has any pets or family members they like to talk about.
# 7. Once all topics are covered, summarise what you learned in two short
#    sentences and say a friendly farewell.
# 8. If the user says something like "goodbye", "bye", or "that's all", wrap up
#    the conversation immediately with a brief farewell message.
# 9. Never mention that you are an AI language model; you are the robot.
# """


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

6. Do not interrupt or finish the user’s sentences.

7. If the user’s answer is unclear, gently confirm what you understood.

8. If the user seems stuck, offer two simple choices.

9. Praise communication effort naturally, but do not overdo it.

10. Avoid complex explanations, jokes based on wordplay, abstract questions, or emotionally sensitive topics.

11. Keep the conversation focused on safe personal topics: name, preferred name, hobbies, music, food, family/pets, daily routines, and preferred communication style.

12. If the user wants to stop, end politely.

13. After around 5 user turns, ask whether they want to continue.

TURN-TAKING BEHAVIOUR

After fixed greeting the user sould tell their name.

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

- the user’s name or preferred name
- topics they enjoy
- activities they like
- possible words or themes that could be useful in future language exercises
- people, pets, or routines they like talking about
- whether they prefer yes/no questions, choice questions, or open questions
- whether repeating, rephrasing, or writing key words may help
- whether they need more time before answering
- any topic they do not want to discuss

"""


# Prompt prefix used when the time limit is reached
PROMPT_CLOSING_TIME = (
    "The conversation time is up. "
    "Please respond briefly to what the user just said, then say a friendly "
    "farewell and end the conversation. User said: "
)

client = OpenAI()
conversation = []          # Full message history
user_input = ""          # Latest transcribed sentence from STT
new_input_ready = False       # Flag: STT has produced a final sentence
robot_speaking = False       # Flag: robot TTS is currently active
last_response_id = None
session_start_iso = None
session_start_time = None
exit_reason = None
saved = False


def asr(frames):
    """
    Called by the robot whenever the STT module produces a result.
    Only act on 'final' sentences
    The robot_speaking flag prevents the robot from reacting to its own voice.
    """
    global user_input, new_input_ready

    if robot_speaking:
        # Ignore everything while the robot is talking to avoid reacting to its own voice
        return

    if frames["data"]["body"]["final"]:
        transcript = str(frames["data"]["body"]["text"]).strip()
        if transcript:
            print(f"[STT] User said: {transcript}")
            user_input = transcript
            new_input_ready = True


def log_turn(role: str, content: str):
    conversation.append({
        "role": role,
        "content": content, })


def save_conversation():
    global saved

    if saved:
        return

    saved = True
    os.makedirs(LOG_DIR, exist_ok=True)

    end_time = time()
    end_iso = datetime.now().isoformat()
    time_marker = end_iso.replace(":", "-")

    duration_seconds = None
    if session_start_time is not None:
        duration_seconds = round(end_time - session_start_time, 2)

    raw_profile = create_personalization_profile()

    try:
        profile = json.loads(raw_profile)
    except json.JSONDecodeError:
        profile = {
            "error": "Profile was not valid JSON",
            "raw_output": raw_profile
        }

    chat_data = {
        "title": TITLE,
        "created_at": session_start_iso,
        "ended_at": end_iso,
        "duration_seconds": duration_seconds,
        "exit_reason": exit_reason,
        "last_response_id": last_response_id,
        "turn_count": len(conversation),
        "content": conversation,
        "personalization_profile": profile,
    }

    filename = f"{LOG_DIR}/{TITLE.lower().replace(' ', '_')}_{time_marker}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2, ensure_ascii=False)

    print(f"[SAVE] Conversation saved to {filename}")


def create_personalization_profile():
    profile_prompt = """
    Analyze the conversation and create a personalization profile.

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
    "future_exercise_suggestions": [],
    "confidence_notes": []
    }

    Rules:
    - Use null if unknown.
    - Use [] if none found.
    - Do not invent facts.
    - Base everything only on the conversation.
    """

    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": profile_prompt},
            {"role": "user", "content": json.dumps(
                conversation, ensure_ascii=False)}
        ]
    )
    return response.output_text.strip()

# LLM response generation


def get_response(user_text: str) -> str:
    global last_response_id

    if last_response_id:
        response = client.responses.create(
            model="gpt-4o",
            previous_response_id=last_response_id,
            input=user_text
        )
    else:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *conversation,
                {"role": "user", "content": user_text}
            ]
        )

    last_response_id = response.id
    reply = response.output_text.strip()

    # Append the new user message to the history
    log_turn("user", user_text)
    log_turn("assistant", reply)
    print(f"[LLM] Robot reply: {reply}")

    return reply

# Main WAMP session


@inlineCallbacks
def main(session, details):
    global user_input, new_input_ready, robot_speaking
    global session_start_iso, session_start_time, exit_reason

    session_start_time = time()
    session_start_iso = datetime.now().isoformat()
    conversation_on = True

    # Robot stands up
    yield session.call("rie.dialogue.config.language", lang="en")

    yield session.call("rom.optional.behavior.play", name="BlocklyStand")

    #  Subscribe the ASR callback to the STT stream
    yield session.subscribe(asr, "rie.dialogue.stt.stream")

    #  Opening greeting and first wave
    greeting = (
        "Hello! I am your robot assistant. "
        "What is your name?"
    )
    robot_speaking = True
    yield session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
    yield session.call("rie.dialogue.say_animated", text=greeting)
    robot_speaking = False

    # Store the greeting in the conversation history
    log_turn("assistant", greeting)
    #  Start listening
    yield session.call("rie.dialogue.stt.stream")

    # Main dialogue loop
    while conversation_on:

        # Wait until STT has produced a new final sentence
        if not new_input_ready:
            yield sleep(0.3)
            continue

        # Grab the user's input and reset the flag
        current_input = user_input
        user_input = ""
        new_input_ready = False

        # Close the STT stream while we process and speak
        # yield session.call("rie.dialogue.stt.close")

        #  Check for exit phrases
        if any(phrase in current_input.lower() for phrase in EXIT_PHRASES):
            exit_reason = "user_exit_phrase"

            farewell = "It was lovely talking to you. Take care and goodbye!"

            log_turn("user", current_input)
            log_turn("assistant", farewell)

            robot_speaking = True
            yield session.call("rie.dialogue.say_animated", text=farewell)
            yield session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
            robot_speaking = False

            conversation_on = False
            break

        # Check if the time limit has been reached
        if time() - session_start_time > MAX_DURATION:
            exit_reason = "time_limit"
            closing_input = PROMPT_CLOSING_TIME + current_input
            reply = get_response(closing_input)
            robot_speaking = True
            yield session.call("rie.dialogue.say_animated", text=reply)
            robot_speaking = False
            conversation_on = False
            break

        # Normal turn: get LLM reply and speak it
        reply = get_response(current_input)
        robot_speaking = True
        yield session.call("rie.dialogue.say_animated", text=reply)
        robot_speaking = False
        # Short pause between robot speech ending and re-opening the microphone
        yield sleep(1.0)

        # Re-open STT stream for the next user turn
        yield session.call("rie.dialogue.stt.stream")

    # Conversation finished: close STT and sit down
    yield session.call("rie.dialogue.stt.close")
    yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")

    # Print the full collected conversation
    print("\n Conversation summary")
    for msg in conversation:
        role = "Robot" if msg["role"] == "assistant" else "User"
        print(f"[{role}] {msg['content']}")
    print("==========================================\n")

    if exit_reason is None:
        exit_reason = "normal_end"

    save_conversation()
    session.leave()

# WAMP component setup


wamp = Component(
    transports=[{
        "url": "ws://wamp.robotsindeklas.nl",
        "serializers": ["msgpack"],
        "max_retries": 0,
    }],
    realm=WAMP_REALM,
)

wamp.on_join(main)

if __name__ == "__main__":
    try:
        run([wamp])
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
        print("\n[STOP] Keyboard interrupt received.")
    finally:
        save_conversation()
