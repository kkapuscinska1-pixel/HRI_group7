from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from openai import OpenAI
import json
from datetime import datetime
from time import time
import os

MAX_DURATION = 60 * 1 # in seconds
TITLE = "Testing on the robot"

client = OpenAI()
last_response_id = None
finish = None
global robot_speaking
robot_speaking = False

with open("first_prompt.txt", "r") as f:
    prompt_init = f.read()
print("Initial prompt loaded.", prompt_init)

prompt_final_time = "This is the last thing the user said, plase respond to it and finish the conversation as the time is up: "
prompt_final_user = "The user said [end_of_conversation_phrase], please finish the conversation with a goodbye message."

starting_conversation = "Hello! I'm your robot assistant. Whats your name?"
history = []
global listening
listening = ""

def asr(frames):
    # frames[‘data’][‘body’][‘final’] returns true if the stt
    # considers the end of a turn / sentence is recognised.
    if not robot_speaking:
        if frames["data"]["body"]["final"]:
            text_heard = frames["data"]["body"]["text"]
            print(text_heard)
            listening += text_heard
            #here we have to add some kind of procesing of what the user said and what the robot
            # history.append({"role": "user", "content": user_input})
            # history.append({"role": "assistant", "content": assistant_reply})

def get_response(PwA_input: str) -> str:
    global last_response_id, finish

    # Get first response
    if last_response_id is None:
        response = client.responses.create(
            model = "gpt-5.4",
            input = prompt_init + PwA_input
        )

    # Get response when time is up 
    elif finish == "time":
        response = client.responses.create(
            model = "gpt-5.4",
            input = prompt_final_time + PwA_input
        )
    
    # Get response when user ends the conversation
    elif finish == "user":
         response = client.responses.create(
            model = "gpt-5.4",
            previous_response_id = last_response_id,
            input = prompt_final_user + PwA_input
        )

    # Get response without additional prompt
    else:
        response = client.responses.create(
            model = "gpt-5.4",
            previous_response_id = last_response_id,
            input = PwA_input
        )
    last_response_id = response.id

    return response.output_text

def save_conversation():
    time_marker = datetime.now().isoformat().replace(":", "-")

    chat_data = {
        "last_response_id": last_response_id,
        "created_at": time_marker,
        "title": TITLE,
        "content": history,
        "listening": listening,
    }

    filename = f"conversations/{TITLE.lower().replace(' ', '_')}_{time_marker}.json"

    with open(filename, "w") as f:
        json.dump(chat_data, f, indent=2)

    print(f"Conversation saved to {filename}")

@inlineCallbacks
def main(session, details):
    global finish


    start_time = time()
	
	# yield session.call("rom.optional.behavior.play", name="BlocklyStand")
	# yield session.call("rie.dialogue.say", text="Hallo! I'm your robot assistant. Whats your name?")
	# yield sleep(2)
	# yield session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
	# session.leave() # Close the connection with the robot
    
    
    try:
        yield session.call("rom.optional.behavior.play", name="BlocklyStand")
        yield session.subscribe(asr, "rie.dialogue.stt.stream")
        yield session.call("rie.dialogue.stt.stream")
        robot_speaking = True
        yield session.call("rie.dialogue.say", text=starting_conversation)
        listening = ""
        robot_speaking = False
        yield session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")

        history.append({"role": "assistant", "content": starting_conversation})

        while True:

            if time() - start_time > MAX_DURATION:
                finish = "time"

            PwA_input = listening

            if "[end_of_conversation_phrase]" in PwA_input.lower():
                finish = "user"

            robot_response = get_response(PwA_input)

            robot_speaking = True
            yield session.call("rie.dialogue.say", text=robot_response)
            listening = ""
            robot_speaking = False

            history.append({"role": "PwA", "content": PwA_input})
            history.append({"role": "robot", "content": robot_response})

            if finish:
                break

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Saving before exit...")
        finish = "keyboard_interrupt"

    finally:
        try:
            yield session.call("rie.dialogue.stt.stop")
        except Exception as e:
            print("Could not stop STT:", e)

        save_conversation()

        try:
            session.leave()
        except Exception as e:
            print("Could not leave session:", e)


wamp = Component(
	transports=[{
		"url": "ws://wamp.robotsindeklas.nl",
		"serializers": ["msgpack"],
		"max_retries": 0
	}],
	realm="rie.69f203e626d8af16808276de",
)

wamp.on_join(main)

if __name__ == "__main__":
	run([wamp])