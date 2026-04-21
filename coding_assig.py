from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from openai import OpenAI
import json
from datetime import datetime
from time import time

MAX_DURATION = 60 * 5 # 5 minutes in seconds
TITLE = "Testing on the robot"

client = OpenAI()
last_response_id = None
finish = None

prompt_init = "You are taking on a role of speach therapist..."
prompt_final_time = "This is the last thing the user said, plase respond to it and finish the conversation as the time is up: "
prompt_final_user = "The user said [end_of_conversation_phrase], please finish the conversation with a goodbye message."

starting_conversation = "Hello! I'm your robot assistant. Whats your name?"
history = []


def asr(frames):
    # frames[‘data’][‘body’][‘final’] returns true if the stt
    # considers the end of a turn / sentence is recognised.
    if frames["data"]["body"]["final"]:
        text_heard = frames["data"]["body"]["text"]
        print(text_heard)
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

@inlineCallbacks
def main(session, details):

    start_time = time()
	
	# yield session.call("rom.optional.behavior.play", name="BlocklyStand")
	# yield session.call("rie.dialogue.say", text="Hallo! I'm your robot assistant. Whats your name?")
	# yield sleep(2)
	# yield session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
	# session.leave() # Close the connection with the robot
    
    yield session.call("rom.optional.behavior.play", name="BlocklyStand")
    yield session.subscribe(asr,"rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.say", text = starting_conversation)
    yield session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")

    history.append({"role": "assistant", "content": starting_conversation})

    while True:

        if time() - start_time > MAX_DURATION:
            finish = "time"

        # listen to the user input and process it in the asr function
        PwA_input = "This is a placeholder for the user input"

        if "[end_of_conversation_phrase]" in PwA_input.lower():
            finish = "user"
        
        # get the response from the model using the get_response function
        robot_response = get_response(PwA_input)
        # say the response using the robot
        yield session.call("rie.dialogue.say", text = robot_response)

        # update the history
        history.append({"role": "PwA", "content": PwA_input})
        history.append({"role": "robot", "content": robot_response})

        if finish:
            break

    yield session.call("rie.dialogue.stt.stop")
    session.leave() # Close the connection with the robot

    time_marker =datetime.now().isoformat()
    chat_data = {
        "last_response_id": last_response_id,
        "created_at": time_marker,
        "title": TITLE,
        "content": history
    }

    with open(f"conversations/{TITLE.lower().replace(' ', '_')}_{time_marker}.json", "w") as f:
        json.dump(chat_data, f, indent=2)


wamp = Component(
	transports=[{
		"url": "ws://wamp.robotsindeklas.nl",
		"serializers": ["msgpack"],
		"max_retries": 0
	}],
	realm="rie.69e7409b2c3865c6a7537c05",
)

wamp.on_join(main)

if __name__ == "__main__":
	run([wamp])