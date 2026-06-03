from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
import json


with open("test.json", "r") as f:
    LLM_answer = json.load(f)

@inlineCallbacks
def main(session, details):
	yield session.call("rom.optional.behavior.play", name="BlocklyStand")

	gesture = session.call(
		"rom.optional.behavior.play",
		name=LLM_answer["gesture"]
	)

	yield session.call(
		"rie.dialogue.say",
		text=LLM_answer["text"],
		lang="en"
	)

	yield gesture

	yield session.call("rom.optional.behavior.play", name="BlocklyCrouch")
	session.leave() # Close the connection with the robot

wamp = Component(
	transports=[{
		"url": "ws://wamp.robotsindeklas.nl",
		"serializers": ["msgpack"],
		"max_retries": 0
	}],
	realm="rie.6a1d57f88a2cba4f82b84492",
)

wamp.on_join(main)

if __name__ == "__main__":
	run([wamp])