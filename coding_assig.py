from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep
from openai import OpenAI

client = OpenAI()

def get_response(PwA: str) -> str:
    response = client.responses.create(
        model="gpt-5.4",
        input=PwA
    )
    return response.output_text

@inlineCallbacks
def main(session, details):
	
	yield session.call("rom.optional.behavior.play", name="BlocklyStand")
	yield session.call("rie.dialogue.say", text="Hallo! I'm your robot assistant. Whats your name?")
	yield sleep(2)
	yield session.call("rom.optional.behavior.play", name="BlocklyWaveRightArm")
	session.leave() # Close the connection with the robot

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