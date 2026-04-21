from autobahn.twisted.component import Component, run
from twisted.internet.defer import inlineCallbacks
from autobahn.twisted.util import sleep


# The function asr(frames) is used in combination with subscribe
def asr(frames):
    # frames[‘data’][‘body’][‘final’] returns true if the stt
    # considers the end of a turn / sentence is recognised.
    if frames["data"]["body"]["final"]:
        text_heard = frames["data"]["body"]["text"]
        print(text_heard)

# The robot asks the user to say something and prints what it has heard

@inlineCallbacks
def main(session, details):
    yield sleep(1)
    info = yield session.call("rom.sensor.hearing.info")
    # change language to English
    yield session.call("rie.dialogue.config.language", lang="en")
    yield session.call("rie.dialogue.say", text="Say something")
    # prompt from the robot to the user to say something
    # listen for 6 sec to the speech input and recognize it
    data = yield session.call("rie.dialogue.stt.read", time=6000)
    for frame in data:
        if (frame["data"]["body"]["final"]):
            print(frame)
    # example when using subscribing and calling STT stream:
    yield session.subscribe(asr,"rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.stt.stream")
    yield session.call("rie.dialogue.say", text="Now you can talk to the robot")

    yield sleep(10)  # wait for 10 seconds to give the user time to say something
    # you still need to write a statement to end the program…
    session.leave()


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