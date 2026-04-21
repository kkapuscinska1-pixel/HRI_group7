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