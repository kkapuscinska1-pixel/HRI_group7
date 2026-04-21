from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-5.4",
    input="Explain this Python function in simple words."
)

print(response.output_text)