from huggingface_hub import InferenceClient

client = InferenceClient(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    token="XXX",
)

for message in client.chat_completion(
	messages=[{"role": "user", "content": "What is the capital of France?"}],
	max_tokens=500,
	stream=False,
):
    print(message.choices[0].delta.content, end="")
