from mlc_llm import MLCEngine

# Create engine
model_path = './dist/llama2-7b-chat-MLC/'
engine = MLCEngine(model_path)

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
    messages=[{'role': 'user', 'content': 'What is the meaning of life?'}],
    model=model_path,
    stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end='', flush=True)
print('\n')

engine.terminate()
