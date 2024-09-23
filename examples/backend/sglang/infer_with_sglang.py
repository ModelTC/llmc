import openai

client = openai.Client(
    base_url='http://127.0.0.1:10000/v1', api_key='EMPTY')

# Text completion
response = client.completions.create(
    model='default',
    prompt='The president of the United States is',
    temperature=0,
    max_tokens=32,
)
print(response)
