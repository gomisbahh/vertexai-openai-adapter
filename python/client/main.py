from openai import OpenAI

# Point the client to your vertex ai api-adapter
# The default for LM Studio is "http://localhost:1234/v1"
# An api_key is not required for vertex ai api-adapter.
try:
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

    # Define the conversation and make the API call
    completion = client.chat.completions.create(
      # The 'model' parameter is required, but its value is ignored in this api adapter for vertex ai.
      model="local-model",
      messages=[
        {"role": "system", "content": "You are a helpful coding assistant who provides concise and accurate answers."},
        {"role": "user", "content": "Write a simple 'Hello, World!' function in Python."}
      ],
      temperature=0.7,
    )

    # Print the model's response
    print("AI Response:")
    print(completion.choices[0].message.content)

except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your local LLM server is running and the base_url is correct.")

