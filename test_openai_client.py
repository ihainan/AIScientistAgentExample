#!/usr/bin/env python3
"""
Test script to verify OpenAI SDK compatibility with the custom endpoint
"""
from openai import OpenAI

# Initialize OpenAI client pointing to our custom endpoint
client = OpenAI(
    base_url="http://localhost:33000/v1",
    api_key="dummy-key"  # Not validated as per requirements
)

def test_streaming():
    """Test streaming chat completion"""
    print("Testing streaming mode...")
    print("-" * 50)

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Model name not validated as per requirements
        messages=[
            {"role": "user", "content": "Say 'Hello, World!' and explain what it means."}
        ],
        stream=True
    )

    print("Response (streaming):")
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

    print("\n" + "-" * 50)
    print("Streaming test completed!\n")

def test_non_streaming():
    """Test non-streaming chat completion"""
    print("Testing non-streaming mode...")
    print("-" * 50)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        stream=False
    )

    print("Response (non-streaming):")
    print(response.choices[0].message.content)
    print("-" * 50)
    print("Non-streaming test completed!\n")

if __name__ == "__main__":
    print("Starting OpenAI SDK compatibility tests...\n")

    try:
        test_streaming()
        test_non_streaming()
        print("All tests passed!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
