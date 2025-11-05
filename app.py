import os
import json
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize OpenAI client with custom base URL
client = OpenAI(
    base_url=os.getenv("SCI_MODEL_BASE_URL"),
    api_key=os.getenv("SCI_MODEL_API_KEY")
)

@app.post("/example")
async def example(request: Request):
    """
    Streaming LLM endpoint that accepts a query and returns streaming results
    """
    body = await request.json()
    query = body.get("query", "")

    print(f"Received query: {query}")
    print(f"Using model: {os.getenv('SCI_LLM_MODEL')}")
    print(f"Base URL: {os.getenv('SCI_MODEL_BASE_URL')}")

    async def generate():
        try:
            # Call LLM model with streaming
            stream = client.chat.completions.create(
                model=os.getenv("SCI_LLM_MODEL"),
                messages=[{"role": "user", "content": query}],
                stream=True
            )

            # Process LLM response and stream back
            # Here we directly return without processing
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                if chunk.choices and len(chunk.choices) > 0:
                    delta_content = chunk.choices[0].delta.content
                    if delta_content:
                        response_data = {
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "delta": {
                                    "content": delta_content
                                }
                            }]
                        }
                        yield f"data: {json.dumps(response_data)}\n\n"

            print(f"Total chunks received: {chunk_count}")
            # Send completion signal
            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Error in streaming: {str(e)}")
            error_data = {
                "object": "error",
                "message": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
