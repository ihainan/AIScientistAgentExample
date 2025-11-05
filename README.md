# AIScientistAgentExample

A Python streaming LLM service built with FastAPI.

## Configuration

Copy the environment template and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` and fill in your API credentials:

```
SCI_MODEL_BASE_URL=your_model_base_url
SCI_LLM_MODEL=your_llm_model
SCI_EMBEDDING_MODEL=your_embedding_model
SCI_MODEL_API_KEY=your_api_key
```

## Build and Deploy

Build and start the service using Docker Compose:

```bash
docker-compose up --build
```

The service will be available at `http://localhost:3000`.

## Usage

Send a POST request to the `/example` endpoint:

```bash
curl -X POST http://localhost:3000/example \
  -H "Content-Type: application/json" \
  -d '{"query":"Hello"}'
```

The service returns streaming responses in Server-Sent Events format.
