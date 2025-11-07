# Medical Chatbot API

Multilingual medical chatbot with FAISS and Gemini LLM supporting English/Japanese.

## Features

- Upload documents (.txt, .pdf, .png, .jpg)
- Multilingual support with auto language detection
- Vector similarity search with FAISS
- AI-powered answer generation
- Cross-language translation

## API Endpoints

- `POST /api/v1/medical/ingest` - Upload documents
- `POST /api/v1/medical/retrieve` - Search similar docs
- `POST /api/v1/medical/generate` - Generate answers

## Quick Start

```bash
docker build -t medical-chatbot .
docker run -p 8000:8000 medical-chatbot
```

## Environment

```
GOOGLE_API_KEY=your_key
```

## Design Notes

**Scalability**: Modular FastAPI architecture enables horizontal scaling through Docker containers. FAISS indices can migrate to distributed storage (Redis/S3) for multi-instance deployments. Service layer abstraction supports async processing with message queues.

**Future Improvements**: Clean separation of endpoints, services, and schemas allows independent scaling. Potential enhancements include vector database migration (Pinecone), streaming responses, authentication, document versioning, and additional language support.