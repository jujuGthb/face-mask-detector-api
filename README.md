# Face Mask Detector API

A FastAPI-based face mask detection service using PyTorch.

## Prerequisites

- Docker Desktop installed

## Quick Start

1. Clone the repository:
```bash
   git clone https://github.com/jujuGthb/face-mask-detector-api
   cd face-mask-detector-api
```

2. Run with Docker:
```bash
   docker compose up --build
```

3. Access the API:
   - API: http://localhost:7001
   - Docs: http://localhost:7001/docs

## Stopping the Service

Press `Ctrl+C` or run:
```bash
docker compose down
```

## Note

First build will take 15-20 minutes to download PyTorch dependencies. Subsequent builds will be faster thanks to Docker caching.
