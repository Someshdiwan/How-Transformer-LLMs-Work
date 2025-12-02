#!/bin/bash
uvicorn tokenizer.app.api:app --host 0.0.0.0 --port 8000
#!/bin/bash

PORT=${PORT:-8000}

uvicorn tokenizer.app.api:app \
    --host 0.0.0.0 \
    --port $PORT
    