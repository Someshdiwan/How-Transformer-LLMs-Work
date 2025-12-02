#!/bin/bash
uvicorn tokenizer.app.api:app --host 0.0.0.0 --port 8000
