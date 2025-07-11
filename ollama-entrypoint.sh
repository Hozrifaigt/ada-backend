#!/bin/sh
# This script ensures that the necessary models are downloaded before starting the main Ollama server.

/bin/ollama serve &

pid=$!

echo "Ollama server started with PID $pid. Waiting for it to be ready..."
sleep 5

echo "Pulling required models..."

# The models are defined by the environment variables we will set in docker-compose.
# We check if the variable is set before trying to pull.
if [ -n "$LLM_MODEL" ]; then
    echo "Pulling LLM model: $LLM_MODEL"
    ollama pull "$LLM_MODEL"
else
    echo "LLM_MODEL not set, skipping."
fi

if [ -n "$EMBEDDING_MODEL" ]; then
    echo "Pulling embedding model: $EMBEDDING_MODEL"
    ollama pull "$EMBEDDING_MODEL"
else
    echo "EMBEDDING_MODEL not set, skipping."
fi

echo "Model pulling complete."

# This makes it the main process of the container, so Docker can manage it properly.
wait $pid
