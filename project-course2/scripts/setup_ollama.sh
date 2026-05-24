#!/bin/bash
# Yakira_S - IR Final Assignment - Ollama Setup Script
# Run this ONCE on your GPU PC before starting Experiment D

echo "=== Setting up Ollama + llama3 ==="

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the llama3 model (~4.7GB)
ollama pull llama3

echo "=== Setup complete. Test with: ollama run llama3 'Hello!' ==="
