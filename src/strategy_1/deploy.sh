#!/bin/bash

set -e

echo "=========================================================="
echo " vLLM Deployment Orchestrator"
echo "=========================================================="


# Execute Python engine
echo "=== Generating the configuration ==="
python -m src.strategy_1.generate_deployment_settings .env

# Backend execution
echo "=== Executing Deployment Engine ==="
sudo docker compose down --remove-orphans
sudo docker compose --env-file .env up -d --remove-orphans

