#!/usr/bin/env bash
set -e

# Reusable Confirmation Function
confirm_action() {
    local prompt_message="$1"
    read -r -p "${prompt_message} [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            return 0
            ;;
        *)
            echo "Operation cancelled by user."
            return 1
            ;;
    esac
}

echo "=========================================================="
echo " vLLM Deployment Orchestrator"
echo "=========================================================="


# Optionally install Docker if Compose is selected
if [[ "${DEPLOYMENT_BACKEND:-compose}" != "dstack" ]] && ! command -v docker &> /dev/null; then
    echo "Docker is not detected on this system."
    if confirm_action "Do you want to download and install Docker automatically?"; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        rm get-docker.sh
    else
        echo "Docker is required to continue. Exiting."
        exit 1
    fi
fi

# Run separate unit tests file
echo "=== Running Python Logic Unit Tests ==="
python3 -m unittest src/strategy_1/tests/test_deploy_engine.py

# Execute Python engine
echo "=== Executing Deployment Engine ==="
python3 -m src.strategy_1.deploy

# Backend execution
echo ""
if [[ "${DEPLOYMENT_BACKEND:-compose}" == "dstack" ]]; then
    if ! command -v dstack &> /dev/null; then
        echo "DEPLOYMENT_BACKEND=dstack requires the dstack CLI."
        exit 1
    fi
    if confirm_action "Configuration files generated. Apply dstack services now?"; then
        for service_file in dstack/*.dstack.yml; do
            [[ -e "$service_file" ]] || continue
            dstack apply -f "$service_file"
        done
    else
        echo "dstack deployment skipped. Output files generated."
    fi
else
    if confirm_action "Configuration files generated. Launch Docker Compose stack now?"; then
        docker compose up -d --remove-orphans
    else
        echo "Docker deployment skipped. Output files generated."
    fi
fi

if [[ "${DEPLOYMENT_BACKEND:-compose}" != "dstack" ]]; then
    echo "=========================================================="
    echo " Server stack updated and running!"
    echo "=========================================================="
fi
