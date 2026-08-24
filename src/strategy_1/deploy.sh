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


# Run separate unit tests file
echo "=== Running Python Logic Unit Tests ==="
# python -m unittest src/strategy_1/tests

# Execute Python engine
echo "=== Executing Deployment Engine ==="
python -m src.strategy_1.generate_deployment_settings

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
    sudo docker compose up -d --remove-orphans    
fi

if [[ "${DEPLOYMENT_BACKEND:-compose}" != "dstack" ]]; then
    echo "=========================================================="
    echo " Server stack updated and running!"
    echo "=========================================================="
fi
