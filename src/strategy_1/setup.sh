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

# Ensure Python 3 exists
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed."
    if confirm_action "Do you want to run 'apt-get update && apt-get install python3 Jinja2'?"; then
        sudo apt-get update && sudo apt-get install -y python3 python3-jinja2
    else
        exit 1
    fi
fi

# Ensure Jinja2 module is available
if ! python3 -c "import jinja2" &> /dev/null; then
    echo "Python package 'jinja2' is missing."
    if confirm_action "Do you want to install python3-jinja2 via apt/pip?"; then
        sudo apt-get install -y python3-jinja2 || pip3 install jinja2
    else
        exit 1
    fi
fi

# Optionally install Docker if missing
if ! command -v docker &> /dev/null; then
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
python3 test_deploy.py

# Execute Python engine
echo "=== Executing Deployment Engine ==="
python3 deploy.py

# Docker Compose execution
echo ""
if confirm_action "Configuration files generated. Launch Docker Compose stack now?"; then
    docker compose up -d --remove-orphans
    echo "=========================================================="
    echo " Server stack updated and running!"
    echo "=========================================================="
else
    echo "Docker deployment skipped. Output files generated."
fi
