#!/usr/bin/env bash
set -e

confirm_action() {
    local prompt_message="$1"
    read -r -p "${prompt_message} [y/N]: " response
    case "$response" in
        [yY][eE][sS]|[yY]) return 0 ;;
        *) echo "Operation cancelled by user."; return 1 ;;
    esac
}

echo "=========================================================="
echo " uv Workspace Sync & Environment Setup"
echo "=========================================================="

# 1. Verify uv installation
if command -v uv &> /dev/null; then
    echo "✓ Found uv: $(uv --version)"
else
   echo "Error: 'uv' is not installed."
    echo "Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi



# 2. Check for pyproject.toml
if [ ! -f "pyproject.toml" ]; then
    echo "Error: 'pyproject.toml' not found in root directory."
    exit 1
fi

# 3. Synchronize virtual environment with pyproject.toml
if confirm_action "Do you want to run 'uv sync' to create/update .venv and install all dependencies?"; then
    echo "Syncing workspace dependencies..."
    
    # Creates .venv, generates/updates uv.lock, and installs dependencies
    uv sync
    uv sync --dev
    source .venv/bin/activate
    pre-commit install

    echo ""
    echo "=========================================================="
    echo "✓ Environment successfully synchronized!"
    echo "  Activate: source .venv/bin/activate"
    echo "=========================================================="
else
    echo "Sync cancelled."
fi
