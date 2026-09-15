"""Command-line interface for configuring and running a LlamasHub deployment."""

import argparse
import getpass
import re
import subprocess
from pathlib import Path

from dotenv import set_key

from src.strategy_1.generate_deployment_settings import main as generate_deployment

KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SETTING_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=")
SECRET_PARTS = ("PASSWORD", "KEY", "SECRET", "TOKEN")
ENV_DIR = Path("/etc/llamashub")


def _env_path(environment):
    """Get the environment file path for the specified environment (deploy or stage)."""
    return ENV_DIR / f"{environment}.env"


def _required_settings(template):
    settings = []
    required_marker = False
    for line in template.read_text().splitlines():
        stripped = line.strip()
        if stripped.lower() == "# required":
            required_marker = True
            continue
        match = SETTING_PATTERN.match(stripped)
        if not match:
            continue
        if required_marker:
            key = match.group(1)
            label = key.replace("_", " ").title()
            secret = any(part in key for part in SECRET_PARTS)
            settings.append((key, label, secret))
        required_marker = False
    return settings


def init_env(environment):
    """Interactively create an environment file with the deployment settings."""
    path = _env_path(environment)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text().strip():
        response = input(f"{path} already exists. Overwrite it? [y/N]: ")
        if response.strip().lower() not in {"y", "yes"}:
            print("Initialization cancelled.")
            return

    template = Path(".env.template")
    if template.exists():
        path.write_text(template.read_text())
    else:
        path.touch()

    for key, prompt, secret in _required_settings(template):
        while True:
            value = getpass.getpass(f"{prompt}: ") if secret else input(f"{prompt}: ")
            if value.strip():
                set_key(str(path), key, value, quote_mode="auto")
                break
            print("A value is required.")
    print(f"Created {path}")


def update_config(environment, key, value):
    if not KEY_PATTERN.fullmatch(key):
        raise SystemExit("Configuration keys must contain only letters, numbers, and underscores.")
    path = _env_path(environment)
    path.parent.mkdir(parents=True, exist_ok=True)
    set_key(str(path), key, value, quote_mode="auto")
    print(f"Updated {key} in {path}")


def start_services(environment):
    """Generate artifacts and start the systemd service."""
    path = _env_path(environment)
    generate_deployment(str(path))
    subprocess.run(
        ["sudo", "systemctl", "start", f"llamashub@{environment}.service"],
        check=True,
    )


def build_parser():
    parser = argparse.ArgumentParser(prog="llamashub")

    # Required environment choice argument
    subparsers = parser.add_subparsers(
        dest="environment", required=True, help="Deployment environment"
    )
    deploy_parser = subparsers.add_parser(
        "deploy", help="Configure or manage the deploy environment"
    )
    stage_parser = subparsers.add_parser("stage", help="Configure or manage the stage environment")

    # Add commands for each environment
    for env_parser in [deploy_parser, stage_parser]:
        env_subparsers = env_parser.add_subparsers(dest="command")
        env_subparsers.add_parser("init", help="Create an environment file interactively")
        env_subparsers.add_parser("start", help="Generate artifacts and start the system service")
        config_parser = env_subparsers.add_parser(
            "config", help="Set a key/value in the environment file"
        )
        config_parser.add_argument("key", help="Configuration key")
        config_parser.add_argument("value", help="Configuration value")

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "init":
        init_env(args.environment)
    elif args.command == "start":
        start_services(args.environment)
    elif args.command == "config":
        update_config(args.environment, args.key, args.value)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
