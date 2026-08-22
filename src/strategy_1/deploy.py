"""
Deployment Orchestrator with inline dependency resolution via uv.
"""

import json
import os
import socket
import subprocess
import sys

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

import src.constants as constants

# Load env file directly into os.environ
load_dotenv(dotenv_path=constants.ENV_FILE, override=True)


class DetailedValidationError(Exception):
    """Exception providing detailed context, location, and instructions to fix."""

    def __init__(self, message, location, fix_instructions):
        self.message = message
        self.location = location
        self.fix_instructions = fix_instructions
        super().__init__(self.__str__())

    def __str__(self):
        return (
            f"\n[DEPLOYMENT ERROR]: {self.message}\n"
            f"  -> File/Location: {self.location}\n"
            f"  -> How to Fix   : {self.fix_instructions}\n"
        )


class SystemRunner:
    """Single subprocess wrapper."""

    @staticmethod
    def run_cmd(cmd_args, capture=True, check=True):
        try:
            res = subprocess.run(
                cmd_args,
                stdout=subprocess.PIPE if capture else None,
                stderr=subprocess.PIPE if capture else None,
                text=True,
                check=check,
            )
            return res.stdout.strip() if capture and res.stdout else ""
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(f"Command '{' '.join(cmd_args)}' failed: {e}") from e


class EnvValidator:
    """Strict verification of mandatory environment variables with no default fallbacks."""

    REQUIRED_VARS = [
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_DB",
        "PGDATA_PATH",
        "HF_HOME_PATH",
        "LITELLM_MASTER_KEY",
        "UI_USERNAME",
        "UI_PASSWORD",
    ]

    @classmethod
    def validate_or_raise(cls):
        missing = [var for var in cls.REQUIRED_VARS if not os.environ.get(var)]
        if missing:
            raise DetailedValidationError(
                message=f"Missing mandatory environment variables: {', '.join(missing)}",
                location=constants.ENV_FILE,
                fix_instructions=(
                    f"Edit '{constants.ENV_FILE}' and set non-empty values for: "
                    f"{', '.join(missing)}."
                ),
            )


class ConfigValidator:
    """Configuration validations."""

    @staticmethod
    def check_duplicate_ports(active_models):
        ports = [m["port"] for m in active_models]
        duplicates = set([p for p in ports if ports.count(p) > 1])
        if duplicates:
            raise DetailedValidationError(
                message=f"Duplicate port assignment detected: {duplicates}",
                location=constants.MODELS_FILE,
                fix_instructions=(
                    "Edit models.json and set unique 'port' attributes for each active model."
                ),
            )

    @staticmethod
    def check_gpu_overload(active_models):
        gpus = [m["gpu_id"] for m in active_models]
        duplicates = set([g for g in gpus if gpus.count(g) > 1])
        if duplicates:
            raise DetailedValidationError(
                message=f"GPU Overload: Multiple models assigned to GPU ID(s): {duplicates}",
                location=constants.MODELS_FILE,
                fix_instructions=(
                    "Change the 'gpu_id' values in models.json so that models run on distinct GPUs."
                ),
            )

    @staticmethod
    def check_gpu_bounds(active_models, total_gpus):
        for m in active_models:
            if m["gpu_id"] >= total_gpus:
                raise DetailedValidationError(
                    message=(
                        f"Model '{m['model_name']}' assigned to GPU {m['gpu_id']}, but system "
                        f"has only {total_gpus} GPU(s)."
                    ),
                    location=constants.MODELS_FILE,
                    fix_instructions=(
                        f"Update 'gpu_id' for '{m['model_name']}' in models.json to a value "
                        f"between 0 and {max(0, total_gpus - 1)}."
                    ),
                )

    @staticmethod
    def check_port_availability(active_models):
        for m in active_models:
            port = m["port"]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    raise DetailedValidationError(
                        message=(
                            f"Host port {port} assigned to '{m['model_name']}' is already in "
                            "use by another process."
                        ),
                        location=f"Host Network Port {port}",
                        fix_instructions=(
                            f"Free up port {port} on your system or assign an available port "
                            "in models.json."
                        ),
                    )


class SystemInspector:
    """System introspection."""

    @staticmethod
    def get_gpu_count():
        try:
            out = SystemRunner.run_cmd(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"])
            return int(out.split("\n")[0])
        except Exception:
            return 0

    @staticmethod
    def get_cuda_version():
        try:
            out = SystemRunner.run_cmd(
                ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"]
            )
            return out.split("\n")[0]
        except Exception:
            return "12.0"

    @classmethod
    def resolve_vllm_tag(cls, cuda_version_str, explicit_override=None):
        if explicit_override:
            return explicit_override

        match = constants.CUDA_VERSION_REGEX.match(cuda_version_str)
        if not match:
            return constants.DEFAULT_VLLM_IMAGE

        ver = float(match.group(1))
        if ver >= 12.4:
            return "vllm/vllm-openai:v0.6.3"
        elif ver >= 12.1:
            return "vllm/vllm-openai:v0.6.1"
        elif ver >= 11.8:
            return "vllm/vllm-openai:v0.5.4-cu118"
        else:
            return constants.FALLBACK_VLLM_IMAGE


class ConfigGenerator:
    """Render external Jinja2 template files using values directly from os.environ."""

    @staticmethod
    def _render_file(template_path, context):
        folder, filename = os.path.split(template_path)
        env = Environment(loader=FileSystemLoader(folder or "."))
        template = env.get_template(filename)
        return template.render(context)

    @classmethod
    def build_litellm_yaml(cls, active_models):
        return cls._render_file(constants.LITELLM_TEMPLATE_PATH, {"active_models": active_models})

    @classmethod
    def build_docker_compose(cls, active_models):
        db_user = os.environ["POSTGRES_USER"]
        db_pass = os.environ["POSTGRES_PASSWORD"]
        db_name = os.environ["POSTGRES_DB"]

        # Dynamically derive DATABASE_URL if not explicitly specified in .env
        db_url = os.environ.get(
            "DATABASE_URL", f"postgresql://{db_user}:{db_pass}@db:5432/{db_name}"
        )

        context = {
            "active_models": active_models,
            "postgres_image": constants.POSTGRES_IMAGE,
            "litellm_image": constants.LITELLM_IMAGE,
            "db_user": db_user,
            "db_pass": db_pass,
            "db_name": db_name,
            "db_url": db_url,
            "litellm_master_key": os.environ["LITELLM_MASTER_KEY"],
            "ui_username": os.environ["UI_USERNAME"],
            "ui_password": os.environ["UI_PASSWORD"],
            "pgdata_path": os.environ["PGDATA_PATH"],
            "hf_home_path": os.environ["HF_HOME_PATH"],
        }
        return cls._render_file(constants.DOCKER_COMPOSE_TEMPLATE_PATH, context)


def main():
    print("=== Deployment Orchestrator Running via uv ===")

    # Validate all required environment variables exist
    EnvValidator.validate_or_raise()

    if not os.path.exists(constants.MODELS_FILE):
        raise DetailedValidationError(
            message=f"Configuration file '{constants.MODELS_FILE}' missing.",
            location=constants.MODELS_FILE,
            fix_instructions=(
                "Create a models.json file containing model configurations in the deployment "
                "directory."
            ),
        )

    with open(constants.MODELS_FILE) as f:
        all_models = json.load(f)

    active_models = [m for m in all_models if m.get("active", False)]
    if not active_models:
        raise DetailedValidationError(
            message="No active models designated for deployment.",
            location=constants.MODELS_FILE,
            fix_instructions="Set 'active': true for at least one model block in models.json.",
        )

    total_gpus = SystemInspector.get_gpu_count()
    cuda_version = SystemInspector.get_cuda_version()

    image_override = os.environ.get("VLLM_IMAGE_OVERRIDE")
    default_resolved_tag = SystemInspector.resolve_vllm_tag(cuda_version, image_override)

    for m in active_models:
        m["image"] = m.get("vllm_image") or default_resolved_tag

    print(f"[System] Physical GPUs detected: {total_gpus}")
    print(f"[System] CUDA version capability: {cuda_version}")

    ConfigValidator.check_duplicate_ports(active_models)
    ConfigValidator.check_gpu_overload(active_models)
    if total_gpus > 0:
        ConfigValidator.check_gpu_bounds(active_models, total_gpus)
    ConfigValidator.check_port_availability(active_models)

    print("[Validation] Configuration and network checks passed.")

    litellm_yaml = ConfigGenerator.build_litellm_yaml(active_models)
    docker_compose_yaml = ConfigGenerator.build_docker_compose(active_models)

    with open(constants.LITELLM_CONFIG_OUTPUT, "w") as f:
        f.write(litellm_yaml)

    with open(constants.DOCKER_COMPOSE_OUTPUT, "w") as f:
        f.write(docker_compose_yaml)

    print(
        f"[Generator] Generated {constants.LITELLM_CONFIG_OUTPUT} and "
        f"{constants.DOCKER_COMPOSE_OUTPUT} successfully."
    )


if __name__ == "__main__":
    try:
        main()
    except DetailedValidationError as err:
        print(err)
        sys.exit(1)
