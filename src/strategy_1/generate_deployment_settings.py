"""
Deployment Orchestrator with inline dependency resolution via uv.
"""

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass

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


@dataclass(frozen=True)
class WorkloadSpec:
    """Normalized model workload shared by deployment backends."""

    model_id: str
    model_name: str
    hf_repo: str
    image: str
    gpu_id: int
    port: int
    max_model_len: int
    replicas: int = 1
    tensor_parallel_size: int = 1
    engine: str = "vllm"


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
    def check_tensor_parallelism(workloads, total_gpus):
        allocated = set()
        for workload in workloads:
            gpu_ids = set(range(workload.gpu_id, workload.gpu_id + workload.tensor_parallel_size))
            if total_gpus > 0 and max(gpu_ids) >= total_gpus:
                raise DetailedValidationError(
                    message=(
                        f"Model '{workload.model_name}' requests "
                        f"TP={workload.tensor_parallel_size} starting at GPU {workload.gpu_id}, "
                        f"but the system has only {total_gpus} GPU(s)."
                    ),
                    location=constants.MODELS_FILE,
                    fix_instructions=(
                        "Set tensor_parallel_size/TENSOR_PARALLEL_SIZE to fit the physical GPUs "
                        "on one server, or choose a valid starting gpu_id."
                    ),
                )
            overlap = allocated.intersection(gpu_ids)
            if overlap:
                raise DetailedValidationError(
                    message=(
                        f"Tensor-parallel GPU allocation overlaps on GPU ID(s): {sorted(overlap)}"
                    ),
                    location=constants.MODELS_FILE,
                    fix_instructions="Assign non-overlapping GPU ranges to active models.",
                )
            allocated.update(gpu_ids)

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

    @staticmethod
    def validate_models(active_models):
        required = ("id", "model_name", "hf_repo", "gpu_id", "port", "max_model_len")
        seen_ids = set()
        for model in active_models:
            missing = [field for field in required if field not in model]
            if missing:
                raise DetailedValidationError(
                    message=(
                        f"Model configuration is missing required field(s): {', '.join(missing)}"
                    ),
                    location=constants.MODELS_FILE,
                    fix_instructions="Add the missing fields to the model entry.",
                )

            model_id = model["id"]
            if not isinstance(model_id, str) or not model_id.strip():
                raise DetailedValidationError(
                    message="Model 'id' must be a non-empty string.",
                    location=constants.MODELS_FILE,
                    fix_instructions="Set a unique, non-empty string for each model 'id'.",
                )
            if model_id in seen_ids:
                raise DetailedValidationError(
                    message=f"Duplicate model ID detected: '{model_id}'.",
                    location=constants.MODELS_FILE,
                    fix_instructions="Give every active model a unique 'id'.",
                )
            seen_ids.add(model_id)

            for field in ("gpu_id", "port", "max_model_len"):
                if not isinstance(model[field], int) or isinstance(model[field], bool):
                    raise DetailedValidationError(
                        message=f"Model '{model_id}' field '{field}' must be an integer.",
                        location=constants.MODELS_FILE,
                        fix_instructions=f"Set '{field}' to a positive integer.",
                    )
                if model[field] < 0 or (field != "gpu_id" and model[field] == 0):
                    raise DetailedValidationError(
                        message=f"Model '{model_id}' field '{field}' has an invalid value.",
                        location=constants.MODELS_FILE,
                        fix_instructions=f"Set '{field}' to a valid positive value.",
                    )
            tensor_parallel_size = model.get("tensor_parallel_size", 1)
            if not isinstance(tensor_parallel_size, int) or isinstance(tensor_parallel_size, bool):
                raise DetailedValidationError(
                    message=f"Model '{model_id}' field 'tensor_parallel_size' must be an integer.",
                    location=constants.MODELS_FILE,
                    fix_instructions="Set 'tensor_parallel_size' to a positive integer.",
                )
            if tensor_parallel_size < 1:
                raise DetailedValidationError(
                    message=(
                        f"Model '{model_id}' field 'tensor_parallel_size' has an invalid value."
                    ),
                    location=constants.MODELS_FILE,
                    fix_instructions="Set 'tensor_parallel_size' to a positive integer.",
                )


class DeploymentPlanner:
    """Build a backend-neutral deployment plan from model configuration."""

    @classmethod
    def build(cls, models, cuda_version="12.0", image_override=None):
        active_models = [model for model in models if model.get("active", False)]
        if not active_models:
            raise DetailedValidationError(
                message="No active models designated for deployment.",
                location=constants.MODELS_FILE,
                fix_instructions="Set 'active': true for at least one model block in models.json.",
            )

        ConfigValidator.validate_models(active_models)
        engine = os.environ.get("INFERENCE_ENGINE", "vllm").lower()
        if engine not in {"vllm", "tgi"}:
            raise DetailedValidationError(
                message=f"Unsupported inference engine: '{engine}'.",
                location=constants.ENV_FILE,
                fix_instructions="Set INFERENCE_ENGINE to 'vllm' or 'tgi'.",
            )
        image = SystemInspector.resolve_image_tag(engine, cuda_version, image_override)
        return tuple(
            WorkloadSpec(
                model_id=model["id"],
                model_name=model["model_name"],
                hf_repo=model["hf_repo"],
                image=model.get("vllm_image") or image,
                gpu_id=model["gpu_id"],
                port=model["port"],
                max_model_len=model["max_model_len"],
                replicas=model.get("replicas", 1),
                tensor_parallel_size=model.get(
                    "tensor_parallel_size", int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
                ),
                engine=engine,
            )
            for model in active_models
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
            return "vllm/vllm-openai:v0.11.0"
        elif ver >= 12.1:
            return "vllm/vllm-openai:v0.11.0"
        elif ver >= 11.8:
            return constants.DEFAULT_VLLM_IMAGE
        else:
            return constants.FALLBACK_VLLM_IMAGE

    @classmethod
    def resolve_image_tag(cls, engine, cuda_version_str, explicit_override=None):
        if engine == "tgi":
            return explicit_override or os.environ.get("TGI_IMAGE", constants.DEFAULT_TGI_IMAGE)
        return cls.resolve_vllm_tag(cuda_version_str, explicit_override)


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
        fallback_model = os.environ.get("LITELLM_FALLBACK_MODEL", "").strip()
        langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "true").lower() == "true"
        context = {
            "active_models": active_models,
            "inference_engine": os.environ.get("INFERENCE_ENGINE", "vllm").lower(),
            "litellm_max_budget": os.environ.get("LITELLM_MAX_BUDGET", "20"),
            "litellm_default_key_max_budget": os.environ.get(
                "LITELLM_DEFAULT_KEY_MAX_BUDGET", "20"
            ),
            "litellm_budget_duration": os.environ.get("LITELLM_BUDGET_DURATION", "1mo"),
            "litellm_fallback_model": fallback_model,
            "litellm_callbacks": [
                callback.strip()
                for callback in os.environ.get(
                    "LITELLM_CALLBACKS", "langfuse" if langfuse_enabled else ""
                ).split(",")
                if callback.strip()
            ],
        }
        return cls._render_file(constants.LITELLM_TEMPLATE_PATH, context)

    @classmethod
    def build_docker_compose(cls, active_models):
        db_user = os.environ["POSTGRES_USER"]
        db_pass = os.environ["POSTGRES_PASSWORD"]
        db_name = os.environ["POSTGRES_DB"]
        langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "true").lower() == "true"
        nginx_enabled = os.environ.get("NGINX_ENABLED", "true").lower() == "true"
        inference_engine = os.environ.get("INFERENCE_ENGINE", "vllm").lower()

        # Dynamically derive DATABASE_URL if not explicitly specified in .env
        db_url = os.environ.get(
            "DATABASE_URL", f"postgresql://{db_user}:{db_pass}@db:5432/{db_name}"
        )

        context = {
            "active_models": active_models,
            "inference_engine": inference_engine,
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
            "langfuse_enabled": langfuse_enabled,
            "langfuse_image": constants.LANGFUSE_IMAGE,
            "clickhouse_image": constants.CLICKHOUSE_IMAGE,
            "redis_image": constants.REDIS_IMAGE,
            "minio_image": constants.MINIO_IMAGE,
            "langfuse_database_url": os.environ.get("LANGFUSE_DATABASE_URL", db_url),
            "langfuse_public_key": os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lh-local"),
            "langfuse_secret_key": os.environ.get("LANGFUSE_SECRET_KEY", "sk-lh-local"),
            "langfuse_nextauth_secret": os.environ.get(
                "LANGFUSE_NEXTAUTH_SECRET", "change-this-nextauth-secret"
            ),
            "langfuse_salt": os.environ.get("LANGFUSE_SALT", "change-this-salt"),
            "langfuse_encryption_key": os.environ.get(
                "LANGFUSE_ENCRYPTION_KEY", "change-this-encryption-key-32-chars"
            ),
            "langfuse_host": os.environ.get("LANGFUSE_HOST", "http://langfuse:3000"),
            "langfuse_minio_access_key": os.environ.get("LANGFUSE_S3_ACCESS_KEY_ID", "langfuse"),
            "langfuse_minio_secret_key": os.environ.get(
                "LANGFUSE_S3_SECRET_ACCESS_KEY", "change-this-minio-secret"
            ),
            "nginx_image": constants.NGINX_IMAGE,
            "api_hostname": os.environ.get("API_HOSTNAME", "api.example.internal"),
            "observability_hostname": os.environ.get(
                "OBSERVABILITY_HOSTNAME", "observability.example.internal"
            ),
            "nginx_tls_certificate": os.environ.get("NGINX_TLS_CERTIFICATE", "./certs/server.crt"),
            "nginx_tls_certificate_key": os.environ.get(
                "NGINX_TLS_CERTIFICATE_KEY", "./certs/server.key"
            ),
            "nginx_enabled": nginx_enabled,
        }
        return cls._render_file(constants.DOCKER_COMPOSE_TEMPLATE_PATH, context)

    @classmethod
    def build_nginx_config(cls):
        context = {
            "api_hostname": os.environ.get("API_HOSTNAME", "api.example.internal"),
            "observability_hostname": os.environ.get(
                "OBSERVABILITY_HOSTNAME", "observability.example.internal"
            ),
            "nginx_tls_certificate": "/etc/nginx/tls/server.crt",
            "nginx_tls_certificate_key": "/etc/nginx/tls/server.key",
            "langfuse_enabled": os.environ.get("LANGFUSE_ENABLED", "true").lower() == "true",
        }
        return cls._render_file(constants.NGINX_TEMPLATE_PATH, context)

    @classmethod
    def build_dstack_service(cls, workload):
        return cls._render_file(constants.DSTACK_TEMPLATE_PATH, {"workload": workload})


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
    backend = os.environ.get("DEPLOYMENT_BACKEND", "compose").lower()
    if backend not in {"compose", "dstack"}:
        raise DetailedValidationError(
            message=f"Unsupported deployment backend: '{backend}'.",
            location=constants.ENV_FILE,
            fix_instructions="Set DEPLOYMENT_BACKEND to 'compose' or 'dstack'.",
        )

    inference_engine = os.environ.get("INFERENCE_ENGINE", "vllm").lower()
    image_override = os.environ.get(
        "TGI_IMAGE" if inference_engine == "tgi" else "VLLM_IMAGE_OVERRIDE"
    )
    workloads = DeploymentPlanner.build(all_models, cuda_version, image_override)
    active_models = [
        {
            "id": workload.model_id,
            "model_name": workload.model_name,
            "hf_repo": workload.hf_repo,
            "image": workload.image,
            "gpu_id": workload.gpu_id,
            "port": workload.port,
            "max_model_len": workload.max_model_len,
            "tensor_parallel_size": workload.tensor_parallel_size,
            "engine": workload.engine,
        }
        for workload in workloads
    ]

    print(f"[System] Physical GPUs detected: {total_gpus}")
    print(f"[System] CUDA version capability: {cuda_version}")

    if backend == "compose":
        ConfigValidator.check_duplicate_ports(active_models)
        ConfigValidator.check_gpu_overload(active_models)
        ConfigValidator.check_tensor_parallelism(workloads, total_gpus)
        if total_gpus > 0:
            ConfigValidator.check_gpu_bounds(active_models, total_gpus)

    print("[Validation] Configuration and network checks passed.")

    if backend == "compose":
        litellm_yaml = ConfigGenerator.build_litellm_yaml(active_models)
        docker_compose_yaml = ConfigGenerator.build_docker_compose(active_models)
        nginx_enabled = os.environ.get("NGINX_ENABLED", "true").lower() == "true"
        nginx_config = ConfigGenerator.build_nginx_config() if nginx_enabled else None

        with open(constants.LITELLM_CONFIG_OUTPUT, "w") as f:
            f.write(litellm_yaml)

        with open(constants.DOCKER_COMPOSE_OUTPUT, "w") as f:
            f.write(docker_compose_yaml)
        if nginx_config is not None:
            with open(constants.NGINX_CONFIG_OUTPUT, "w") as f:
                f.write(nginx_config)
    else:
        os.makedirs(constants.DSTACK_OUTPUT_DIR, exist_ok=True)
        for workload in workloads:
            output_path = os.path.join(
                constants.DSTACK_OUTPUT_DIR, f"{workload.model_id}.dstack.yml"
            )
            with open(output_path, "w") as f:
                f.write(ConfigGenerator.build_dstack_service(workload))

    print(
        "[Generator] Generated "
        + (
            f"{constants.LITELLM_CONFIG_OUTPUT} and {constants.DOCKER_COMPOSE_OUTPUT}"
            f" and {constants.NGINX_CONFIG_OUTPUT}"
            if backend == "compose"
            else f"{constants.DSTACK_OUTPUT_DIR}/*.dstack.yml"
        )
        + " successfully."
    )


if __name__ == "__main__":
    try:
        main()
    except DetailedValidationError as err:
        print(err)
        sys.exit(1)
