"""
Centralized constants, regexes, and concrete container image tags.
"""

import re

ENV_FILE = ".env"
MODELS_FILE = "models.json"
DOCKER_COMPOSE_OUTPUT = "docker-compose.yml"
LITELLM_CONFIG_OUTPUT = "litellm_config.yaml"
NGINX_CONFIG_OUTPUT = "nginx.conf"
DSTACK_OUTPUT_DIR = "dstack"

# Templates
LITELLM_TEMPLATE_PATH = "templates/litellm_config.yaml.jinja"
DOCKER_COMPOSE_TEMPLATE_PATH = "templates/docker-compose.yml.jinja"
NGINX_TEMPLATE_PATH = "templates/nginx.conf.jinja"
DSTACK_TEMPLATE_PATH = "templates/dstack_service.yaml.jinja"

# Regular Expressions
CUDA_VERSION_REGEX = re.compile(r"^(\d+\.\d+)")

# Fixed Docker Image Tags (Req #5: No :latest images allowed)
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.11.0"
FALLBACK_VLLM_IMAGE = "vllm/vllm-openai:v0.11.0"
DEFAULT_TGI_IMAGE = "ghcr.io/huggingface/text-generation-inference:3.3.5"
LITELLM_IMAGE = "ghcr.io/berriai/litellm:v1.98.0"
POSTGRES_IMAGE = "postgres:15.5-alpine"
LANGFUSE_IMAGE = "langfuse/langfuse:2.95.9"
CLICKHOUSE_IMAGE = "clickhouse/clickhouse-server:24.8.14.39-alpine"
REDIS_IMAGE = "redis:7.4.2-alpine"
MINIO_IMAGE = "minio/minio:RELEASE.2024-12-18T13-15-44Z"
NGINX_IMAGE = "nginx:1.27.3-alpine"
