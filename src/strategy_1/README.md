
# vLLM Multi-Model Deployment Orchestrator

This project generates deployment artifacts for multiple vLLM inference servers behind a LiteLLM proxy. Docker Compose remains the local fallback, and dstack can provision model services with scheduler-managed retries and health probes. Model, GPU, port, credential, and image settings are kept in configuration files rather than hard-coded in the deployment command.

The Python tooling is installed into a local virtual environment managed by [`uv`](https://docs.astral.sh/uv/). Docker runs the actual inference, proxy, and database services.

## 1. System Architecture

```text
                                                               [Host machine]
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ Docker Compose                                                           │
 │                                                                          │
 │  ┌──────────────┐       ┌─────────────────┐       ┌──────────────────┐   │
 │  │ vLLM model 1 │──────▶│                 │──────▶│                  │   │
 │  │ GPU N / port │       │ LiteLLM proxy   │       │ PostgreSQL       │   │
 │  ├──────────────┤       │ port 4000       │       │ port 5432        │   │
 │  │ vLLM model 2 │──────▶│                 │       │                  │   │
 │  │ GPU M / port │       └────────┬────────┘       │                  │   │
 │  └──────────────┘                │                └────────┬─────────┘   │
 └──────────────────────────────────┼─────────────────────────┼────────────-┘
                                                                        │                         │
                                                 OpenAI-compatible API        Named volume: pgdata
                                                 http://HOST:4000/v1
```

- Each active model becomes a `vllm-<id>` service. LiteLLM reaches it by its Compose service name, for example `http://vllm-qwen-7b:8000/v1`.
- LiteLLM is the public API and is the only service published on host port `4000` by the template.
- PostgreSQL is reachable inside the Compose network as `db:5432`.
- A model uses one GPU and one host port. The validator rejects duplicate GPU IDs and duplicate ports.

## 2. Repository and Filesystem Structure

```text
.
├── .env                         # Local secrets and host paths; ignored by Git
├── .env.template                # Safe starting point for .env
├── models.json                  # Active model and GPU definitions
├── pyproject.toml               # Project metadata, uv dependencies, Ruff settings
├── requirements.txt             # Alternative pip dependency list
├── requirements-dev.txt         # Development tools for the pip workflow
├── setup_uv.sh                  # Interactive uv environment setup
├── templates/
│   ├── docker-compose.jinja.yaml # Compose template
│   └── litellm_config.jinja.yaml # LiteLLM model-list template
├── src/
│   ├── constants.py              # File names, image tags, and CUDA matching
│   └── strategy_1/
│       ├── deploy.py              # Validation, image resolution, and generation
│       ├── setup.sh               # Legacy interactive setup/launch helper
│       └── tests/test_deploy_engine.py
└── vs_code_intergation/README.md # VS Code model configuration example
```

Generated in the project root by the deployment engine:

```text
docker-compose.yml       # Generated Compose stack
litellm_config.yaml      # Generated LiteLLM model list
dstack/<model>.dstack.yml # Generated dstack model services when selected
```

These generated files are deployment artifacts. Re-run the generator after changing `.env`, `models.json`, or a template.

## 3. What Is Downloaded and Saved

### During local setup

`uv sync` creates `.venv/`, resolves dependencies from `pyproject.toml`, and may create or update `uv.lock`. The application needs Jinja2 and `python-dotenv`; development checks use Ruff and pytest.

### During Docker deployment

Docker pulls the fixed image tags selected by the generator:

- vLLM: normally `vllm/vllm-openai:v0.6.3`, selected from the detected CUDA version
- LiteLLM: `ghcr.io/berriai/litellm:v1.52.0`
- PostgreSQL: `postgres:15.5-alpine`

The vLLM containers download each configured Hugging Face repository on first use and cache it at the host path in `HF_HOME_PATH`, mounted as `/root/.cache/huggingface` in each vLLM container. Image selection can be overridden with `VLLM_IMAGE_OVERRIDE`, and an individual model can set `vllm_image`.

### Persistent and generated data

- PostgreSQL data is stored in the named Docker volume `pgdata`, mounted at `PGDATA_PATH` in the database container.
- `docker-compose.yml` and `litellm_config.yaml` are written to the repository root.
- `.env` contains credentials and paths and is excluded by `.gitignore`. Treat it as a secret.
- No model weights are stored in this repository unless `HF_HOME_PATH` is deliberately pointed into it.

## 4. Configure `.env`

Create the local environment file from the template:

```bash
cp .env.template .env
```

Set every required value. Empty values fail validation before any files are generated.

| Variable | Purpose |
| --- | --- |
| `POSTGRES_USER` | Database username |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_DB` | Database name |
| `PGDATA_PATH` | Host directory for PostgreSQL data |
| `HF_HOME_PATH` | Host directory for Hugging Face model cache |
| `LITELLM_MASTER_KEY` | LiteLLM master/admin key |
| `UI_USERNAME` | LiteLLM UI username |
| `UI_PASSWORD` | LiteLLM UI password |

Optional values:

- `DATABASE_URL` overrides the derived URL. Without it, the generator uses `postgresql://<user>:<password>@db:5432/<database>`.
- `VLLM_IMAGE_OVERRIDE` forces one vLLM image for models that do not specify `vllm_image`.
- `TENSOR_PARALLEL_SIZE` sets the default vLLM tensor parallel size. The engine validates each
    model's GPU range against the detected physical GPU count and rejects overlapping ranges.
- `LITELLM_MAX_BUDGET` and `LITELLM_DEFAULT_KEY_MAX_BUDGET` default to `$20`; set
    `LITELLM_BUDGET_DURATION` to control the reset period.
- `LITELLM_FALLBACK_MODEL` names a configured LiteLLM model for fallback routing.
- `LITELLM_CALLBACKS` defaults to `langfuse`. Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
    and optionally `LANGFUSE_HOST` to record requests, inputs, and outputs in Langfuse. Compose
    defaults this host to the self-hosted `langfuse` service on the internal network.
- Set `LANGFUSE_ENABLED=false` to omit the self-hosted Langfuse services and disable the default
    callback. The generated Compose stack includes Langfuse, ClickHouse, Redis, and MinIO with
    telemetry to the public cloud disabled.

Use absolute, writable host paths for `PGDATA_PATH` and `HF_HOME_PATH`. Do not commit real credentials or API keys.

## 5. Configure `models.json`

`models.json` is a JSON array. Each model entry supports:

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Unique Compose-safe identifier used in `vllm-<id>` |
| `model_name` | yes | Name exposed through LiteLLM |
| `hf_repo` | yes | Hugging Face repository to download |
| `gpu_id` | yes | Numeric GPU assigned to the model |
| `port` | yes | Internal vLLM port and host port used for validation |
| `max_model_len` | yes | vLLM maximum context length |
| `active` | yes | Set to `true` to generate/deploy the model |
| `vllm_image` | no | Per-model vLLM image override |
| `tensor_parallel_size` | no | Number of GPUs used by this model; defaults to `TENSOR_PARALLEL_SIZE` |

Example:

```json
[
    {
        "id": "qwen-7b",
        "model_name": "qwen2.5-coder-7b",
        "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "gpu_id": 0,
        "port": 8000,
        "max_model_len": 4096,
        "active": true
    }
]
```

Only active entries are rendered. Give active models distinct GPU IDs and ports. The engine checks duplicate ports, duplicate GPU assignments, port availability, and GPU bounds when NVIDIA GPUs are detected. At least one model must be active.

## 6. Configure YAML Templates

The files in `templates/` are Jinja2 inputs, not files that must normally be edited after every deployment:

- `templates/docker-compose.jinja.yaml` defines the database, vLLM services, LiteLLM service, mounts, GPU reservations, and dependencies.
- `templates/litellm_config.jinja.yaml` creates one LiteLLM model entry for every active model.

The self-hosted Langfuse services use pinned container images. Mirror or preload those images in
the air-gapped registry before running the generated Compose stack; deployment cannot pull images
from the public registry after network isolation.

The generator supplies environment values and the active model list to these templates. If you customize a template, preserve the variable names used by `src/strategy_1/deploy.py`, then regenerate and inspect both output files before starting Docker.

## 7. Setup

Prerequisites:

- Linux host with Docker Compose v2 and an NVIDIA driver/container runtime for GPU inference
- Python 3.6 or newer; Python 3.10+ is recommended for current tooling
- `uv` installed and available on `PATH`
- Network access to the container registry and Hugging Face

From the repository root:

```bash
chmod +x setup_uv.sh
./setup_uv.sh
```

The script asks before running `uv sync`. The equivalent non-interactive command is:

```bash
uv sync
```

Then configure `.env` and `models.json` as described above. Run the tests before deployment:

```bash
uv run pytest src/strategy_1/tests/test_deploy_engine.py
```

## 8. Generate and Deploy

Always run commands from the repository root because the application resolves `.env`, `models.json`, and `templates/` using repository-relative paths.

Generate the deployment files:

```bash
uv run python -m src.strategy_1.deploy
```

The command validates the environment and model configuration, resolves the vLLM image, and writes Compose artifacts by default. Review the generated files, then start the stack:

```bash
docker compose up -d --remove-orphans
docker compose ps
docker compose logs -f litellm
```

To generate dstack services instead, set `DEPLOYMENT_BACKEND=dstack` in `.env` or the shell environment:

```bash
DEPLOYMENT_BACKEND=dstack uv run python -m src.strategy_1.deploy
dstack apply -f dstack/qwen-7b.dstack.yml
```

Each dstack service has a vLLM `/health` probe and retries capacity, runtime, and interruption failures. The current POC does not make LiteLLM, PostgreSQL, the dstack server, or the public gateway highly available; use a stable or managed deployment for those components.

The API is available at `http://<HOST>:4000/v1`. Use the model names from `models.json` as the API `model` value. The VS Code configuration example is in `vs_code_intergation/README.md`; replace `<YOUR-SERVER-IP>` and use a valid LiteLLM virtual key.

To stop the services while retaining PostgreSQL data:

```bash
docker compose down
```

To remove the stack and its database volume, which permanently deletes PostgreSQL data:

```bash
docker compose down -v
```

## 9. Updating the Deployment

1. Stop or leave the current stack running while editing `.env` or `models.json`.
2. Run the tests and regenerate the YAML files.
3. Inspect the model list, image tags, ports, mounts, and credentials in the generated files.
4. Apply the changes with `docker compose up -d --remove-orphans`.

If a port, GPU, or environment value is invalid, the generator exits with a detailed error and does not intentionally start Docker. Check `docker compose logs <service>` for runtime issues such as image pulls, CUDA compatibility, permissions on host paths, or Hugging Face access.

## 10. Security and Operational Notes

- Replace every sample secret in `.env` before exposing the API.
- Restrict access to host port `4000` with the host firewall or a reverse proxy/TLS layer.
- Keep `.env`, generated credentials, and model cache permissions limited to the deployment operator.
- Pin image tags as provided; avoid changing them to `:latest` in production.
- Ensure each model fits the assigned GPU. The validator prevents GPU sharing but cannot measure model memory requirements.

## 11. License and Attribution

This project is licensed under the Apache License 2.0. Commercial use is permitted,
provided that copyright, license, and attribution notices are preserved.

See the repository [LICENSE](../../LICENSE) and [NOTICE](../../NOTICE) files for details.

