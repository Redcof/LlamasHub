# /// script
# dependencies = [
#   "python-dotenv",
#   "jinja2",
# ]
# ///

#!/usr/bin/env python3
"""
Unit test suite for Deployment Orchestrator.
"""

import os
import unittest

from src.strategy_1.deploy import (
    ConfigGenerator,
    ConfigValidator,
    DeploymentPlanner,
    DetailedValidationError,
    EnvValidator,
    SystemInspector,
    WorkloadSpec,
)


class TestDeployEngine(unittest.TestCase):
    def setUp(self):
        """Set up complete os.environ context before each test."""
        os.environ["POSTGRES_USER"] = "unittest_user"
        os.environ["POSTGRES_PASSWORD"] = "unittest_pass"
        os.environ["POSTGRES_DB"] = "unittest_db"
        os.environ["PGDATA_PATH"] = "/tmp/pgdata"
        os.environ["HF_HOME_PATH"] = "/tmp/hf"
        os.environ["LITELLM_MASTER_KEY"] = "sk-test-master-key"
        os.environ["UI_USERNAME"] = "test_admin"
        os.environ["UI_PASSWORD"] = "test_password"

        # Ensure optional overrides don't leak from host env
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        if "VLLM_IMAGE_OVERRIDE" in os.environ:
            del os.environ["VLLM_IMAGE_OVERRIDE"]

    def test_missing_env_vars_raises_error(self):
        """Verify missing mandatory env vars trigger DetailedValidationError."""
        del os.environ["LITELLM_MASTER_KEY"]
        with self.assertRaises(DetailedValidationError) as ctx:
            EnvValidator.validate_or_raise()
        self.assertIn("Missing mandatory environment variables", str(ctx.exception))
        self.assertIn("LITELLM_MASTER_KEY", str(ctx.exception))

    def test_duplicate_ports(self):
        """Verify duplicate port check catches conflicts."""
        bad_models = [
            {"id": "m1", "model_name": "m1", "port": 8000},
            {"id": "m2", "model_name": "m2", "port": 8000},
        ]
        with self.assertRaises(DetailedValidationError):
            ConfigValidator.check_duplicate_ports(bad_models)

    def test_gpu_overload(self):
        """Verify multiple models assigned to same GPU ID triggers error."""
        bad_models = [
            {"id": "m1", "model_name": "m1", "gpu_id": 0},
            {"id": "m2", "model_name": "m2", "gpu_id": 0},
        ]
        with self.assertRaises(DetailedValidationError):
            ConfigValidator.check_gpu_overload(bad_models)

    def test_duplicate_active_ids_raise_error(self):
        models = [
            {
                "id": "same",
                "model_name": "one",
                "hf_repo": "org/one",
                "gpu_id": 0,
                "port": 8000,
                "max_model_len": 4096,
                "active": True,
            },
            {
                "id": "same",
                "model_name": "two",
                "hf_repo": "org/two",
                "gpu_id": 1,
                "port": 8001,
                "max_model_len": 4096,
                "active": True,
            },
        ]
        with self.assertRaisesRegex(DetailedValidationError, "Duplicate model ID"):
            DeploymentPlanner.build(models)

    def test_planner_returns_normalized_workload(self):
        models = [
            {
                "id": "qwen",
                "model_name": "qwen-coder",
                "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "gpu_id": 0,
                "port": 8000,
                "max_model_len": 4096,
                "active": True,
                "replicas": 2,
            }
        ]

        plan = DeploymentPlanner.build(models, cuda_version="12.5")

        self.assertEqual(
            plan,
            (
                WorkloadSpec(
                    model_id="qwen",
                    model_name="qwen-coder",
                    hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
                    image="vllm/vllm-openai:v0.6.3",
                    gpu_id=0,
                    port=8000,
                    max_model_len=4096,
                    replicas=2,
                ),
            ),
        )

    def test_dstack_service_rendering(self):
        workload = WorkloadSpec(
            model_id="qwen",
            model_name="qwen-coder",
            hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
            image="vllm/vllm-openai:v0.6.3",
            gpu_id=0,
            port=8000,
            max_model_len=4096,
        )

        service = ConfigGenerator.build_dstack_service(workload)

        self.assertIn("type: service", service)
        self.assertIn("name: qwen", service)
        self.assertIn("gpu: 1", service)
        self.assertIn("vllm serve", service)
        self.assertIn("/health", service)
        self.assertIn("retry:", service)

    def test_no_latest_tag_in_resolution(self):
        """Verify resolution uses fixed versions and never :latest."""
        tag = SystemInspector.resolve_vllm_tag("12.5")
        self.assertNotIn("latest", tag)
        self.assertEqual(tag, "vllm/vllm-openai:v0.6.3")

    def test_tensor_parallelism_is_configurable_and_bounded(self):
        models = [
            {
                "id": "qwen",
                "model_name": "qwen",
                "hf_repo": "org/qwen",
                "gpu_id": 0,
                "port": 8000,
                "max_model_len": 4096,
                "tensor_parallel_size": 2,
                "active": True,
            }
        ]
        plan = DeploymentPlanner.build(models)
        self.assertEqual(plan[0].tensor_parallel_size, 2)
        with self.assertRaisesRegex(DetailedValidationError, "requests TP=2"):
            ConfigValidator.check_tensor_parallelism(plan, 1)

    def test_litellm_production_controls_render(self):
        os.environ["LITELLM_FALLBACK_MODEL"] = "backup-model"
        config = ConfigGenerator.build_litellm_yaml(
            [
                {
                    "id": "qwen",
                    "model_name": "qwen",
                    "hf_repo": "org/qwen",
                    "port": 8000,
                }
            ]
        )
        self.assertIn("default_key_max_budget: 20", config)
        self.assertIn('"qwen": ["backup-model"]', config)
        self.assertIn("- langfuse", config)

    def test_compose_healthchecks_and_tensor_parallelism_render(self):
        compose = ConfigGenerator.build_docker_compose(
            [
                {
                    "id": "qwen",
                    "model_name": "qwen",
                    "hf_repo": "org/qwen",
                    "gpu_id": 0,
                    "port": 8000,
                    "max_model_len": 4096,
                    "tensor_parallel_size": 2,
                    "image": "vllm/vllm-openai:v0.6.3",
                }
            ]
        )
        self.assertIn("--tensor-parallel-size 2", compose)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", compose)
        self.assertIn("device_ids: ['0', '1']", compose)
        self.assertIn("/health/readiness", compose)
        self.assertIn("http://localhost:8000/health", compose)
        self.assertIn("image: langfuse/langfuse:2.95.9", compose)
        self.assertIn("LANGFUSE_HOST=http://langfuse:3000", compose)

    def test_jinja_rendering_and_env_variables(self):
        """Verify Jinja2 template rendering produces correct credentials and DB URL."""
        models = [
            {
                "id": "qwen",
                "model_name": "qwen2.5-coder-7b",
                "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "gpu_id": 0,
                "port": 8000,
                "max_model_len": 4096,
                "image": "vllm/vllm-openai:v0.6.3",
            }
        ]

        compose_out = ConfigGenerator.build_docker_compose(models)

        # Docker Compose checks
        self.assertNotIn("version: '3.8'", compose_out)
        self.assertNotIn(":latest", compose_out)

        # Environment Variable Rendering
        self.assertIn("POSTGRES_USER: unittest_user", compose_out)
        self.assertIn("pgdata:/tmp/pgdata", compose_out)
        self.assertIn(
            "DATABASE_URL=postgresql://unittest_user:unittest_pass@db:5432/unittest_db",
            compose_out,
        )
        self.assertIn("LITELLM_MASTER_KEY=sk-test-master-key", compose_out)
        self.assertIn("UI_USERNAME=test_admin", compose_out)
        self.assertIn("UI_PASSWORD=test_password", compose_out)

    def test_explicit_database_url_override(self):
        """Verify custom DATABASE_URL in os.environ overrides derived default."""
        os.environ["DATABASE_URL"] = (
            "postgresql://custom_user:custom_pass@external-db:5432/custom_db"
        )

        models = [
            {
                "id": "qwen",
                "model_name": "qwen2.5-coder-7b",
                "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "gpu_id": 0,
                "port": 8000,
                "max_model_len": 4096,
                "image": "vllm/vllm-openai:v0.6.3",
            }
        ]

        compose_out = ConfigGenerator.build_docker_compose(models)
        self.assertIn(
            "DATABASE_URL=postgresql://custom_user:custom_pass@external-db:5432/custom_db",
            compose_out,
        )


if __name__ == "__main__":
    unittest.main()
