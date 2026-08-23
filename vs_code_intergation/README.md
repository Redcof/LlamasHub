```json
{
  "models": [
    {
      "title": "CodeLlama 7B (In-House)",
      "provider": "openai",
      "model": "codellama-7b",
      "apiKey": "sk-user-specific-virtual-key",
      "apiBase": "https://api.example.internal/v1"
    },
    {
      "title": "Qwen 2.5 Coder (In-House)",
      "provider": "openai",
      "model": "qwen2.5-coder-7b",
      "apiKey": "sk-user-specific-virtual-key",
      "apiBase": "https://api.example.internal/v1"
    }
  ]
}
```

Replace `api.example.internal` with `API_HOSTNAME`. Each developer needs a LiteLLM virtual key
created by an administrator; do not use `LITELLM_MASTER_KEY` here. The model values must match
`model_name` in `models.json`.

```bash
curl --cacert internal-ca.crt https://api.example.internal/v1/models \
  -H "Authorization: Bearer ${LITELLM_VIRTUAL_KEY}"
```
