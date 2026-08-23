curl -X POST "http://<PRIMARY_NODE_IP>:4000/key/generate" \
  -H "Authorization: Bearer sk-litellm-master-key-admin-only" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "developer_john",
    "max_budget": 10.00,
    "budget_duration": "30d",
    "rate_limit": 120,
    "models": ["qwen-coder", "llama-3-8b"]
  }'
  