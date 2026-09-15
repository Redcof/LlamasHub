curl http://localhost:4000/v1/chat/completions \
-H "Authorization: Bearer sk-master-6059aca8f6967c7b0f7af17747b7b2b4dd9eaed5e1c8fc907e5fa7503e0e8722" \
-H "Content-Type: application/json" \
-d '{
  "model":"qwen2.5-coder-7b",
  "stream": true,
  "messages":[
    {
      "role":"user",
      "content":"Say hello in one sentence"
    }
  ]
}'