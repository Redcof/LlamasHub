# LlamasHub

Deploy multiple open-source LLMs on your own NVIDIA GPU server and access them through a single OpenAI-compatible API.

LlamasHub combines vLLM, LiteLLM, PostgreSQL, and Docker Compose into a simple deployment workflow. Configure your models in `models.json`, provide credentials in `.env`, and generate the complete inference stack with one command.

Deployment dashboards, API endpoints, internal service addresses, ports, capabilities, and access
boundaries are catalogued in [the deployment guide](src/strategy_1/README.md#9-dashboard-and-endpoint-inventory).


## License

LlamasHub is licensed under the Apache License 2.0. Commercial use is permitted,
provided that copyright, license, and attribution notices are preserved.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.


## How to Start the Service

- Docker
- `NVIDIA Driver Version >= 570.133.07` with    `CUDA Version>= 12.8`
- A `huggingface` account with a `API key` e.g. `hf_cDDN....`

### Install
```
sh setup_uv.sh
```


### Configure

**./models.json**
- To activate models set `"activate=true"`. Number of active models dependes on your GPU limitations
- `tensor_parallel_size=x` tells how many CUDA device to use during inferences. Incase you have 2 CUDA devices, you have set it to 2 then you ARE NOT ALLOWED to activate another model.
- `model_name` a name as you wish - we prefer - `<prefix><param-count><context-length>` e.g. `i7B32K` meaning - a model with `7B parameter` and `32000 context length`
- `port=xxxxx` must be `unique`
- `gpu_id=xx` on which CUDA the model loads, must be `unique` as as per your GPU
- `max_model_len=32768` The total context (`Input Prompt + Output`) length must be set as per model and GPU limitations. Smaller context allows more and faster intereaction while restrict agentic capabilities. `32768` is good for coding.
- `hf_repo=...` A qualified HF model

**./templates/litellm_config.yaml.jinja**
- Set `model_list > model_name > litellm_params > max_tokens: 4096` - This is size of `Output` per interaction. `4096` is good for coding.


**.env**

- Copy `cp .env.template .env` update the values carefully
- DO NOT COMMIT
- Use `openssl rand -hex 32` to generate random hex
- A `huggingface` account with a `API key` e.g. `hf_cDDN....`


### 🏃Start 

```
sh src/strategy_1/deploy.sh
```

- A litellm server will start at port 4000. Use `UI_USERNAME`, `UI_PASSWORD` to access it.



## For User

### Cline VSCode Plugin
- API Provider=`LiteLLM`
- Base URL=`http://hostname/v1` ()
- API Key=`LITELLM_MASTER_KEY`
- Refresh Models
- Done

**!!!! Edit !!!! `~/.cline/data/globalState.json`**
```json
// ~/.cline/data/globalState.json
{
    ...
    "actModeLiteLlmModelInfo": {
        "name": "model-name-1",
        "contextWindow": 128000, # <- set as per `models:max_model_len` config e.g. 32000
        "maxTokens": -1,         # <- set as per `litellm:max_tokens` config e.g. 4096
    ...
  }
  ...
}
```
- Restart `VS Code`


