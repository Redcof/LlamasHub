# LlamasHub

Deploy multiple open-source LLMs on your own NVIDIA GPU server and access them through a single OpenAI-compatible API.

LlamasHub combines vLLM, LiteLLM, PostgreSQL, and Docker Compose into a simple deployment workflow. Configure your models in `models.json`, provide credentials in `.env`, and generate the complete inference stack with one command.

## Install and run

Install the command-line tool from PyPI:

```bash
python3 -m pip install llamashub
llamashub deploy init
```

`llamashub deploy init` interactively creates `/etc/llamashub/deploy.env`. For a staging environment, use:

```bash
llamashub stage init
```

Update any setting later with:

```bash
llamashub deploy config KEY VALUE
llamashub stage config KEY VALUE
```

Install `systemd/llamashub@.service` as `/etc/systemd/system/llamashub@.service`, then start the deployment with:

```bash
sudo systemctl daemon-reload
sudo systemctl start llamashub@deploy
journalctl -u llamashub@deploy -f
```

Or for staging:

```bash
sudo systemctl start llamashub@stage
journalctl -u llamashub@stage -f
```

The `llamashub deploy start` or `llamashub stage start` command also starts the systemd service.
The service keeps the generated Docker Compose stack running after deployment.
Generator and Docker Compose output is collected by `journald`; stop the stack with
`sudo systemctl stop llamashub@deploy` or `sudo systemctl stop llamashub@stage`.

Deployment dashboards, API endpoints, internal service addresses, ports, capabilities, and access
boundaries are catalogued in [the deployment guide](src/strategy_1/README.md#9-dashboard-and-endpoint-inventory).

## License

LlamasHub is licensed under the Apache License 2.0. Commercial use is permitted,
provided that copyright, license, and attribution notices are preserved.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
