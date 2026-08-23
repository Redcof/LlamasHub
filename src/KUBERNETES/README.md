[ Developer Tools: VSCode / PyCharm / Cloud Code / Graphify ]
                             │
                             ▼
 ┌───────────────────────────────────────────────────────┐
 │               LITELLM HIGH-AVAILABILITY               │
 │           (Load Balancer + Usage Tracking)            │
 └───────────────┬───────────────────────┬───────────────┘
                 │                       │
                 ▼                       ▼
 ┌────────────────────────┐    ┌────────────────────────┐
 │   PostgreSQL Cluster   │    │  Redis HA (Sentinel)   │
 │ (API Keys, Spend, User)│    │(Rate Limits, Caching)  │
 └────────────────────────┘    └────────────────────────┘
                 │                       │
 ┌───────────────┴───────────────────────┴─────────────────┐
 │       KUBERNETES / RAY INFERENCE ROUTER (vLLM)          │
 ├───────────────────┬───────────────────┬─────────────────┤
 │     Node 1        │      Node 2       │    Node 3       │
 │  (2x GPU Servers) │   (2x GPU Servers)│ (2x GPU Servers)│
 └───────────────────┘───────────────────┴─────────────────┘

 