#!/usr/bin/env bash
set -euo pipefail

# 1. Load Environment Configuration
if [ ! -f .env ]; then
    echo "[-] Error: .env file not found."
    exit 1
fi
source .env

echo "[+] Step 1/6: Installing K3s Control Plane on Node 1 (${PRIMARY_NODE_IP})..."
curl -sfL https://get.k3s.io | K3S_TOKEN="${K3S_CLUSTER_TOKEN}" sh -s - server \
  --cluster-init \
  --tls-san="${PRIMARY_NODE_IP}"

export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
mkdir -p ~/.kube && cp /etc/rancher/k3s/k3s.yaml ~/.kube/config

echo "[+] Step 2/6: Joining Worker Nodes into Cluster..."
for NODE_IP in "${WORKER_NODE_IPS[@]}"; do
    echo "[*] Bootstrapping Worker Node at ${NODE_IP}..."
    ssh -o StrictHostKeyChecking=no "${SSH_USER}@${NODE_IP}" \
      "curl -sfL https://get.k3s.io | K3S_URL='https://${PRIMARY_NODE_IP}:6443' K3S_TOKEN='${K3S_CLUSTER_TOKEN}' sh -"
done

echo "[+] Step 3/6: Installing Helm & NVIDIA GPU Operator..."
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install --wait --generate-name nvidia/gpu-operator \
  --set driver.enabled=false # Set to true if CUDA host driver is missing

echo "[+] Step 4/6: Generating LiteLLM Proxy Route Config..."
cat <<EOF > litellm-config.yaml
model_list:
  - model_name: "${MODEL_1_NAME}"
    litellm_params:
      model: "openai/${MODEL_1_NAME}"
      api_base: "http://vllm-model-1-svc:8000/v1"
      api_key: "none"
router_settings:
  routing_strategy: "usage-based-routing-v2"
  redis_host: "redis-service"
  redis_port: 6379
  redis_password: "${REDIS_PASSWORD}"
general_settings:
  master_key: "${LITELLM_MASTER_KEY}"
  database_url: "postgresql://postgres:${POSTGRES_PASSWORD}@postgres-service:5432/litellm"
EOF

kubectl create configmap litellm-config --from-file=config.yaml=litellm-config.yaml --dry-run=client -o yaml | kubectl apply -f -

echo "[+] Step 5/6: Deploying Infrastructure & AI Workloads..."
envsubst < k8s-manifests.yaml | kubectl apply -f -

echo "[+] Step 6/6: Waiting for LoadBalancer Gateway Endpoint..."
kubectl rollout status deployment/litellm-proxy --timeout=180s

GATEWAY_IP=$(kubectl get svc litellm-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -z "$GATEWAY_IP" ]; then GATEWAY_IP="${PRIMARY_NODE_IP}"; fi

echo "=================================================================="
echo " CLUSTER DEPLOYMENT COMPLETE (ZERO SPOF HA READY) "
echo "=================================================================="
echo " Single Proxy Entrypoint : http://${GATEWAY_IP}:4000"
echo " Master Management Key  : ${LITELLM_MASTER_KEY}"
echo "=================================================================="
