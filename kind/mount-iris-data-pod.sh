#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_PATH="$ROOT_DIR/data/iris.csv"
NAMESPACE="kubeflow"

if [ ! -f "$CSV_PATH" ]; then
  echo "Fichier introuvable : $CSV_PATH"
  exit 1
fi

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f kind/iris-data-pvc.yaml

POD_NAME="iris-data-loader"
kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --ignore-not-found=true

cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
spec:
  securityContext:
    runAsNonRoot: true
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: iris-data-container
      image: busybox:1.36
      command: ["sh", "-c", "sleep 3600"]
      securityContext:
        allowPrivilegeEscalation: false
        runAsNonRoot: true
        capabilities:
          drop: ["ALL"]
      volumeMounts:
        - name: iris-data-volume
          mountPath: /data
          readOnly: false
  volumes:
    - name: iris-data-volume
      persistentVolumeClaim:
        claimName: iris-data-pvc
EOF

kubectl cp "$CSV_PATH" "$NAMESPACE/$POD_NAME:/data/iris.csv"

kubectl exec -n "$NAMESPACE" "$POD_NAME" -- sh -c "ls -l /data && head -n 5 /data/iris.csv"

echo ""
echo "Le dataset est maintenant monté dans /data/iris.csv sur le pod iris-data-loader"
echo "Tu peux ensuite pointer le pipeline Kubeflow sur /data/iris.csv"
