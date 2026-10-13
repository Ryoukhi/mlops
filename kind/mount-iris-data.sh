#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_PATH="$ROOT_DIR/data/iris.csv"
NAMESPACE="kubeflow"
CONFIGMAP_NAME="iris-data"
POD_NAME="iris-debug"

if [ ! -f "$CSV_PATH" ]; then
  echo "Fichier introuvable : $CSV_PATH"
  exit 1
fi

echo "[1/4] Vérification du namespace Kubeflow..."
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || {
  echo "Namespace $NAMESPACE absent. Vérifie l'installation de Kubeflow."
  exit 1
}

echo "[2/4] Suppression du ConfigMap existant..."
kubectl delete configmap "$CONFIGMAP_NAME" -n "$NAMESPACE" --ignore-not-found=true

echo "[3/4] Création du ConfigMap avec iris.csv..."
kubectl create configmap "$CONFIGMAP_NAME" -n "$NAMESPACE" --from-file=iris.csv="$CSV_PATH"

echo "[4/4] Création d'un pod de test avec montage /data..."
kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --ignore-not-found=true || true

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
    - name: debug-iris
      image: busybox:1.36
      command: ["sh", "-c", "sleep 3600"]
      securityContext:
        allowPrivilegeEscalation: false
        runAsNonRoot: true
        capabilities:
          drop: ["ALL"]
      volumeMounts:
        - name: iris-volume
          mountPath: /data
  volumes:
    - name: iris-volume
      configMap:
        name: ${CONFIGMAP_NAME}
EOF

kubectl wait --for=condition=Ready pod/"$POD_NAME" -n "$NAMESPACE" --timeout=120s

echo "[4/4] Vérification du montage..."
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- sh -c "ls -l /data && echo '---' && head -n 5 /data/iris.csv"

echo ""
echo "Le dataset est maintenant visible dans /data/iris.csv dans le pod de test."
echo "Important : dans Kubeflow pipeline, les composants sont des conteneurs distincts, donc il faut un PVC monté via KFP (mount_pvc) pour que /data soit disponible dans chaque étape du pipeline."
echo "Exemple de montage KFP :"
echo "   from kfp.kubernetes import mount_pvc"
echo "   task = mount_pvc(task, pvc_name='iris-data-pvc', mount_path='/data')"
