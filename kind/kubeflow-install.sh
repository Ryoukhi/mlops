#!/usr/bin/env bash
set -euo pipefail

echo "[1/6] Création du cluster Kind..."
kind create cluster --name mlops-kind --config kind/kind-config.yaml || true

echo "[2/6] Vérification du cluster..."
kubectl cluster-info
kubectl get nodes

echo "[3/6] Installation de Kubeflow via kubectl apply..."
# Version stable de référence (à ajuster selon votre besoin)
kubectl apply -k "github.com/kubeflow/manifests.git/manifests/quick-start?ref=v1.9.0"

echo "[4/6] Attente des pods..."
kubectl wait --for=condition=Available --timeout=20m deployment -n kubeflow --all || true
kubectl get pods -A | head -200

echo "[5/6] Vérification d'accès central"
kubectl get ingress -A || true
kubectl get svc -n istio-system || true

echo "[6/6] Informations de port-forward"
printf "\nAccès Kubeflow UI :\n"
printf "kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80\n"
printf "kubectl port-forward -n kubeflow svc/centraldashboard 8081:80\n"
printf "\nPuis ouvrez : http://localhost:8081\n"
