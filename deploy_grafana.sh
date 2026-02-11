PROJECT_ID="total-method-473207-e5"
REGION="europe-west1"
REPO="ml-apps"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/grafana-cloudrun:latest"

# Replace with your Prometheus Cloud Run URL
PROM_URL="https://prometheus-242173489543.europe-west1.run.app"

gcloud builds submit --config cloudbuild-grafana.yaml .

gcloud run deploy grafana \
  --image "$IMAGE" \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars PROMETHEUS_URL="$PROM_URL" \
  --memory 512Mi \
  --port 8080
