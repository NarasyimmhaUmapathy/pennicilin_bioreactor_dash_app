PROJECT_ID="total-method-473207-e5"
REGION="europe-west1"
REPO="ml-apps"
IMAGE="europe-west1-docker.pkg.dev/total-method-473207-e5/ml-apps/prometheus-cloudrun"

gcloud builds submit --config cloudbuild-prometheus.yaml .

gcloud run deploy prometheus \
  --image "$IMAGE" \
  --allow-unauthenticated \
  --memory 512Mi \
  --port 8080 \
  --region "$REGION"
