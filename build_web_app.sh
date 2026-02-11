#docker build -f web_app/Dockerfile -t penicillin-web-app .
PROJECT_ID=total-method-473207-e5

WEB_IMAGE="europe-west1-docker.pkg.dev/$PROJECT_ID/ml-apps/penicillin-web-app:multistage-build"


gcloud builds submit --config cloudbuild-web-app.yaml .



gcloud run deploy penicillin-web-app \
  --image "$WEB_IMAGE" \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-env-vars INFERENCE_API_URL="https://inference-api-242173489543.europe-west3.run.app" \
  --set-env-vars GRAFANA_URL="https://narasyimmha.grafana.net/public-dashboards/c666b2b0aa544d20ae6ca0dddb28ad4b" \
  --memory=2Gi \
  --concurrency=1
