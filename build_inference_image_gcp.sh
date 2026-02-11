PROJECT_ID=total-method-473207-e5

PROJECT_ID="total-method-473207-e5"
SA_NAME="bucket-pennicilin-data"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"


INFERENCE_IMAGE="europe-west1-docker.pkg.dev/$PROJECT_ID/ml-apps/inference-api:monitoring_endpoint"

gcloud builds submit --tag "$INFERENCE_IMAGE"   .


gcloud run deploy inference-api \
  --image $INFERENCE_IMAGE \
  --region europe-west3 \
  --allow-unauthenticated \
  --service-account $SA_EMAIL \
  --set-env-vars MODEL_MLFLOW_URI="models:/workspace.default.catboost_model/11" \
  --set-env-vars REPORT_BUCKET="pennicilin_batch_yield" \
  --memory=2Gi \
  --concurrency=1 \
  --cpu=2 \
  --timeout=300



