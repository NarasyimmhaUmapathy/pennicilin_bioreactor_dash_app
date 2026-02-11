REGION=europe-west3
REPO=pennicilin-model

gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="Docker images for ML services"

gcloud auth configure-docker $REGION-docker.pkg.dev
