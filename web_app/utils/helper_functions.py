from google.cloud import storage
from datetime import timedelta



def sign_gcs_uri(gcs_uri: str, minutes: int = 30) -> str:
    # gs://bucket/path/to/file.html -> signed https url
    path = gcs_uri.replace("gs://", "", 1)
    bucket_name, blob_name = path.split("/", 1)

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=minutes),
        method="GET",
    )