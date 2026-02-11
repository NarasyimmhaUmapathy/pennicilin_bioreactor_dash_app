from google.cloud import storage
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import io
from loguru import logger



def upload_text_to_gcs(bucket: str, blob_path: str, text: str, content_type: str):
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(blob_path)
    blob.upload_from_string(text, content_type=content_type)
    return f"gs://{bucket}/{blob_path}"


def upload_file_to_gcs(bucket: str, blob_path: str, local_path: str, content_type: str):
    client = storage.Client()
    b = client.bucket(bucket)
    blob = b.blob(blob_path)
    blob.upload_from_filename(local_path, content_type=content_type)
    return f"gs://{bucket}/{blob_path}"

def save_and_upload_drift_report(batch_number: int, report_html_path: str, report_json_path: str):
    bucket = "pennicilin_batch_yield"
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    html_gcs = f"drift_reports/batch={batch_number}/report_{ts}.html"
    json_gcs = f"drift_reports/batch={batch_number}/report_{ts}.json"

    html_uri = upload_file_to_gcs(report_html_path, bucket, html_gcs)
    json_uri = upload_file_to_gcs(report_json_path, bucket, json_gcs)

    # Optional “latest” pointers (overwrite each run)
    upload_file_to_gcs(report_html_path, bucket, f"drift_reports/batch={batch_number}/latest.html")
    upload_file_to_gcs(report_json_path, bucket, f"drift_reports/batch={batch_number}/latest.json")

    return {"drift_report_html_uri": html_uri, "drift_report_json_uri": json_uri}


def upload_file_to_gcs(local_path: str, bucket_name: str, gcs_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{gcs_path}"




def upload_predictions_parquet(
    client: storage.Client,
    bucket_name: str,
    prefix: str,
    batch_number: int,
    predictions,
    rmse: float | None = None,
) -> str:
    """
    Upload predictions as a parquet file to GCS and return the gs:// URI.
    Does **not** depend on timezone.utc to avoid NameError.
    """

    # 1) Build a timestamp in UTC, no timezone object needed
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 2) Create DataFrame
    df = pd.DataFrame({"prediction": list(predictions)})
    if rmse is not None:
        df["rmse"] = rmse

    # 3) Serialize to parquet in memory
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="snappy")
    buf.seek(0)

    # 4) Build GCS object name
    object_name = f"{prefix}/batch={batch_number}/predictions_{ts}.parquet"
    uri = f"gs://{bucket_name}/{object_name}"

    logger.info(f"Uploading predictions parquet to {uri}")

    # 5) Upload to GCS
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_file(buf, content_type="application/octet-stream")

    logger.info("Upload complete")
    return uri

def load_csv_from_gcs(bucket_name: str, folder_name: str, data_name: str) -> pd.DataFrame:
    df = pd.read_csv(f"gs://{bucket_name}/{folder_name}/{data_name}")
    return df

def parse_gcs_uri(uri: str):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    path = uri[len("gs://"):]
    bucket, blob = path.split("/", 1)
    return bucket, blob