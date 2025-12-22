"""
GCP Cloud Domain

Covers:
- Google Cloud SDK (google-cloud-* packages)
- Cloud Functions, Cloud Run
- BigQuery, Firestore, Cloud Storage
- Pub/Sub, Cloud Tasks
- GKE (Google Kubernetes Engine)
- Vertex AI
"""

from typing import List
from .base import BaseDomain, DomainExample


class GCPCloudDomain(BaseDomain):
    """GCP cloud services training examples."""

    def get_name(self) -> str:
        return "GCP Cloud"

    def get_description(self) -> str:
        return "GCP services including Cloud Functions, BigQuery, Firestore, and Vertex AI"

    def get_subdomains(self) -> List[str]:
        return ["cloud_functions", "bigquery", "firestore", "cloud_storage", "pubsub", "vertex_ai"]

    def get_examples(self) -> List[DomainExample]:
        examples = []
        examples.extend(self._cloud_functions_examples())
        examples.extend(self._bigquery_examples())
        examples.extend(self._firestore_examples())
        examples.extend(self._cloud_storage_examples())
        examples.extend(self._pubsub_examples())
        examples.extend(self._vertex_ai_examples())
        return examples

    def _cloud_functions_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a GCP Cloud Function with HTTP trigger",
                code='''import functions_framework
from flask import jsonify
import logging
from google.cloud import secretmanager
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_secret(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """Retrieve a secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function with CORS support.

    Args:
        request (flask.Request): The request object.
    Returns:
        Response object using Flask's jsonify.
    """
    # Set CORS headers for preflight requests
    if request.method == "OPTIONS":
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }
        return ("", 204, headers)

    # Set CORS headers for main requests
    headers = {"Access-Control-Allow-Origin": "*"}

    try:
        request_json = request.get_json(silent=True)
        request_args = request.args

        if request_json and "name" in request_json:
            name = request_json["name"]
        elif request_args and "name" in request_args:
            name = request_args["name"]
        else:
            name = "World"

        response_data = {
            "message": f"Hello, {name}!",
            "method": request.method,
            "path": request.path,
        }

        logger.info(f"Processed request for: {name}")
        return (jsonify(response_data), 200, headers)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return (jsonify({"error": str(e)}), 500, headers)


@functions_framework.http
def process_webhook(request):
    """Process incoming webhook with validation."""
    import hmac
    import hashlib

    # Verify webhook signature
    signature = request.headers.get("X-Webhook-Signature")
    if not signature:
        return jsonify({"error": "Missing signature"}), 401

    secret = get_secret("my-project", "webhook-secret")
    payload = request.get_data()

    expected_sig = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(signature, f"sha256={expected_sig}"):
        return jsonify({"error": "Invalid signature"}), 401

    # Process the webhook
    data = request.get_json()
    event_type = data.get("type")

    handlers = {
        "user.created": handle_user_created,
        "order.completed": handle_order_completed,
        "payment.received": handle_payment_received,
    }

    handler = handlers.get(event_type)
    if handler:
        result = handler(data)
        return jsonify({"status": "processed", "result": result})

    return jsonify({"status": "ignored", "event": event_type})


def handle_user_created(data):
    logger.info(f"New user: {data.get('user_id')}")
    return {"action": "welcome_email_sent"}

def handle_order_completed(data):
    logger.info(f"Order completed: {data.get('order_id')}")
    return {"action": "fulfillment_started"}

def handle_payment_received(data):
    logger.info(f"Payment received: {data.get('payment_id')}")
    return {"action": "receipt_generated"}
''',
                domain="gcp",
                subdomain="cloud_functions",
                tags=["serverless", "http", "webhook", "cors"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Create a GCP Cloud Function triggered by Pub/Sub",
                code='''import base64
import functions_framework
from google.cloud import bigquery
from google.cloud import storage
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def process_pubsub_message(cloud_event):
    """Process a Pub/Sub message.

    Args:
        cloud_event: CloudEvent containing Pub/Sub message data
    """
    # Decode the Pub/Sub message
    message_data = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    attributes = cloud_event.data["message"].get("attributes", {})

    logger.info(f"Received message: {message_data}")
    logger.info(f"Attributes: {attributes}")

    try:
        payload = json.loads(message_data)

        # Route based on message type
        message_type = payload.get("type")

        if message_type == "analytics_event":
            process_analytics_event(payload)
        elif message_type == "log_event":
            process_log_event(payload)
        elif message_type == "etl_job":
            trigger_etl_job(payload)
        else:
            logger.warning(f"Unknown message type: {message_type}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse message: {e}")
        # Don't raise - acknowledge message to prevent infinite retries
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise  # Raise to trigger retry


def process_analytics_event(payload: dict):
    """Store analytics event in BigQuery."""
    client = bigquery.Client()

    table_id = "project.dataset.analytics_events"

    rows_to_insert = [{
        "event_id": payload.get("event_id"),
        "user_id": payload.get("user_id"),
        "event_type": payload.get("event_type"),
        "properties": json.dumps(payload.get("properties", {})),
        "timestamp": payload.get("timestamp"),
    }]

    errors = client.insert_rows_json(table_id, rows_to_insert)

    if errors:
        logger.error(f"BigQuery insert errors: {errors}")
        raise RuntimeError(f"Failed to insert rows: {errors}")

    logger.info(f"Inserted analytics event: {payload.get('event_id')}")


def process_log_event(payload: dict):
    """Archive log event to Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket("my-logs-bucket")

    # Organize by date
    from datetime import datetime
    date_str = datetime.utcnow().strftime("%Y/%m/%d")
    blob_name = f"logs/{date_str}/{payload.get('log_id')}.json"

    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(payload),
        content_type="application/json"
    )

    logger.info(f"Archived log to: gs://my-logs-bucket/{blob_name}")


def trigger_etl_job(payload: dict):
    """Trigger a Dataflow ETL job."""
    from googleapiclient.discovery import build

    dataflow = build("dataflow", "v1b3")

    project_id = "my-project"
    template_path = f"gs://my-templates-bucket/templates/{payload.get('template')}"

    request_body = {
        "jobName": f"etl-{payload.get('job_id')}",
        "parameters": payload.get("parameters", {}),
        "environment": {
            "tempLocation": "gs://my-temp-bucket/temp",
            "zone": "us-central1-f",
        }
    }

    response = dataflow.projects().templates().launch(
        projectId=project_id,
        gcsPath=template_path,
        body=request_body
    ).execute()

    logger.info(f"Launched Dataflow job: {response.get('job', {}).get('id')}")
''',
                domain="gcp",
                subdomain="cloud_functions",
                tags=["pubsub", "bigquery", "storage", "dataflow"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create a Cloud Function triggered by Cloud Storage events",
                code='''import functions_framework
from google.cloud import vision
from google.cloud import storage
from google.cloud import firestore
import json
import logging
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def process_uploaded_image(cloud_event):
    """Process image uploaded to Cloud Storage.

    Triggered by finalize events on a GCS bucket.
    Performs image analysis and stores results.
    """
    data = cloud_event.data

    bucket_name = data["bucket"]
    file_name = data["name"]
    content_type = data.get("contentType", "")

    logger.info(f"Processing: gs://{bucket_name}/{file_name}")

    # Only process images
    if not content_type.startswith("image/"):
        logger.info(f"Skipping non-image file: {content_type}")
        return

    # Skip thumbnails we create
    if file_name.startswith("thumbnails/"):
        return

    try:
        # Download the image
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        image_bytes = blob.download_as_bytes()

        # Analyze with Vision API
        analysis_result = analyze_image(image_bytes)

        # Generate thumbnail
        thumbnail_path = generate_thumbnail(
            bucket, file_name, image_bytes
        )

        # Store metadata in Firestore
        store_image_metadata(
            bucket_name, file_name,
            analysis_result, thumbnail_path
        )

        logger.info(f"Successfully processed: {file_name}")

    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        raise


def analyze_image(image_bytes: bytes) -> dict:
    """Analyze image using Vision API."""
    client = vision.ImageAnnotatorClient()

    image = vision.Image(content=image_bytes)

    # Perform multiple detection types
    features = [
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION, max_results=10),
        vision.Feature(type_=vision.Feature.Type.FACE_DETECTION),
        vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
        vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
    ]

    request = vision.AnnotateImageRequest(image=image, features=features)
    response = client.annotate_image(request=request)

    # Extract labels
    labels = [
        {"description": label.description, "score": label.score}
        for label in response.label_annotations
    ]

    # Extract dominant colors
    colors = []
    if response.image_properties_annotation.dominant_colors:
        for color in response.image_properties_annotation.dominant_colors.colors[:5]:
            colors.append({
                "red": color.color.red,
                "green": color.color.green,
                "blue": color.color.blue,
                "score": color.score,
                "pixel_fraction": color.pixel_fraction,
            })

    # Safe search results
    safe_search = response.safe_search_annotation
    safety = {
        "adult": safe_search.adult.name,
        "violence": safe_search.violence.name,
        "racy": safe_search.racy.name,
    }

    # Face count
    face_count = len(response.face_annotations)

    return {
        "labels": labels,
        "dominant_colors": colors,
        "safety": safety,
        "face_count": face_count,
    }


def generate_thumbnail(bucket, original_path: str, image_bytes: bytes) -> str:
    """Generate and upload thumbnail."""
    # Create thumbnail
    img = Image.open(io.BytesIO(image_bytes))
    img.thumbnail((200, 200))

    # Convert to bytes
    thumb_buffer = io.BytesIO()
    img.save(thumb_buffer, format=img.format or "JPEG")
    thumb_bytes = thumb_buffer.getvalue()

    # Upload thumbnail
    thumb_path = f"thumbnails/{original_path}"
    thumb_blob = bucket.blob(thumb_path)
    thumb_blob.upload_from_string(
        thumb_bytes,
        content_type="image/jpeg"
    )

    return thumb_path


def store_image_metadata(bucket: str, path: str, analysis: dict, thumbnail: str):
    """Store image metadata in Firestore."""
    db = firestore.Client()

    doc_id = path.replace("/", "_").replace(".", "_")

    doc_ref = db.collection("images").document(doc_id)
    doc_ref.set({
        "bucket": bucket,
        "path": path,
        "thumbnail": thumbnail,
        "analysis": analysis,
        "processed_at": firestore.SERVER_TIMESTAMP,
    })
''',
                domain="gcp",
                subdomain="cloud_functions",
                tags=["storage", "vision", "firestore", "image_processing"],
                difficulty="advanced"
            ),
        ]

    def _bigquery_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a BigQuery client with complex queries and data operations",
                code='''from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryManager:
    """Manager for BigQuery operations."""

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)

    def create_dataset(self, dataset_id: str, location: str = "US") -> bigquery.Dataset:
        """Create a dataset if it doesn't exist."""
        dataset_ref = f"{self.project_id}.{dataset_id}"

        try:
            dataset = self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {dataset_id} already exists")
        except NotFound:
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = location
            dataset = self.client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Created dataset {dataset_id}")

        return dataset

    def create_table(
        self,
        dataset_id: str,
        table_id: str,
        schema: List[bigquery.SchemaField],
        partition_field: Optional[str] = None,
        clustering_fields: Optional[List[str]] = None
    ) -> bigquery.Table:
        """Create a partitioned and clustered table."""
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        table = bigquery.Table(table_ref, schema=schema)

        # Configure partitioning
        if partition_field:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_field,
            )

        # Configure clustering
        if clustering_fields:
            table.clustering_fields = clustering_fields

        table = self.client.create_table(table, exists_ok=True)
        logger.info(f"Created table {table_id}")

        return table

    def run_query(
        self,
        query: str,
        params: Optional[List[bigquery.ScalarQueryParameter]] = None,
        dry_run: bool = False
    ) -> bigquery.QueryJob:
        """Run a parameterized query."""
        job_config = bigquery.QueryJobConfig(
            query_parameters=params or [],
            dry_run=dry_run,
            use_query_cache=True,
        )

        query_job = self.client.query(query, job_config=job_config)

        if dry_run:
            logger.info(
                f"Query will process {query_job.total_bytes_processed:,} bytes"
            )
            return query_job

        # Wait for completion
        results = query_job.result()
        logger.info(f"Query processed {query_job.total_bytes_processed:,} bytes")

        return query_job

    def query_to_dataframe(self, query: str) -> pd.DataFrame:
        """Run query and return as pandas DataFrame."""
        return self.client.query(query).to_dataframe()

    def load_dataframe(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        table_id: str,
        write_disposition: str = "WRITE_APPEND"
    ):
        """Load a pandas DataFrame to BigQuery."""
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"

        job_config = bigquery.LoadJobConfig(
            write_disposition=write_disposition,
            autodetect=True,
        )

        job = self.client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        job.result()

        logger.info(f"Loaded {len(df)} rows to {table_id}")

    def create_view(
        self,
        dataset_id: str,
        view_id: str,
        query: str
    ) -> bigquery.Table:
        """Create or update a view."""
        view_ref = f"{self.project_id}.{dataset_id}.{view_id}"

        view = bigquery.Table(view_ref)
        view.view_query = query

        view = self.client.create_table(view, exists_ok=True)
        logger.info(f"Created view {view_id}")

        return view


# Complex analytical queries
ANALYTICS_QUERIES = {
    "user_retention": """
        WITH first_activity AS (
            SELECT
                user_id,
                DATE(MIN(event_timestamp)) as cohort_date
            FROM `{project}.analytics.events`
            WHERE event_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
            GROUP BY user_id
        ),
        user_activity AS (
            SELECT
                e.user_id,
                f.cohort_date,
                DATE(e.event_timestamp) as activity_date,
                DATE_DIFF(DATE(e.event_timestamp), f.cohort_date, DAY) as day_number
            FROM `{project}.analytics.events` e
            JOIN first_activity f ON e.user_id = f.user_id
        )
        SELECT
            cohort_date,
            day_number,
            COUNT(DISTINCT user_id) as users,
            ROUND(
                COUNT(DISTINCT user_id) * 100.0 /
                FIRST_VALUE(COUNT(DISTINCT user_id)) OVER (
                    PARTITION BY cohort_date ORDER BY day_number
                ), 2
            ) as retention_rate
        FROM user_activity
        WHERE day_number <= 30
        GROUP BY cohort_date, day_number
        ORDER BY cohort_date, day_number
    """,

    "funnel_analysis": """
        WITH funnel_events AS (
            SELECT
                session_id,
                user_id,
                event_name,
                event_timestamp,
                CASE event_name
                    WHEN 'page_view' THEN 1
                    WHEN 'add_to_cart' THEN 2
                    WHEN 'begin_checkout' THEN 3
                    WHEN 'purchase' THEN 4
                END as step_number
            FROM `{project}.analytics.events`
            WHERE DATE(event_timestamp) = @report_date
              AND event_name IN ('page_view', 'add_to_cart', 'begin_checkout', 'purchase')
        ),
        max_step_per_session AS (
            SELECT
                session_id,
                MAX(step_number) as max_step
            FROM funnel_events
            GROUP BY session_id
        )
        SELECT
            step_number,
            CASE step_number
                WHEN 1 THEN 'Page View'
                WHEN 2 THEN 'Add to Cart'
                WHEN 3 THEN 'Begin Checkout'
                WHEN 4 THEN 'Purchase'
            END as step_name,
            COUNT(*) as sessions,
            ROUND(COUNT(*) * 100.0 /
                  SUM(COUNT(*)) OVER (), 2) as percentage
        FROM max_step_per_session
        GROUP BY step_number
        ORDER BY step_number
    """,

    "revenue_by_segment": """
        SELECT
            customer_segment,
            DATE_TRUNC(order_date, MONTH) as month,
            COUNT(DISTINCT customer_id) as customers,
            COUNT(order_id) as orders,
            SUM(order_total) as revenue,
            AVG(order_total) as avg_order_value,
            SUM(order_total) / COUNT(DISTINCT customer_id) as revenue_per_customer
        FROM `{project}.sales.orders` o
        JOIN `{project}.customers.segments` s ON o.customer_id = s.customer_id
        WHERE order_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 12 MONTH)
        GROUP BY customer_segment, month
        ORDER BY month DESC, revenue DESC
    """
}


# Usage example
def main():
    bq = BigQueryManager("my-project")

    # Create dataset and table
    bq.create_dataset("analytics")

    schema = [
        bigquery.SchemaField("event_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("user_id", "STRING"),
        bigquery.SchemaField("event_name", "STRING"),
        bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
        bigquery.SchemaField("properties", "JSON"),
    ]

    bq.create_table(
        "analytics", "events", schema,
        partition_field="event_timestamp",
        clustering_fields=["user_id", "event_name"]
    )

    # Run parameterized query
    query = """
        SELECT user_id, COUNT(*) as event_count
        FROM `my-project.analytics.events`
        WHERE event_timestamp >= @start_date
          AND event_name = @event_name
        GROUP BY user_id
        HAVING event_count >= @min_count
        ORDER BY event_count DESC
        LIMIT 100
    """

    params = [
        bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", "2024-01-01"),
        bigquery.ScalarQueryParameter("event_name", "STRING", "purchase"),
        bigquery.ScalarQueryParameter("min_count", "INT64", 5),
    ]

    job = bq.run_query(query, params)

    for row in job:
        print(f"{row.user_id}: {row.event_count}")


if __name__ == "__main__":
    main()
''',
                domain="gcp",
                subdomain="bigquery",
                tags=["analytics", "sql", "data_warehouse", "partitioning"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create BigQuery scheduled queries and data pipelines",
                code='''from google.cloud import bigquery
from google.cloud import bigquery_datatransfer
from google.protobuf import struct_pb2
import json
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BigQueryPipelineManager:
    """Manage BigQuery scheduled queries and data transfers."""

    def __init__(self, project_id: str, location: str = "us"):
        self.project_id = project_id
        self.location = location
        self.bq_client = bigquery.Client(project=project_id)
        self.transfer_client = bigquery_datatransfer.DataTransferServiceClient()
        self.parent = f"projects/{project_id}/locations/{location}"

    def create_scheduled_query(
        self,
        name: str,
        query: str,
        destination_table: str,
        schedule: str = "every 24 hours",
        write_disposition: str = "WRITE_TRUNCATE",
        partitioning_field: Optional[str] = None,
    ) -> str:
        """Create a scheduled query.

        Args:
            name: Display name for the transfer
            query: SQL query to execute
            destination_table: Target table (dataset.table)
            schedule: Cron-like schedule (e.g., "every 24 hours", "every monday 09:00")
            write_disposition: WRITE_TRUNCATE, WRITE_APPEND, or WRITE_EMPTY
            partitioning_field: Field for partition filtering

        Returns:
            Transfer config name
        """
        # Build params
        params = struct_pb2.Struct()
        params.update({
            "query": query,
            "destination_table_name_template": destination_table,
            "write_disposition": write_disposition,
        })

        if partitioning_field:
            params.update({
                "partitioning_field": partitioning_field,
            })

        transfer_config = bigquery_datatransfer.TransferConfig(
            display_name=name,
            data_source_id="scheduled_query",
            params=params,
            schedule=schedule,
            disabled=False,
        )

        response = self.transfer_client.create_transfer_config(
            parent=self.parent,
            transfer_config=transfer_config,
        )

        logger.info(f"Created scheduled query: {response.name}")
        return response.name

    def create_incremental_query(
        self,
        name: str,
        source_table: str,
        destination_table: str,
        timestamp_column: str,
        schedule: str = "every 1 hours",
    ) -> str:
        """Create an incremental scheduled query using run_time parameter."""
        query = f"""
            -- Incremental load: only new records since last run
            SELECT *
            FROM `{self.project_id}.{source_table}`
            WHERE {timestamp_column} >= TIMESTAMP_SUB(@run_time, INTERVAL 1 HOUR)
              AND {timestamp_column} < @run_time
        """

        return self.create_scheduled_query(
            name=name,
            query=query,
            destination_table=destination_table,
            schedule=schedule,
            write_disposition="WRITE_APPEND",
        )

    def create_s3_to_bq_transfer(
        self,
        name: str,
        s3_uri: str,
        destination_table: str,
        aws_access_key: str,
        aws_secret_key: str,
        file_format: str = "CSV",
        schedule: str = "every 24 hours",
    ) -> str:
        """Create a data transfer from S3 to BigQuery."""
        params = struct_pb2.Struct()
        params.update({
            "data_path_template": s3_uri,
            "destination_table_name_template": destination_table,
            "access_key_id": aws_access_key,
            "secret_access_key": aws_secret_key,
            "file_format": file_format,
            "max_bad_records": "0",
            "ignore_unknown_values": "false",
        })

        transfer_config = bigquery_datatransfer.TransferConfig(
            display_name=name,
            data_source_id="amazon_s3",
            params=params,
            schedule=schedule,
        )

        response = self.transfer_client.create_transfer_config(
            parent=self.parent,
            transfer_config=transfer_config,
        )

        logger.info(f"Created S3 transfer: {response.name}")
        return response.name

    def create_gcs_to_bq_transfer(
        self,
        name: str,
        gcs_uri: str,
        destination_dataset: str,
        destination_table: str,
        schema: Optional[list] = None,
        schedule: str = "every 24 hours",
    ) -> str:
        """Create a data transfer from GCS to BigQuery."""
        params = struct_pb2.Struct()
        params.update({
            "data_path_template": gcs_uri,
            "destination_table_name_template": destination_table,
            "file_format": "CSV",
            "skip_leading_rows": "1",
            "write_disposition": "WRITE_TRUNCATE",
        })

        if schema:
            params.update({"schema": json.dumps(schema)})

        transfer_config = bigquery_datatransfer.TransferConfig(
            display_name=name,
            data_source_id="google_cloud_storage",
            destination_dataset_id=destination_dataset,
            params=params,
            schedule=schedule,
        )

        response = self.transfer_client.create_transfer_config(
            parent=self.parent,
            transfer_config=transfer_config,
        )

        logger.info(f"Created GCS transfer: {response.name}")
        return response.name

    def trigger_manual_run(self, transfer_config_name: str):
        """Trigger a manual run of a scheduled query."""
        from datetime import datetime
        from google.protobuf.timestamp_pb2 import Timestamp

        now = Timestamp()
        now.FromDatetime(datetime.utcnow())

        response = self.transfer_client.start_manual_transfer_runs(
            parent=transfer_config_name,
            requested_run_time=now,
        )

        logger.info(f"Triggered manual run: {response.runs}")
        return response

    def list_transfers(self) -> list:
        """List all transfer configurations."""
        transfers = []

        request = bigquery_datatransfer.ListTransferConfigsRequest(
            parent=self.parent,
        )

        for config in self.transfer_client.list_transfer_configs(request):
            transfers.append({
                "name": config.name,
                "display_name": config.display_name,
                "data_source_id": config.data_source_id,
                "schedule": config.schedule,
                "state": config.state.name,
            })

        return transfers

    def get_run_history(self, transfer_config_name: str, limit: int = 10) -> list:
        """Get run history for a transfer."""
        runs = []

        request = bigquery_datatransfer.ListTransferRunsRequest(
            parent=transfer_config_name,
        )

        for run in self.transfer_client.list_transfer_runs(request):
            runs.append({
                "name": run.name,
                "state": run.state.name,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "error_status": run.error_status.message if run.error_status else None,
            })

            if len(runs) >= limit:
                break

        return runs


# Usage
def main():
    pipeline = BigQueryPipelineManager("my-project")

    # Create daily aggregation query
    pipeline.create_scheduled_query(
        name="Daily Sales Summary",
        query="""
            SELECT
                DATE(order_timestamp) as order_date,
                product_category,
                COUNT(*) as order_count,
                SUM(order_total) as total_revenue,
                AVG(order_total) as avg_order_value
            FROM `my-project.sales.orders`
            WHERE DATE(order_timestamp) = DATE_SUB(@run_date, INTERVAL 1 DAY)
            GROUP BY order_date, product_category
        """,
        destination_table="analytics.daily_sales_summary${run_date}",
        schedule="every day 02:00",
        write_disposition="WRITE_TRUNCATE",
        partitioning_field="order_date",
    )

    # Create incremental transfer
    pipeline.create_incremental_query(
        name="Hourly Event Sync",
        source_table="raw.events",
        destination_table="processed.events",
        timestamp_column="created_at",
        schedule="every 1 hours",
    )

    # List all transfers
    for transfer in pipeline.list_transfers():
        print(f"{transfer['display_name']}: {transfer['state']}")


if __name__ == "__main__":
    main()
''',
                domain="gcp",
                subdomain="bigquery",
                tags=["scheduled_query", "data_transfer", "etl", "pipeline"],
                difficulty="advanced"
            ),
        ]

    def _firestore_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a Firestore database manager with CRUD operations and queries",
                code='''from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter, Query
from google.cloud.firestore_v1.base_query import BaseCompositeFilter
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class User:
    """User model."""
    email: str
    name: str
    role: str = "user"
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class FirestoreManager:
    """Manager for Firestore operations."""

    def __init__(self, project_id: Optional[str] = None):
        self.db = firestore.Client(project=project_id)

    # --- Basic CRUD Operations ---

    def create_document(
        self,
        collection: str,
        data: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        """Create a document with optional custom ID."""
        data["created_at"] = firestore.SERVER_TIMESTAMP
        data["updated_at"] = firestore.SERVER_TIMESTAMP

        if document_id:
            doc_ref = self.db.collection(collection).document(document_id)
            doc_ref.set(data)
            return document_id
        else:
            doc_ref = self.db.collection(collection).add(data)
            return doc_ref[1].id

    def get_document(
        self,
        collection: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        doc_ref = self.db.collection(collection).document(document_id)
        doc = doc_ref.get()

        if doc.exists:
            return {"id": doc.id, **doc.to_dict()}
        return None

    def update_document(
        self,
        collection: str,
        document_id: str,
        data: Dict[str, Any],
        merge: bool = True
    ):
        """Update a document."""
        data["updated_at"] = firestore.SERVER_TIMESTAMP

        doc_ref = self.db.collection(collection).document(document_id)
        doc_ref.set(data, merge=merge)

    def delete_document(self, collection: str, document_id: str):
        """Delete a document."""
        self.db.collection(collection).document(document_id).delete()

    # --- Query Operations ---

    def query_documents(
        self,
        collection: str,
        filters: List[tuple] = None,
        order_by: str = None,
        order_direction: str = "ASCENDING",
        limit: int = None,
        offset: int = None,
    ) -> List[Dict[str, Any]]:
        """Query documents with filters.

        Args:
            collection: Collection name
            filters: List of (field, operator, value) tuples
            order_by: Field to order by
            order_direction: ASCENDING or DESCENDING
            limit: Max documents to return
            offset: Number of documents to skip
        """
        query = self.db.collection(collection)

        # Apply filters
        if filters:
            for field, operator, value in filters:
                query = query.where(filter=FieldFilter(field, operator, value))

        # Apply ordering
        if order_by:
            direction = (
                firestore.Query.DESCENDING
                if order_direction == "DESCENDING"
                else firestore.Query.ASCENDING
            )
            query = query.order_by(order_by, direction=direction)

        # Apply limit and offset
        if offset:
            # Get the offset document first
            offset_docs = list(query.limit(offset).stream())
            if offset_docs:
                last_doc = offset_docs[-1]
                query = query.start_after(last_doc)

        if limit:
            query = query.limit(limit)

        # Execute query
        results = []
        for doc in query.stream():
            results.append({"id": doc.id, **doc.to_dict()})

        return results

    def compound_query(
        self,
        collection: str,
        and_filters: List[tuple] = None,
        or_filters: List[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """Execute compound AND/OR queries (requires composite index)."""
        query = self.db.collection(collection)

        conditions = []

        if and_filters:
            for field, operator, value in and_filters:
                conditions.append(FieldFilter(field, operator, value))

        # Note: OR queries require Firestore Rules and composite indexes
        if conditions:
            for condition in conditions:
                query = query.where(filter=condition)

        return [{"id": doc.id, **doc.to_dict()} for doc in query.stream()]

    # --- Batch Operations ---

    def batch_create(
        self,
        collection: str,
        documents: List[Dict[str, Any]]
    ) -> List[str]:
        """Create multiple documents in a batch (max 500)."""
        batch = self.db.batch()
        doc_ids = []

        for doc_data in documents:
            doc_data["created_at"] = firestore.SERVER_TIMESTAMP
            doc_data["updated_at"] = firestore.SERVER_TIMESTAMP

            doc_ref = self.db.collection(collection).document()
            batch.set(doc_ref, doc_data)
            doc_ids.append(doc_ref.id)

        batch.commit()
        logger.info(f"Batch created {len(documents)} documents")

        return doc_ids

    def batch_update(
        self,
        collection: str,
        updates: List[tuple],  # [(doc_id, data), ...]
    ):
        """Update multiple documents in a batch."""
        batch = self.db.batch()

        for doc_id, data in updates:
            data["updated_at"] = firestore.SERVER_TIMESTAMP
            doc_ref = self.db.collection(collection).document(doc_id)
            batch.update(doc_ref, data)

        batch.commit()
        logger.info(f"Batch updated {len(updates)} documents")

    def batch_delete(self, collection: str, document_ids: List[str]):
        """Delete multiple documents in a batch."""
        batch = self.db.batch()

        for doc_id in document_ids:
            doc_ref = self.db.collection(collection).document(doc_id)
            batch.delete(doc_ref)

        batch.commit()
        logger.info(f"Batch deleted {len(document_ids)} documents")

    # --- Transaction Operations ---

    def transfer_credits(
        self,
        from_user_id: str,
        to_user_id: str,
        amount: int,
    ) -> bool:
        """Transfer credits between users using a transaction."""

        @firestore.transactional
        def update_in_transaction(transaction, from_ref, to_ref):
            from_doc = from_ref.get(transaction=transaction)
            to_doc = to_ref.get(transaction=transaction)

            if not from_doc.exists or not to_doc.exists:
                raise ValueError("User not found")

            from_credits = from_doc.get("credits") or 0

            if from_credits < amount:
                raise ValueError("Insufficient credits")

            to_credits = to_doc.get("credits") or 0

            transaction.update(from_ref, {
                "credits": from_credits - amount,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })
            transaction.update(to_ref, {
                "credits": to_credits + amount,
                "updated_at": firestore.SERVER_TIMESTAMP,
            })

            return True

        from_ref = self.db.collection("users").document(from_user_id)
        to_ref = self.db.collection("users").document(to_user_id)

        transaction = self.db.transaction()
        return update_in_transaction(transaction, from_ref, to_ref)

    # --- Subcollections ---

    def add_to_subcollection(
        self,
        parent_collection: str,
        parent_id: str,
        subcollection: str,
        data: Dict[str, Any],
    ) -> str:
        """Add a document to a subcollection."""
        data["created_at"] = firestore.SERVER_TIMESTAMP

        doc_ref = (
            self.db.collection(parent_collection)
            .document(parent_id)
            .collection(subcollection)
            .add(data)
        )

        return doc_ref[1].id

    def get_subcollection(
        self,
        parent_collection: str,
        parent_id: str,
        subcollection: str,
    ) -> List[Dict[str, Any]]:
        """Get all documents from a subcollection."""
        docs = (
            self.db.collection(parent_collection)
            .document(parent_id)
            .collection(subcollection)
            .stream()
        )

        return [{"id": doc.id, **doc.to_dict()} for doc in docs]

    # --- Real-time Listeners ---

    def watch_document(
        self,
        collection: str,
        document_id: str,
        callback: callable,
    ):
        """Watch a document for changes."""
        doc_ref = self.db.collection(collection).document(document_id)

        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                callback({"id": doc.id, **doc.to_dict()})

        return doc_ref.on_snapshot(on_snapshot)

    def watch_collection(
        self,
        collection: str,
        callback: callable,
        filters: List[tuple] = None,
    ):
        """Watch a collection for changes."""
        query = self.db.collection(collection)

        if filters:
            for field, operator, value in filters:
                query = query.where(filter=FieldFilter(field, operator, value))

        def on_snapshot(query_snapshot, changes, read_time):
            for change in changes:
                doc = change.document
                callback({
                    "type": change.type.name,
                    "id": doc.id,
                    "data": doc.to_dict(),
                })

        return query.on_snapshot(on_snapshot)


# Usage
def main():
    fs = FirestoreManager()

    # Create user
    user_id = fs.create_document("users", {
        "email": "test@example.com",
        "name": "Test User",
        "role": "admin",
        "credits": 100,
    })

    # Query users
    admins = fs.query_documents(
        "users",
        filters=[("role", "==", "admin")],
        order_by="created_at",
        order_direction="DESCENDING",
        limit=10,
    )

    # Add order to user's subcollection
    order_id = fs.add_to_subcollection(
        "users", user_id, "orders",
        {"product": "Widget", "quantity": 5, "total": 49.99}
    )

    print(f"Created user: {user_id}")
    print(f"Created order: {order_id}")
    print(f"Found {len(admins)} admins")


if __name__ == "__main__":
    main()
''',
                domain="gcp",
                subdomain="firestore",
                tags=["nosql", "database", "realtime", "transactions"],
                difficulty="advanced"
            ),
        ]

    def _cloud_storage_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a Cloud Storage manager with advanced operations",
                code='''from google.cloud import storage
from google.cloud.storage import transfer_manager
from typing import List, Optional, Dict, Any, BinaryIO
from datetime import timedelta
import hashlib
import mimetypes
import logging
from pathlib import Path
import concurrent.futures

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CloudStorageManager:
    """Manager for Google Cloud Storage operations."""

    def __init__(self, project_id: Optional[str] = None):
        self.client = storage.Client(project=project_id)

    # --- Bucket Operations ---

    def create_bucket(
        self,
        bucket_name: str,
        location: str = "US",
        storage_class: str = "STANDARD",
        enable_versioning: bool = False,
        lifecycle_rules: List[Dict] = None,
    ) -> storage.Bucket:
        """Create a bucket with configuration."""
        bucket = self.client.bucket(bucket_name)
        bucket.storage_class = storage_class

        # Configure versioning
        bucket.versioning_enabled = enable_versioning

        # Configure lifecycle rules
        if lifecycle_rules:
            bucket.lifecycle_rules = lifecycle_rules

        new_bucket = self.client.create_bucket(bucket, location=location)
        logger.info(f"Created bucket: {bucket_name}")

        return new_bucket

    def configure_bucket_lifecycle(
        self,
        bucket_name: str,
        delete_age_days: int = None,
        archive_age_days: int = None,
        delete_noncurrent_versions_days: int = None,
    ):
        """Configure bucket lifecycle rules."""
        bucket = self.client.bucket(bucket_name)
        rules = []

        # Delete old objects
        if delete_age_days:
            rules.append({
                "action": {"type": "Delete"},
                "condition": {"age": delete_age_days}
            })

        # Move to archive storage
        if archive_age_days:
            rules.append({
                "action": {"type": "SetStorageClass", "storageClass": "ARCHIVE"},
                "condition": {"age": archive_age_days}
            })

        # Delete old versions
        if delete_noncurrent_versions_days:
            rules.append({
                "action": {"type": "Delete"},
                "condition": {
                    "numNewerVersions": 3,
                    "isLive": False,
                    "daysSinceNoncurrentTime": delete_noncurrent_versions_days,
                }
            })

        bucket.lifecycle_rules = rules
        bucket.patch()
        logger.info(f"Updated lifecycle rules for {bucket_name}")

    def enable_bucket_cors(
        self,
        bucket_name: str,
        origins: List[str] = ["*"],
        methods: List[str] = ["GET", "HEAD", "PUT", "POST"],
        max_age_seconds: int = 3600,
    ):
        """Enable CORS on a bucket."""
        bucket = self.client.bucket(bucket_name)

        bucket.cors = [{
            "origin": origins,
            "method": methods,
            "responseHeader": ["Content-Type", "x-goog-resumable"],
            "maxAgeSeconds": max_age_seconds,
        }]

        bucket.patch()
        logger.info(f"Enabled CORS for {bucket_name}")

    # --- Object Operations ---

    def upload_file(
        self,
        bucket_name: str,
        source_path: str,
        destination_blob: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload a file to Cloud Storage."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)

        # Auto-detect content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(source_path)

        # Set metadata
        if metadata:
            blob.metadata = metadata

        blob.upload_from_filename(source_path, content_type=content_type)

        logger.info(f"Uploaded {source_path} to gs://{bucket_name}/{destination_blob}")
        return f"gs://{bucket_name}/{destination_blob}"

    def upload_from_string(
        self,
        bucket_name: str,
        data: str,
        destination_blob: str,
        content_type: str = "text/plain",
    ) -> str:
        """Upload string data to Cloud Storage."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)

        blob.upload_from_string(data, content_type=content_type)

        return f"gs://{bucket_name}/{destination_blob}"

    def upload_from_stream(
        self,
        bucket_name: str,
        file_obj: BinaryIO,
        destination_blob: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload from a file-like object."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)

        blob.upload_from_file(file_obj, content_type=content_type)

        return f"gs://{bucket_name}/{destination_blob}"

    def parallel_upload_directory(
        self,
        bucket_name: str,
        source_directory: str,
        destination_prefix: str = "",
        max_workers: int = 8,
    ) -> List[str]:
        """Upload a directory in parallel using transfer_manager."""
        bucket = self.client.bucket(bucket_name)
        source_path = Path(source_directory)

        # Collect all files
        file_paths = [str(p) for p in source_path.rglob("*") if p.is_file()]

        # Create blob names
        blob_names = []
        for fp in file_paths:
            relative = Path(fp).relative_to(source_path)
            blob_name = f"{destination_prefix}/{relative}" if destination_prefix else str(relative)
            blob_names.append(blob_name)

        # Parallel upload
        results = transfer_manager.upload_many_from_filenames(
            bucket,
            file_paths,
            source_directory=source_directory,
            blob_name_prefix=destination_prefix,
            max_workers=max_workers,
        )

        uploaded = []
        for path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to upload {path}: {result}")
            else:
                uploaded.append(f"gs://{bucket_name}/{result.name}")

        logger.info(f"Uploaded {len(uploaded)}/{len(file_paths)} files")
        return uploaded

    def download_file(
        self,
        bucket_name: str,
        source_blob: str,
        destination_path: str,
    ):
        """Download a file from Cloud Storage."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        blob.download_to_filename(destination_path)
        logger.info(f"Downloaded gs://{bucket_name}/{source_blob} to {destination_path}")

    def download_as_bytes(
        self,
        bucket_name: str,
        source_blob: str,
    ) -> bytes:
        """Download blob content as bytes."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(source_blob)

        return blob.download_as_bytes()

    # --- Signed URLs ---

    def generate_signed_url(
        self,
        bucket_name: str,
        blob_name: str,
        expiration_minutes: int = 15,
        method: str = "GET",
        content_type: Optional[str] = None,
    ) -> str:
        """Generate a signed URL for temporary access."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method=method,
            content_type=content_type,
        )

        return url

    def generate_resumable_upload_url(
        self,
        bucket_name: str,
        blob_name: str,
        content_type: str,
        expiration_minutes: int = 60,
    ) -> str:
        """Generate a signed URL for resumable uploads."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="POST",
            content_type=content_type,
            headers={"x-goog-resumable": "start"},
        )

        return url

    # --- Listing and Search ---

    def list_blobs(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[Dict[str, Any]]:
        """List blobs in a bucket."""
        bucket = self.client.bucket(bucket_name)

        blobs = bucket.list_blobs(
            prefix=prefix,
            delimiter=delimiter,
            max_results=max_results,
        )

        result = []
        for blob in blobs:
            result.append({
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_type,
                "updated": blob.updated.isoformat() if blob.updated else None,
                "md5_hash": blob.md5_hash,
            })

        return result

    def list_prefixes(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """List 'folders' (common prefixes) in a bucket."""
        bucket = self.client.bucket(bucket_name)

        iterator = bucket.list_blobs(prefix=prefix, delimiter="/")

        # Consume the iterator to get prefixes
        list(iterator)

        return list(iterator.prefixes)

    # --- Copy and Move ---

    def copy_blob(
        self,
        source_bucket: str,
        source_blob: str,
        dest_bucket: str,
        dest_blob: str,
    ) -> str:
        """Copy a blob between buckets."""
        source_bucket_obj = self.client.bucket(source_bucket)
        dest_bucket_obj = self.client.bucket(dest_bucket)

        source_blob_obj = source_bucket_obj.blob(source_blob)

        dest_blob_obj = source_bucket_obj.copy_blob(
            source_blob_obj, dest_bucket_obj, dest_blob
        )

        logger.info(f"Copied to gs://{dest_bucket}/{dest_blob}")
        return f"gs://{dest_bucket}/{dest_blob}"

    def move_blob(
        self,
        source_bucket: str,
        source_blob: str,
        dest_bucket: str,
        dest_blob: str,
    ) -> str:
        """Move a blob (copy and delete)."""
        self.copy_blob(source_bucket, source_blob, dest_bucket, dest_blob)

        # Delete source
        source_bucket_obj = self.client.bucket(source_bucket)
        source_bucket_obj.blob(source_blob).delete()

        logger.info(f"Moved to gs://{dest_bucket}/{dest_blob}")
        return f"gs://{dest_bucket}/{dest_blob}"

    def delete_blob(self, bucket_name: str, blob_name: str):
        """Delete a blob."""
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Deleted gs://{bucket_name}/{blob_name}")

    def delete_prefix(self, bucket_name: str, prefix: str):
        """Delete all blobs with a prefix."""
        bucket = self.client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        for blob in blobs:
            blob.delete()
            logger.info(f"Deleted gs://{bucket_name}/{blob.name}")


# Usage
def main():
    gcs = CloudStorageManager()

    # Create bucket with lifecycle
    gcs.create_bucket(
        "my-app-data",
        location="US",
        enable_versioning=True,
    )

    # Configure lifecycle
    gcs.configure_bucket_lifecycle(
        "my-app-data",
        delete_age_days=365,
        archive_age_days=90,
    )

    # Enable CORS
    gcs.enable_bucket_cors(
        "my-app-data",
        origins=["https://myapp.com"],
    )

    # Upload file
    gcs.upload_file(
        "my-app-data",
        "/local/path/file.json",
        "data/file.json",
        metadata={"version": "1.0"},
    )

    # Generate signed URL
    url = gcs.generate_signed_url(
        "my-app-data",
        "data/file.json",
        expiration_minutes=30,
    )
    print(f"Signed URL: {url}")


if __name__ == "__main__":
    main()
''',
                domain="gcp",
                subdomain="cloud_storage",
                tags=["storage", "signed_url", "lifecycle", "parallel_upload"],
                difficulty="advanced"
            ),
        ]

    def _pubsub_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a Pub/Sub publisher and subscriber with error handling",
                code='''from google.cloud import pubsub_v1
from google.api_core import retry
from google.api_core.exceptions import NotFound, AlreadyExists
from concurrent import futures
from typing import Callable, Optional, List, Dict, Any
import json
import logging
import time
from dataclasses import dataclass
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PubSubConfig:
    """Pub/Sub configuration."""
    project_id: str
    ack_deadline_seconds: int = 60
    max_messages: int = 100
    flow_control_max_messages: int = 1000
    flow_control_max_bytes: int = 100 * 1024 * 1024  # 100MB


class PubSubPublisher:
    """Pub/Sub publisher with batching and retry."""

    def __init__(self, config: PubSubConfig):
        self.config = config

        # Configure batch settings
        batch_settings = pubsub_v1.types.BatchSettings(
            max_messages=100,
            max_bytes=1024 * 1024,  # 1MB
            max_latency=0.1,  # 100ms
        )

        self.publisher = pubsub_v1.PublisherClient(
            batch_settings=batch_settings,
        )

    def create_topic(self, topic_id: str) -> str:
        """Create a topic if it doesn't exist."""
        topic_path = self.publisher.topic_path(self.config.project_id, topic_id)

        try:
            self.publisher.create_topic(request={"name": topic_path})
            logger.info(f"Created topic: {topic_id}")
        except AlreadyExists:
            logger.info(f"Topic already exists: {topic_id}")

        return topic_path

    def publish(
        self,
        topic_id: str,
        data: Dict[str, Any],
        attributes: Optional[Dict[str, str]] = None,
        ordering_key: Optional[str] = None,
    ) -> str:
        """Publish a message to a topic."""
        topic_path = self.publisher.topic_path(self.config.project_id, topic_id)

        # Encode data as JSON bytes
        message_bytes = json.dumps(data).encode("utf-8")

        # Prepare publish kwargs
        kwargs = {"data": message_bytes}

        if attributes:
            kwargs["attributes"] = attributes

        if ordering_key:
            kwargs["ordering_key"] = ordering_key

        # Publish with retry
        future = self.publisher.publish(topic_path, **kwargs)

        try:
            message_id = future.result(timeout=30)
            logger.debug(f"Published message: {message_id}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to publish: {e}")
            raise

    def publish_batch(
        self,
        topic_id: str,
        messages: List[Dict[str, Any]],
    ) -> List[str]:
        """Publish multiple messages."""
        topic_path = self.publisher.topic_path(self.config.project_id, topic_id)

        futures = []
        for msg in messages:
            message_bytes = json.dumps(msg).encode("utf-8")
            future = self.publisher.publish(topic_path, data=message_bytes)
            futures.append(future)

        # Wait for all to complete
        message_ids = []
        for future in futures:
            try:
                message_id = future.result(timeout=60)
                message_ids.append(message_id)
            except Exception as e:
                logger.error(f"Batch publish error: {e}")
                message_ids.append(None)

        return message_ids


class PubSubSubscriber:
    """Pub/Sub subscriber with message processing."""

    def __init__(self, config: PubSubConfig):
        self.config = config

        # Configure flow control
        flow_control = pubsub_v1.types.FlowControl(
            max_messages=config.flow_control_max_messages,
            max_bytes=config.flow_control_max_bytes,
        )

        self.subscriber = pubsub_v1.SubscriberClient()
        self.flow_control = flow_control
        self._streaming_pull_future = None

    def create_subscription(
        self,
        topic_id: str,
        subscription_id: str,
        push_endpoint: Optional[str] = None,
        filter_expression: Optional[str] = None,
        dead_letter_topic: Optional[str] = None,
        max_delivery_attempts: int = 5,
    ) -> str:
        """Create a subscription with configuration."""
        topic_path = self.subscriber.topic_path(self.config.project_id, topic_id)
        subscription_path = self.subscriber.subscription_path(
            self.config.project_id, subscription_id
        )

        request = {
            "name": subscription_path,
            "topic": topic_path,
            "ack_deadline_seconds": self.config.ack_deadline_seconds,
        }

        # Push configuration
        if push_endpoint:
            request["push_config"] = {"push_endpoint": push_endpoint}

        # Message filter
        if filter_expression:
            request["filter"] = filter_expression

        # Dead letter policy
        if dead_letter_topic:
            dead_letter_path = self.subscriber.topic_path(
                self.config.project_id, dead_letter_topic
            )
            request["dead_letter_policy"] = {
                "dead_letter_topic": dead_letter_path,
                "max_delivery_attempts": max_delivery_attempts,
            }

        try:
            self.subscriber.create_subscription(request=request)
            logger.info(f"Created subscription: {subscription_id}")
        except AlreadyExists:
            logger.info(f"Subscription already exists: {subscription_id}")

        return subscription_path

    def subscribe(
        self,
        subscription_id: str,
        callback: Callable[[Dict[str, Any], Dict[str, str]], None],
        error_callback: Optional[Callable[[Exception], None]] = None,
    ):
        """Subscribe to messages with callback."""
        subscription_path = self.subscriber.subscription_path(
            self.config.project_id, subscription_id
        )

        def message_handler(message: pubsub_v1.subscriber.message.Message):
            try:
                # Decode message
                data = json.loads(message.data.decode("utf-8"))
                attributes = dict(message.attributes)

                # Process message
                callback(data, attributes)

                # Acknowledge
                message.ack()
                logger.debug(f"Processed message: {message.message_id}")

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                message.nack()
            except Exception as e:
                logger.error(f"Processing error: {e}")
                message.nack()

        # Start streaming pull
        self._streaming_pull_future = self.subscriber.subscribe(
            subscription_path,
            callback=message_handler,
            flow_control=self.flow_control,
        )

        logger.info(f"Listening on {subscription_id}...")

        try:
            self._streaming_pull_future.result()
        except Exception as e:
            if error_callback:
                error_callback(e)
            else:
                logger.error(f"Subscriber error: {e}")
                raise

    def pull_sync(
        self,
        subscription_id: str,
        max_messages: int = 10,
        timeout: float = 10.0,
    ) -> List[Dict[str, Any]]:
        """Synchronously pull messages."""
        subscription_path = self.subscriber.subscription_path(
            self.config.project_id, subscription_id
        )

        response = self.subscriber.pull(
            request={
                "subscription": subscription_path,
                "max_messages": max_messages,
            },
            timeout=timeout,
        )

        messages = []
        ack_ids = []

        for received in response.received_messages:
            try:
                data = json.loads(received.message.data.decode("utf-8"))
                messages.append({
                    "data": data,
                    "attributes": dict(received.message.attributes),
                    "message_id": received.message.message_id,
                    "publish_time": received.message.publish_time.isoformat(),
                })
                ack_ids.append(received.ack_id)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid message")

        # Acknowledge all
        if ack_ids:
            self.subscriber.acknowledge(
                request={
                    "subscription": subscription_path,
                    "ack_ids": ack_ids,
                }
            )

        return messages

    def stop(self):
        """Stop the subscriber."""
        if self._streaming_pull_future:
            self._streaming_pull_future.cancel()
            self._streaming_pull_future = None


# Message processor with retry decorator
def with_retry(max_retries: int = 3, backoff: float = 1.0):
    """Decorator for message processing with retry."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = backoff * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        raise
        return wrapper
    return decorator


# Usage
def main():
    config = PubSubConfig(project_id="my-project")

    # Publisher
    publisher = PubSubPublisher(config)
    publisher.create_topic("events")

    # Publish message
    publisher.publish(
        "events",
        {"type": "user_signup", "user_id": "123"},
        attributes={"source": "api", "version": "1.0"},
    )

    # Subscriber
    subscriber = PubSubSubscriber(config)
    subscriber.create_subscription(
        "events",
        "events-processor",
        dead_letter_topic="events-dead-letter",
    )

    # Process messages
    @with_retry(max_retries=3)
    def process_event(data: Dict, attributes: Dict):
        print(f"Processing: {data}")

    subscriber.subscribe("events-processor", process_event)


if __name__ == "__main__":
    main()
''',
                domain="gcp",
                subdomain="pubsub",
                tags=["messaging", "events", "streaming", "dead_letter"],
                difficulty="advanced"
            ),
        ]

    def _vertex_ai_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a Vertex AI pipeline for model training and deployment",
                code='''from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from typing import Optional, Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VertexAIManager:
    """Manager for Vertex AI operations."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        staging_bucket: str = None,
    ):
        self.project_id = project_id
        self.location = location
        self.staging_bucket = staging_bucket

        aiplatform.init(
            project=project_id,
            location=location,
            staging_bucket=staging_bucket,
        )

    # --- Dataset Management ---

    def create_tabular_dataset(
        self,
        display_name: str,
        gcs_source: str,
    ) -> aiplatform.TabularDataset:
        """Create a tabular dataset from GCS."""
        dataset = aiplatform.TabularDataset.create(
            display_name=display_name,
            gcs_source=gcs_source,
        )

        logger.info(f"Created dataset: {dataset.resource_name}")
        return dataset

    def create_image_dataset(
        self,
        display_name: str,
        gcs_source: str,
        import_schema_uri: str = None,
    ) -> aiplatform.ImageDataset:
        """Create an image dataset from GCS."""
        if not import_schema_uri:
            import_schema_uri = aiplatform.schema.dataset.ioformat.image.single_label_classification

        dataset = aiplatform.ImageDataset.create(
            display_name=display_name,
            gcs_source=gcs_source,
            import_schema_uri=import_schema_uri,
        )

        return dataset

    # --- Custom Training ---

    def run_custom_training(
        self,
        display_name: str,
        script_path: str,
        container_uri: str,
        dataset: aiplatform.TabularDataset = None,
        model_display_name: str = None,
        machine_type: str = "n1-standard-4",
        accelerator_type: str = None,
        accelerator_count: int = 0,
        args: List[str] = None,
        environment_variables: Dict[str, str] = None,
    ) -> aiplatform.Model:
        """Run a custom training job."""

        job = aiplatform.CustomTrainingJob(
            display_name=display_name,
            script_path=script_path,
            container_uri=container_uri,
            requirements=["pandas", "scikit-learn", "google-cloud-storage"],
            model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        )

        model = job.run(
            dataset=dataset,
            model_display_name=model_display_name or f"{display_name}-model",
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            args=args or [],
            environment_variables=environment_variables or {},
        )

        logger.info(f"Training complete: {model.resource_name}")
        return model

    def run_automl_training(
        self,
        display_name: str,
        dataset: aiplatform.TabularDataset,
        target_column: str,
        prediction_type: str = "classification",
        training_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        budget_milli_node_hours: int = 1000,
    ) -> aiplatform.Model:
        """Run AutoML training."""

        if prediction_type == "classification":
            job = aiplatform.AutoMLTabularTrainingJob(
                display_name=display_name,
                optimization_prediction_type="classification",
            )
        else:
            job = aiplatform.AutoMLTabularTrainingJob(
                display_name=display_name,
                optimization_prediction_type="regression",
            )

        model = job.run(
            dataset=dataset,
            target_column=target_column,
            training_fraction_split=training_fraction,
            validation_fraction_split=validation_fraction,
            test_fraction_split=test_fraction,
            budget_milli_node_hours=budget_milli_node_hours,
        )

        logger.info(f"AutoML training complete: {model.resource_name}")
        return model

    # --- Model Deployment ---

    def deploy_model(
        self,
        model: aiplatform.Model,
        endpoint_display_name: str = None,
        machine_type: str = "n1-standard-4",
        min_replica_count: int = 1,
        max_replica_count: int = 5,
        accelerator_type: str = None,
        accelerator_count: int = 0,
    ) -> aiplatform.Endpoint:
        """Deploy a model to an endpoint."""

        # Create or get endpoint
        if endpoint_display_name:
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_display_name}"'
            )
            if endpoints:
                endpoint = endpoints[0]
            else:
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_display_name,
                )
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=f"{model.display_name}-endpoint",
            )

        # Deploy model
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=model.display_name,
            machine_type=machine_type,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            traffic_percentage=100,
        )

        logger.info(f"Deployed to endpoint: {endpoint.resource_name}")
        return endpoint

    def batch_predict(
        self,
        model: aiplatform.Model,
        gcs_source: str,
        gcs_destination_prefix: str,
        machine_type: str = "n1-standard-4",
        starting_replica_count: int = 1,
        max_replica_count: int = 10,
    ) -> aiplatform.BatchPredictionJob:
        """Run batch prediction."""

        job = model.batch_predict(
            job_display_name=f"{model.display_name}-batch-predict",
            gcs_source=gcs_source,
            gcs_destination_prefix=gcs_destination_prefix,
            machine_type=machine_type,
            starting_replica_count=starting_replica_count,
            max_replica_count=max_replica_count,
        )

        job.wait()
        logger.info(f"Batch prediction complete: {job.resource_name}")

        return job

    # --- Online Prediction ---

    def predict(
        self,
        endpoint: aiplatform.Endpoint,
        instances: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Make online predictions."""

        predictions = endpoint.predict(instances=instances)

        return predictions.predictions

    # --- Model Monitoring ---

    def create_model_monitoring(
        self,
        endpoint: aiplatform.Endpoint,
        objective_configs: Dict[str, Any] = None,
        alert_config: Dict[str, Any] = None,
    ):
        """Set up model monitoring."""
        from google.cloud.aiplatform import model_monitoring

        # Default skew detection
        skew_config = model_monitoring.SkewDetectionConfig(
            data_source="gs://my-bucket/training-data.csv",
            skew_thresholds={
                "feature1": 0.3,
                "feature2": 0.3,
            },
        )

        # Default drift detection
        drift_config = model_monitoring.DriftDetectionConfig(
            drift_thresholds={
                "feature1": 0.3,
                "feature2": 0.3,
            },
        )

        objective_config = model_monitoring.ObjectiveConfig(
            skew_detection_config=skew_config,
            drift_detection_config=drift_config,
        )

        # Create monitoring job
        monitoring_job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name=f"{endpoint.display_name}-monitoring",
            endpoint=endpoint,
            objective_configs=objective_config,
            logging_sampling_strategy=model_monitoring.RandomSampleConfig(
                sample_rate=0.1
            ),
        )

        logger.info(f"Created monitoring job: {monitoring_job.resource_name}")
        return monitoring_job


# Training script example (train.py)
TRAINING_SCRIPT = """
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from google.cloud import storage
import pickle
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-column", default="target")
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.data_path)

    # Prepare features
    X = df.drop(columns=[args.target_column])
    y = df[args.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    print(f"Accuracy: {score:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
"""


# Usage
def main():
    vertex = VertexAIManager(
        project_id="my-project",
        staging_bucket="gs://my-staging-bucket",
    )

    # Create dataset
    dataset = vertex.create_tabular_dataset(
        display_name="customer-churn",
        gcs_source="gs://my-bucket/data/churn.csv",
    )

    # Run custom training
    model = vertex.run_custom_training(
        display_name="churn-predictor",
        script_path="train.py",
        container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
        dataset=dataset,
        machine_type="n1-standard-8",
        args=["--target-column", "churned"],
    )

    # Deploy model
    endpoint = vertex.deploy_model(
        model=model,
        endpoint_display_name="churn-predictor-endpoint",
        min_replica_count=1,
        max_replica_count=3,
    )

    # Make prediction
    predictions = vertex.predict(
        endpoint=endpoint,
        instances=[
            {"age": 35, "tenure": 24, "monthly_charges": 65.5},
            {"age": 42, "tenure": 12, "monthly_charges": 89.0},
        ],
    )

    print(f"Predictions: {predictions}")


if __name__ == "__main__":
    main()
''',
                domain="gcp",
                subdomain="vertex_ai",
                tags=["ml", "training", "deployment", "automl"],
                difficulty="advanced"
            ),
        ]
