"""
Azure Cloud Domain

Covers:
- Azure SDK (azure-* packages)
- Azure Functions
- Blob Storage, Cosmos DB, Service Bus
- Azure SQL, Azure Data Factory
- AKS (Azure Kubernetes Service)
- Azure ML
"""

from typing import List
from .base import BaseDomain, DomainExample


class AzureCloudDomain(BaseDomain):
    """Azure cloud services training examples."""

    def get_name(self) -> str:
        return "Azure Cloud"

    def get_description(self) -> str:
        return "Azure services including Functions, Blob Storage, Cosmos DB, and Azure ML"

    def get_subdomains(self) -> List[str]:
        return ["azure_functions", "blob_storage", "cosmos_db", "service_bus", "azure_ml"]

    def get_examples(self) -> List[DomainExample]:
        examples = []
        examples.extend(self._azure_functions_examples())
        examples.extend(self._blob_storage_examples())
        examples.extend(self._cosmos_db_examples())
        examples.extend(self._service_bus_examples())
        examples.extend(self._azure_ml_examples())
        return examples

    def _azure_functions_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an Azure Function with HTTP trigger and Cosmos DB binding",
                code='''import azure.functions as func
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

app = func.FunctionApp()


@app.route(route="items", methods=["GET", "POST"])
@app.cosmos_db_output(
    arg_name="outputDocument",
    database_name="MyDatabase",
    container_name="Items",
    connection="CosmosDBConnectionString"
)
def http_trigger(req: func.HttpRequest, outputDocument: func.Out[func.Document]) -> func.HttpResponse:
    """HTTP trigger function with Cosmos DB output binding."""
    logging.info("Processing HTTP request")

    try:
        if req.method == "GET":
            # Handle GET - return item info
            item_id = req.params.get("id")
            if not item_id:
                return func.HttpResponse(
                    json.dumps({"error": "Missing 'id' parameter"}),
                    status_code=400,
                    mimetype="application/json"
                )
            return func.HttpResponse(
                json.dumps({"message": f"Query for item {item_id}"}),
                mimetype="application/json"
            )

        elif req.method == "POST":
            # Handle POST - create new item
            try:
                body = req.get_json()
            except ValueError:
                return func.HttpResponse(
                    json.dumps({"error": "Invalid JSON body"}),
                    status_code=400,
                    mimetype="application/json"
                )

            # Create document
            document = {
                "id": body.get("id", str(datetime.utcnow().timestamp())),
                "name": body.get("name"),
                "category": body.get("category"),
                "createdAt": datetime.utcnow().isoformat(),
                "status": "active"
            }

            # Write to Cosmos DB via output binding
            outputDocument.set(func.Document.from_dict(document))

            return func.HttpResponse(
                json.dumps({"message": "Item created", "item": document}),
                status_code=201,
                mimetype="application/json"
            )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )


@app.route(route="items/{id}", methods=["GET", "PUT", "DELETE"])
@app.cosmos_db_input(
    arg_name="inputDocument",
    database_name="MyDatabase",
    container_name="Items",
    connection="CosmosDBConnectionString",
    id="{id}",
    partition_key="{id}"
)
def item_operations(
    req: func.HttpRequest,
    inputDocument: func.DocumentList
) -> func.HttpResponse:
    """CRUD operations on individual items."""
    item_id = req.route_params.get("id")

    if not inputDocument:
        return func.HttpResponse(
            json.dumps({"error": f"Item {item_id} not found"}),
            status_code=404,
            mimetype="application/json"
        )

    if req.method == "GET":
        return func.HttpResponse(
            json.dumps(inputDocument[0].to_dict()),
            mimetype="application/json"
        )

    elif req.method == "PUT":
        # Update logic would go here
        return func.HttpResponse(
            json.dumps({"message": "Updated"}),
            mimetype="application/json"
        )

    elif req.method == "DELETE":
        # Delete logic would go here
        return func.HttpResponse(
            json.dumps({"message": "Deleted"}),
            status_code=204,
            mimetype="application/json"
        )


@app.timer_trigger(
    arg_name="timer",
    schedule="0 */5 * * * *",  # Every 5 minutes
    run_on_startup=False
)
def timer_trigger(timer: func.TimerRequest) -> None:
    """Timer trigger for scheduled tasks."""
    if timer.past_due:
        logging.info("Timer is past due!")

    logging.info("Timer trigger function executed")

    # Perform scheduled task
    cleanup_expired_items()


def cleanup_expired_items():
    """Clean up expired items from database."""
    from azure.cosmos import CosmosClient
    import os

    connection_string = os.environ["CosmosDBConnectionString"]
    client = CosmosClient.from_connection_string(connection_string)

    database = client.get_database_client("MyDatabase")
    container = database.get_container_client("Items")

    # Query for expired items
    query = """
        SELECT * FROM c
        WHERE c.expiresAt < GetCurrentDateTime()
    """

    expired_items = list(container.query_items(query, enable_cross_partition_query=True))

    for item in expired_items:
        container.delete_item(item["id"], partition_key=item["id"])
        logging.info(f"Deleted expired item: {item['id']}")

    logging.info(f"Cleaned up {len(expired_items)} expired items")


@app.blob_trigger(
    arg_name="blob",
    path="uploads/{name}",
    connection="AzureWebJobsStorage"
)
@app.queue_output(
    arg_name="msg",
    queue_name="processing-queue",
    connection="AzureWebJobsStorage"
)
def blob_trigger(blob: func.InputStream, msg: func.Out[str]) -> None:
    """Blob trigger for file processing."""
    logging.info(f"Processing blob: {blob.name}, Size: {blob.length} bytes")

    # Read and process blob content
    content = blob.read()

    # Queue message for further processing
    message = {
        "blob_name": blob.name,
        "size": blob.length,
        "processed_at": datetime.utcnow().isoformat()
    }

    msg.set(json.dumps(message))
    logging.info(f"Queued processing message for {blob.name}")


@app.queue_trigger(
    arg_name="msg",
    queue_name="processing-queue",
    connection="AzureWebJobsStorage"
)
def queue_trigger(msg: func.QueueMessage) -> None:
    """Queue trigger for async processing."""
    logging.info(f"Processing queue message: {msg.id}")

    try:
        body = json.loads(msg.get_body().decode("utf-8"))
        logging.info(f"Message body: {body}")

        # Process the message
        process_queue_message(body)

    except Exception as e:
        logging.error(f"Error processing message: {str(e)}")
        raise  # Let the function fail to trigger retry


def process_queue_message(message: Dict[str, Any]) -> None:
    """Process a queue message."""
    blob_name = message.get("blob_name")
    logging.info(f"Processing blob: {blob_name}")
    # Add processing logic here
''',
                domain="azure",
                subdomain="azure_functions",
                tags=["serverless", "http", "cosmos_db", "timer", "queue"],
                difficulty="advanced"
            ),
        ]

    def _blob_storage_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an Azure Blob Storage manager with advanced operations",
                code='''from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    BlobType,
    ContentSettings,
    generate_blob_sas,
    generate_container_sas,
    BlobSasPermissions,
    ContainerSasPermissions,
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from typing import List, Dict, Any, Optional, BinaryIO, Generator
from datetime import datetime, timedelta
import os
import logging
import mimetypes
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureBlobManager:
    """Manager for Azure Blob Storage operations."""

    def __init__(
        self,
        connection_string: str = None,
        account_url: str = None,
        credential: Any = None,
    ):
        if connection_string:
            self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        elif account_url:
            credential = credential or DefaultAzureCredential()
            self.blob_service = BlobServiceClient(account_url, credential=credential)
        else:
            raise ValueError("Either connection_string or account_url required")

        self.account_name = self.blob_service.account_name

    # --- Container Operations ---

    def create_container(
        self,
        container_name: str,
        public_access: str = None,
        metadata: Dict[str, str] = None,
    ) -> ContainerClient:
        """Create a container."""
        try:
            container = self.blob_service.create_container(
                container_name,
                public_access=public_access,
                metadata=metadata,
            )
            logger.info(f"Created container: {container_name}")
            return container
        except ResourceExistsError:
            logger.info(f"Container already exists: {container_name}")
            return self.blob_service.get_container_client(container_name)

    def delete_container(self, container_name: str):
        """Delete a container."""
        self.blob_service.delete_container(container_name)
        logger.info(f"Deleted container: {container_name}")

    def list_containers(self, prefix: str = None) -> List[Dict[str, Any]]:
        """List all containers."""
        containers = []
        for container in self.blob_service.list_containers(name_starts_with=prefix):
            containers.append({
                "name": container.name,
                "last_modified": container.last_modified.isoformat(),
                "metadata": container.metadata,
            })
        return containers

    # --- Blob Upload Operations ---

    def upload_file(
        self,
        container_name: str,
        local_path: str,
        blob_name: str = None,
        overwrite: bool = True,
        content_type: str = None,
        metadata: Dict[str, str] = None,
    ) -> str:
        """Upload a file to blob storage."""
        if not blob_name:
            blob_name = Path(local_path).name

        # Auto-detect content type
        if not content_type:
            content_type, _ = mimetypes.guess_type(local_path)

        content_settings = ContentSettings(content_type=content_type)

        blob_client = self.blob_service.get_blob_client(container_name, blob_name)

        with open(local_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=overwrite,
                content_settings=content_settings,
                metadata=metadata,
            )

        url = blob_client.url
        logger.info(f"Uploaded {local_path} to {url}")
        return url

    def upload_data(
        self,
        container_name: str,
        blob_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> str:
        """Upload bytes data to blob storage."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)

        content_settings = ContentSettings(content_type=content_type)

        blob_client.upload_blob(
            data,
            overwrite=overwrite,
            content_settings=content_settings,
        )

        return blob_client.url

    def upload_stream(
        self,
        container_name: str,
        blob_name: str,
        stream: BinaryIO,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload from a stream."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)

        blob_client.upload_blob(
            stream,
            blob_type=BlobType.BlockBlob,
            content_settings=ContentSettings(content_type=content_type),
        )

        return blob_client.url

    def upload_directory(
        self,
        container_name: str,
        local_dir: str,
        blob_prefix: str = "",
        max_concurrency: int = 4,
    ) -> List[str]:
        """Upload an entire directory."""
        uploaded = []
        local_path = Path(local_dir)

        from concurrent.futures import ThreadPoolExecutor

        def upload_single(file_path: Path):
            relative = file_path.relative_to(local_path)
            blob_name = f"{blob_prefix}/{relative}" if blob_prefix else str(relative)
            return self.upload_file(container_name, str(file_path), blob_name)

        files = [f for f in local_path.rglob("*") if f.is_file()]

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            uploaded = list(executor.map(upload_single, files))

        logger.info(f"Uploaded {len(uploaded)} files from {local_dir}")
        return uploaded

    # --- Blob Download Operations ---

    def download_file(
        self,
        container_name: str,
        blob_name: str,
        local_path: str,
    ):
        """Download a blob to a local file."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)

        with open(local_path, "wb") as f:
            stream = blob_client.download_blob()
            f.write(stream.readall())

        logger.info(f"Downloaded {blob_name} to {local_path}")

    def download_data(self, container_name: str, blob_name: str) -> bytes:
        """Download blob as bytes."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)
        return blob_client.download_blob().readall()

    def download_stream(
        self,
        container_name: str,
        blob_name: str,
    ) -> Generator[bytes, None, None]:
        """Download blob as a stream (for large files)."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)
        stream = blob_client.download_blob()

        for chunk in stream.chunks():
            yield chunk

    # --- Blob Management ---

    def list_blobs(
        self,
        container_name: str,
        prefix: str = None,
        include_metadata: bool = False,
    ) -> List[Dict[str, Any]]:
        """List blobs in a container."""
        container = self.blob_service.get_container_client(container_name)

        blobs = []
        include = ["metadata"] if include_metadata else None

        for blob in container.list_blobs(name_starts_with=prefix, include=include):
            blobs.append({
                "name": blob.name,
                "size": blob.size,
                "content_type": blob.content_settings.content_type,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                "etag": blob.etag,
                "metadata": blob.metadata if include_metadata else None,
            })

        return blobs

    def delete_blob(self, container_name: str, blob_name: str):
        """Delete a blob."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)
        blob_client.delete_blob()
        logger.info(f"Deleted blob: {blob_name}")

    def delete_blobs(self, container_name: str, prefix: str):
        """Delete all blobs with a prefix."""
        container = self.blob_service.get_container_client(container_name)

        blobs = list(container.list_blobs(name_starts_with=prefix))

        for blob in blobs:
            container.delete_blob(blob.name)
            logger.info(f"Deleted: {blob.name}")

        logger.info(f"Deleted {len(blobs)} blobs with prefix: {prefix}")

    def copy_blob(
        self,
        source_container: str,
        source_blob: str,
        dest_container: str,
        dest_blob: str,
    ) -> str:
        """Copy a blob within the same account."""
        source_client = self.blob_service.get_blob_client(source_container, source_blob)
        dest_client = self.blob_service.get_blob_client(dest_container, dest_blob)

        dest_client.start_copy_from_url(source_client.url)

        return dest_client.url

    # --- SAS Token Generation ---

    def generate_blob_sas_url(
        self,
        container_name: str,
        blob_name: str,
        expiry_hours: int = 1,
        permission: str = "r",
    ) -> str:
        """Generate a SAS URL for a blob."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)

        # Get account key from connection string
        account_key = self._get_account_key()

        permissions = BlobSasPermissions(
            read="r" in permission,
            write="w" in permission,
            delete="d" in permission,
            add="a" in permission,
            create="c" in permission,
        )

        sas_token = generate_blob_sas(
            account_name=self.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=permissions,
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )

        return f"{blob_client.url}?{sas_token}"

    def generate_container_sas_url(
        self,
        container_name: str,
        expiry_hours: int = 24,
        permission: str = "rl",
    ) -> str:
        """Generate a SAS URL for a container."""
        account_key = self._get_account_key()

        permissions = ContainerSasPermissions(
            read="r" in permission,
            write="w" in permission,
            delete="d" in permission,
            list="l" in permission,
        )

        sas_token = generate_container_sas(
            account_name=self.account_name,
            container_name=container_name,
            account_key=account_key,
            permission=permissions,
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )

        container_url = f"https://{self.account_name}.blob.core.windows.net/{container_name}"
        return f"{container_url}?{sas_token}"

    def _get_account_key(self) -> str:
        """Extract account key from connection string."""
        # Parse from connection string
        conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
        for part in conn_str.split(";"):
            if part.startswith("AccountKey="):
                return part.split("=", 1)[1]
        raise ValueError("Account key not found")

    # --- Blob Properties ---

    def get_blob_properties(
        self,
        container_name: str,
        blob_name: str,
    ) -> Dict[str, Any]:
        """Get blob properties and metadata."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)
        props = blob_client.get_blob_properties()

        return {
            "name": props.name,
            "size": props.size,
            "content_type": props.content_settings.content_type,
            "content_encoding": props.content_settings.content_encoding,
            "last_modified": props.last_modified.isoformat(),
            "etag": props.etag,
            "metadata": props.metadata,
            "blob_type": props.blob_type,
            "lease_status": props.lease.status,
        }

    def set_blob_metadata(
        self,
        container_name: str,
        blob_name: str,
        metadata: Dict[str, str],
    ):
        """Set blob metadata."""
        blob_client = self.blob_service.get_blob_client(container_name, blob_name)
        blob_client.set_blob_metadata(metadata)


# Usage
def main():
    manager = AzureBlobManager(
        connection_string=os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    )

    # Create container
    manager.create_container("my-data", public_access=None)

    # Upload file
    url = manager.upload_file(
        "my-data",
        "/local/path/file.json",
        "data/file.json",
        metadata={"version": "1.0"}
    )

    # Generate SAS URL for download
    sas_url = manager.generate_blob_sas_url(
        "my-data",
        "data/file.json",
        expiry_hours=24,
        permission="r"
    )
    print(f"SAS URL: {sas_url}")

    # List blobs
    blobs = manager.list_blobs("my-data", prefix="data/")
    for blob in blobs:
        print(f"{blob['name']}: {blob['size']} bytes")


if __name__ == "__main__":
    main()
''',
                domain="azure",
                subdomain="blob_storage",
                tags=["storage", "sas", "upload", "download"],
                difficulty="intermediate"
            ),
        ]

    def _cosmos_db_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an Azure Cosmos DB manager with CRUD and query operations",
                code='''from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.identity import DefaultAzureCredential
from typing import Dict, List, Any, Optional, Iterator
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosmosDBManager:
    """Manager for Azure Cosmos DB operations."""

    def __init__(
        self,
        endpoint: str = None,
        key: str = None,
        connection_string: str = None,
    ):
        if connection_string:
            self.client = CosmosClient.from_connection_string(connection_string)
        elif endpoint and key:
            self.client = CosmosClient(endpoint, key)
        elif endpoint:
            credential = DefaultAzureCredential()
            self.client = CosmosClient(endpoint, credential)
        else:
            raise ValueError("Endpoint/key or connection_string required")

    # --- Database Operations ---

    def create_database(self, database_name: str) -> Any:
        """Create a database if it doesn't exist."""
        try:
            database = self.client.create_database(database_name)
            logger.info(f"Created database: {database_name}")
        except exceptions.CosmosResourceExistsError:
            database = self.client.get_database_client(database_name)
            logger.info(f"Database already exists: {database_name}")

        return database

    def delete_database(self, database_name: str):
        """Delete a database."""
        self.client.delete_database(database_name)
        logger.info(f"Deleted database: {database_name}")

    # --- Container Operations ---

    def create_container(
        self,
        database_name: str,
        container_name: str,
        partition_key: str,
        throughput: int = 400,
        unique_keys: List[str] = None,
        default_ttl: int = None,
    ) -> Any:
        """Create a container with configuration."""
        database = self.client.get_database_client(database_name)

        partition_key_path = PartitionKey(path=f"/{partition_key}")

        # Configure unique keys
        unique_key_policy = None
        if unique_keys:
            unique_key_policy = {
                "uniqueKeys": [{"paths": [f"/{k}" for k in unique_keys]}]
            }

        try:
            container = database.create_container(
                id=container_name,
                partition_key=partition_key_path,
                offer_throughput=throughput,
                unique_key_policy=unique_key_policy,
                default_ttl=default_ttl,
            )
            logger.info(f"Created container: {container_name}")
        except exceptions.CosmosResourceExistsError:
            container = database.get_container_client(container_name)
            logger.info(f"Container already exists: {container_name}")

        return container

    # --- Document CRUD Operations ---

    def create_item(
        self,
        database_name: str,
        container_name: str,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a document."""
        container = self._get_container(database_name, container_name)

        # Add timestamp
        item["_createdAt"] = datetime.utcnow().isoformat()
        item["_updatedAt"] = item["_createdAt"]

        result = container.create_item(body=item)
        logger.info(f"Created item: {result['id']}")

        return result

    def upsert_item(
        self,
        database_name: str,
        container_name: str,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create or update a document."""
        container = self._get_container(database_name, container_name)

        item["_updatedAt"] = datetime.utcnow().isoformat()

        result = container.upsert_item(body=item)
        logger.info(f"Upserted item: {result['id']}")

        return result

    def read_item(
        self,
        database_name: str,
        container_name: str,
        item_id: str,
        partition_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Read a document by ID."""
        container = self._get_container(database_name, container_name)

        try:
            return container.read_item(item_id, partition_key=partition_key)
        except exceptions.CosmosResourceNotFoundError:
            return None

    def replace_item(
        self,
        database_name: str,
        container_name: str,
        item_id: str,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Replace a document."""
        container = self._get_container(database_name, container_name)

        item["_updatedAt"] = datetime.utcnow().isoformat()

        return container.replace_item(item_id, body=item)

    def patch_item(
        self,
        database_name: str,
        container_name: str,
        item_id: str,
        partition_key: str,
        operations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Patch a document with partial updates."""
        container = self._get_container(database_name, container_name)

        # Add timestamp update
        operations.append({
            "op": "set",
            "path": "/_updatedAt",
            "value": datetime.utcnow().isoformat()
        })

        return container.patch_item(
            item=item_id,
            partition_key=partition_key,
            patch_operations=operations
        )

    def delete_item(
        self,
        database_name: str,
        container_name: str,
        item_id: str,
        partition_key: str,
    ):
        """Delete a document."""
        container = self._get_container(database_name, container_name)

        container.delete_item(item_id, partition_key=partition_key)
        logger.info(f"Deleted item: {item_id}")

    # --- Query Operations ---

    def query_items(
        self,
        database_name: str,
        container_name: str,
        query: str,
        parameters: List[Dict[str, Any]] = None,
        partition_key: str = None,
        max_items: int = None,
    ) -> Iterator[Dict[str, Any]]:
        """Query documents with SQL-like syntax."""
        container = self._get_container(database_name, container_name)

        kwargs = {
            "query": query,
            "parameters": parameters or [],
        }

        if partition_key:
            kwargs["partition_key"] = partition_key
        else:
            kwargs["enable_cross_partition_query"] = True

        if max_items:
            kwargs["max_item_count"] = max_items

        return container.query_items(**kwargs)

    def query_all(
        self,
        database_name: str,
        container_name: str,
        query: str,
        parameters: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query and return all matching documents."""
        return list(self.query_items(
            database_name, container_name, query, parameters
        ))

    # --- Batch Operations ---

    def bulk_upsert(
        self,
        database_name: str,
        container_name: str,
        items: List[Dict[str, Any]],
        partition_key_path: str,
    ) -> List[Dict[str, Any]]:
        """Bulk upsert documents (grouped by partition key)."""
        container = self._get_container(database_name, container_name)

        results = []
        timestamp = datetime.utcnow().isoformat()

        for item in items:
            item["_updatedAt"] = timestamp
            try:
                result = container.upsert_item(body=item)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to upsert {item.get('id')}: {e}")

        logger.info(f"Bulk upserted {len(results)}/{len(items)} items")
        return results

    def transactional_batch(
        self,
        database_name: str,
        container_name: str,
        partition_key: str,
        operations: List[Dict[str, Any]],
    ):
        """Execute a transactional batch (all or nothing)."""
        container = self._get_container(database_name, container_name)

        batch = []
        for op in operations:
            if op["type"] == "create":
                batch.append(("create", (op["body"],), {}))
            elif op["type"] == "upsert":
                batch.append(("upsert", (op["body"],), {}))
            elif op["type"] == "replace":
                batch.append(("replace", (op["id"], op["body"]), {}))
            elif op["type"] == "delete":
                batch.append(("delete", (op["id"],), {}))

        return container.execute_item_batch(
            batch_operations=batch,
            partition_key=partition_key
        )

    # --- Change Feed ---

    def read_change_feed(
        self,
        database_name: str,
        container_name: str,
        partition_key: str = None,
        start_time: datetime = None,
    ) -> Iterator[Dict[str, Any]]:
        """Read change feed for real-time sync."""
        container = self._get_container(database_name, container_name)

        kwargs = {}
        if partition_key:
            kwargs["partition_key"] = partition_key

        if start_time:
            kwargs["start_time"] = start_time
        else:
            kwargs["is_start_from_beginning"] = True

        return container.query_items_change_feed(**kwargs)

    # --- Stored Procedures ---

    def create_stored_procedure(
        self,
        database_name: str,
        container_name: str,
        sproc_id: str,
        body: str,
    ):
        """Create a stored procedure."""
        container = self._get_container(database_name, container_name)

        sproc_definition = {
            "id": sproc_id,
            "body": body,
        }

        try:
            container.scripts.create_stored_procedure(sproc_definition)
            logger.info(f"Created stored procedure: {sproc_id}")
        except exceptions.CosmosResourceExistsError:
            container.scripts.replace_stored_procedure(sproc_id, sproc_definition)
            logger.info(f"Updated stored procedure: {sproc_id}")

    def execute_stored_procedure(
        self,
        database_name: str,
        container_name: str,
        sproc_id: str,
        partition_key: str,
        params: List[Any] = None,
    ) -> Any:
        """Execute a stored procedure."""
        container = self._get_container(database_name, container_name)

        return container.scripts.execute_stored_procedure(
            sproc=sproc_id,
            params=params or [],
            partition_key=partition_key,
        )

    # --- Helper Methods ---

    def _get_container(self, database_name: str, container_name: str):
        """Get container client."""
        database = self.client.get_database_client(database_name)
        return database.get_container_client(container_name)


# Example stored procedure for bulk delete
BULK_DELETE_SPROC = """
function bulkDelete(ids) {
    var context = getContext();
    var container = context.getCollection();
    var response = context.getResponse();

    if (!ids || ids.length === 0) {
        response.setBody({deleted: 0});
        return;
    }

    var deleted = 0;
    var index = 0;

    function deleteNext() {
        if (index >= ids.length) {
            response.setBody({deleted: deleted});
            return;
        }

        var isAccepted = container.deleteDocument(
            container.getAltLink() + '/docs/' + ids[index],
            {},
            function(err) {
                if (err) throw err;
                deleted++;
                index++;
                deleteNext();
            }
        );

        if (!isAccepted) {
            response.setBody({deleted: deleted, continuation: ids.slice(index)});
        }
    }

    deleteNext();
}
"""


# Usage
def main():
    cosmos = CosmosDBManager(
        endpoint=os.environ["COSMOS_ENDPOINT"],
        key=os.environ["COSMOS_KEY"]
    )

    # Create database and container
    cosmos.create_database("MyApp")
    cosmos.create_container(
        "MyApp", "Users",
        partition_key="tenantId",
        throughput=400,
        unique_keys=["email"]
    )

    # Create user
    user = cosmos.create_item("MyApp", "Users", {
        "id": "user-123",
        "tenantId": "tenant-1",
        "email": "user@example.com",
        "name": "John Doe",
        "role": "admin"
    })

    # Query users
    admins = cosmos.query_all(
        "MyApp", "Users",
        "SELECT * FROM c WHERE c.role = @role",
        [{"name": "@role", "value": "admin"}]
    )

    print(f"Found {len(admins)} admins")


if __name__ == "__main__":
    main()
''',
                domain="azure",
                subdomain="cosmos_db",
                tags=["nosql", "database", "query", "stored_procedure"],
                difficulty="advanced"
            ),
        ]

    def _service_bus_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an Azure Service Bus manager with queues and topics",
                code='''from azure.servicebus import ServiceBusClient, ServiceBusMessage, ServiceBusSender
from azure.servicebus.management import ServiceBusAdministrationClient
from azure.identity import DefaultAzureCredential
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MessagePayload:
    """Standard message payload."""
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: str = None
    correlation_id: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class ServiceBusManager:
    """Manager for Azure Service Bus operations."""

    def __init__(
        self,
        connection_string: str = None,
        namespace: str = None,
    ):
        if connection_string:
            self.client = ServiceBusClient.from_connection_string(connection_string)
            self.admin = ServiceBusAdministrationClient.from_connection_string(connection_string)
        elif namespace:
            credential = DefaultAzureCredential()
            self.client = ServiceBusClient(namespace, credential)
            self.admin = ServiceBusAdministrationClient(namespace, credential)
        else:
            raise ValueError("connection_string or namespace required")

    # --- Queue Management ---

    def create_queue(
        self,
        queue_name: str,
        max_size_mb: int = 1024,
        default_message_ttl: timedelta = timedelta(days=14),
        lock_duration: timedelta = timedelta(seconds=60),
        dead_lettering: bool = True,
        max_delivery_count: int = 10,
    ):
        """Create a queue with configuration."""
        try:
            self.admin.create_queue(
                queue_name,
                max_size_in_megabytes=max_size_mb,
                default_message_time_to_live=default_message_ttl,
                lock_duration=lock_duration,
                dead_lettering_on_message_expiration=dead_lettering,
                max_delivery_count=max_delivery_count,
            )
            logger.info(f"Created queue: {queue_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Queue already exists: {queue_name}")
            else:
                raise

    def delete_queue(self, queue_name: str):
        """Delete a queue."""
        self.admin.delete_queue(queue_name)
        logger.info(f"Deleted queue: {queue_name}")

    # --- Topic Management ---

    def create_topic(
        self,
        topic_name: str,
        max_size_mb: int = 1024,
        default_message_ttl: timedelta = timedelta(days=14),
    ):
        """Create a topic."""
        try:
            self.admin.create_topic(
                topic_name,
                max_size_in_megabytes=max_size_mb,
                default_message_time_to_live=default_message_ttl,
            )
            logger.info(f"Created topic: {topic_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Topic already exists: {topic_name}")
            else:
                raise

    def create_subscription(
        self,
        topic_name: str,
        subscription_name: str,
        max_delivery_count: int = 10,
        lock_duration: timedelta = timedelta(seconds=60),
        filter_rule: str = None,
    ):
        """Create a subscription with optional filter."""
        try:
            self.admin.create_subscription(
                topic_name,
                subscription_name,
                max_delivery_count=max_delivery_count,
                lock_duration=lock_duration,
            )
            logger.info(f"Created subscription: {subscription_name}")

            # Add SQL filter if provided
            if filter_rule:
                from azure.servicebus.management import SqlRuleFilter

                self.admin.create_rule(
                    topic_name,
                    subscription_name,
                    "CustomFilter",
                    filter=SqlRuleFilter(filter_rule)
                )
                # Delete the default "true" filter
                try:
                    self.admin.delete_rule(topic_name, subscription_name, "$Default")
                except:
                    pass

        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Subscription already exists: {subscription_name}")
            else:
                raise

    # --- Sending Messages ---

    def send_message(
        self,
        queue_or_topic: str,
        payload: MessagePayload,
        session_id: str = None,
        scheduled_time: datetime = None,
        properties: Dict[str, Any] = None,
    ) -> str:
        """Send a single message."""
        with self.client.get_queue_sender(queue_or_topic) as sender:
            message = ServiceBusMessage(
                body=json.dumps(asdict(payload)),
                session_id=session_id,
                application_properties=properties or {},
            )

            if scheduled_time:
                message.scheduled_enqueue_time_utc = scheduled_time

            sender.send_messages(message)

            logger.debug(f"Sent message: {payload.id}")
            return payload.id

    def send_batch(
        self,
        queue_or_topic: str,
        payloads: List[MessagePayload],
    ) -> int:
        """Send a batch of messages efficiently."""
        with self.client.get_queue_sender(queue_or_topic) as sender:
            batch = sender.create_message_batch()
            sent_count = 0

            for payload in payloads:
                message = ServiceBusMessage(
                    body=json.dumps(asdict(payload))
                )

                try:
                    batch.add_message(message)
                except ValueError:
                    # Batch is full, send and create new batch
                    sender.send_messages(batch)
                    sent_count += len(batch)
                    batch = sender.create_message_batch()
                    batch.add_message(message)

            # Send remaining messages
            if len(batch) > 0:
                sender.send_messages(batch)
                sent_count += len(batch)

            logger.info(f"Sent batch of {sent_count} messages")
            return sent_count

    def publish_to_topic(
        self,
        topic_name: str,
        payload: MessagePayload,
        routing_properties: Dict[str, Any] = None,
    ):
        """Publish message to a topic with routing properties."""
        with self.client.get_topic_sender(topic_name) as sender:
            message = ServiceBusMessage(
                body=json.dumps(asdict(payload)),
                application_properties=routing_properties or {},
            )

            sender.send_messages(message)
            logger.debug(f"Published to topic: {payload.id}")

    # --- Receiving Messages ---

    def receive_messages(
        self,
        queue_name: str,
        max_messages: int = 10,
        max_wait_time: float = 5.0,
        auto_complete: bool = False,
    ) -> List[Dict[str, Any]]:
        """Receive messages from a queue."""
        with self.client.get_queue_receiver(queue_name) as receiver:
            messages = receiver.receive_messages(
                max_message_count=max_messages,
                max_wait_time=max_wait_time,
            )

            results = []
            for msg in messages:
                try:
                    payload = json.loads(str(msg))
                    results.append({
                        "message_id": msg.message_id,
                        "payload": payload,
                        "enqueued_time": msg.enqueued_time_utc.isoformat() if msg.enqueued_time_utc else None,
                        "delivery_count": msg.delivery_count,
                    })

                    if auto_complete:
                        receiver.complete_message(msg)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in message: {msg.message_id}")
                    receiver.dead_letter_message(msg, reason="InvalidJSON")

            return results

    def process_messages(
        self,
        queue_name: str,
        handler: Callable[[Dict[str, Any]], bool],
        max_messages: int = 1,
        max_wait_time: float = 30.0,
    ):
        """Continuously process messages with a handler."""
        with self.client.get_queue_receiver(queue_name) as receiver:
            while True:
                messages = receiver.receive_messages(
                    max_message_count=max_messages,
                    max_wait_time=max_wait_time,
                )

                if not messages:
                    continue

                for msg in messages:
                    try:
                        payload = json.loads(str(msg))

                        # Call handler
                        success = handler(payload)

                        if success:
                            receiver.complete_message(msg)
                            logger.info(f"Processed: {msg.message_id}")
                        else:
                            receiver.abandon_message(msg)
                            logger.warning(f"Abandoned: {msg.message_id}")

                    except Exception as e:
                        logger.error(f"Error processing {msg.message_id}: {e}")
                        receiver.abandon_message(msg)

    def receive_from_subscription(
        self,
        topic_name: str,
        subscription_name: str,
        max_messages: int = 10,
    ) -> List[Dict[str, Any]]:
        """Receive messages from a topic subscription."""
        with self.client.get_subscription_receiver(topic_name, subscription_name) as receiver:
            messages = receiver.receive_messages(max_message_count=max_messages)

            results = []
            for msg in messages:
                payload = json.loads(str(msg))
                results.append(payload)
                receiver.complete_message(msg)

            return results

    # --- Dead Letter Queue ---

    def receive_dead_letters(
        self,
        queue_name: str,
        max_messages: int = 10,
    ) -> List[Dict[str, Any]]:
        """Receive messages from dead letter queue."""
        dlq_path = f"{queue_name}/$deadletterqueue"

        with self.client.get_queue_receiver(dlq_path) as receiver:
            messages = receiver.receive_messages(max_message_count=max_messages)

            results = []
            for msg in messages:
                results.append({
                    "message_id": msg.message_id,
                    "body": str(msg),
                    "dead_letter_reason": msg.dead_letter_reason,
                    "dead_letter_description": msg.dead_letter_error_description,
                    "delivery_count": msg.delivery_count,
                })

            return results

    def requeue_dead_letter(
        self,
        queue_name: str,
        message_id: str,
    ):
        """Move a dead letter message back to the main queue."""
        dlq_path = f"{queue_name}/$deadletterqueue"

        with self.client.get_queue_receiver(dlq_path) as receiver:
            messages = receiver.receive_messages(max_message_count=100)

            for msg in messages:
                if msg.message_id == message_id:
                    # Resend to main queue
                    with self.client.get_queue_sender(queue_name) as sender:
                        new_msg = ServiceBusMessage(body=str(msg))
                        sender.send_messages(new_msg)

                    receiver.complete_message(msg)
                    logger.info(f"Requeued message: {message_id}")
                    return

        logger.warning(f"Message not found: {message_id}")

    # --- Session Support ---

    def send_session_message(
        self,
        queue_name: str,
        session_id: str,
        payload: MessagePayload,
    ):
        """Send a message to a specific session."""
        with self.client.get_queue_sender(queue_name) as sender:
            message = ServiceBusMessage(
                body=json.dumps(asdict(payload)),
                session_id=session_id,
            )
            sender.send_messages(message)

    def receive_session_messages(
        self,
        queue_name: str,
        session_id: str = None,
        max_messages: int = 10,
    ) -> List[Dict[str, Any]]:
        """Receive messages from a session."""
        with self.client.get_queue_receiver(queue_name, session_id=session_id) as receiver:
            messages = receiver.receive_messages(max_message_count=max_messages)

            results = []
            for msg in messages:
                payload = json.loads(str(msg))
                results.append({
                    "session_id": msg.session_id,
                    "payload": payload,
                })
                receiver.complete_message(msg)

            return results


# Usage
def main():
    sb = ServiceBusManager(
        connection_string=os.environ["SERVICE_BUS_CONNECTION_STRING"]
    )

    # Create queue
    sb.create_queue("orders", max_delivery_count=5)

    # Create topic with subscriptions
    sb.create_topic("events")
    sb.create_subscription("events", "all-events")
    sb.create_subscription(
        "events", "priority-events",
        filter_rule="priority = 'high'"
    )

    # Send message
    payload = MessagePayload(
        id="order-123",
        type="order.created",
        data={"product": "Widget", "quantity": 5}
    )
    sb.send_message("orders", payload)

    # Process messages
    def handle_order(data: Dict) -> bool:
        print(f"Processing order: {data}")
        return True

    # This would run continuously
    # sb.process_messages("orders", handle_order)


if __name__ == "__main__":
    main()
''',
                domain="azure",
                subdomain="service_bus",
                tags=["messaging", "queue", "topic", "pubsub"],
                difficulty="advanced"
            ),
        ]

    def _azure_ml_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an Azure ML pipeline for model training and deployment",
                code='''from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import (
    AmlCompute,
    Environment,
    Model,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureMLManager:
    """Manager for Azure Machine Learning operations."""

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
    ):
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id,
            resource_group,
            workspace_name,
        )

    # --- Compute Management ---

    def create_compute_cluster(
        self,
        name: str,
        vm_size: str = "Standard_DS3_v2",
        min_instances: int = 0,
        max_instances: int = 4,
        idle_time_before_scale_down: int = 120,
    ) -> AmlCompute:
        """Create a compute cluster for training."""
        try:
            compute = self.ml_client.compute.get(name)
            logger.info(f"Compute cluster exists: {name}")
            return compute
        except Exception:
            pass

        compute = AmlCompute(
            name=name,
            size=vm_size,
            min_instances=min_instances,
            max_instances=max_instances,
            idle_time_before_scale_down=idle_time_before_scale_down,
        )

        self.ml_client.compute.begin_create_or_update(compute).result()
        logger.info(f"Created compute cluster: {name}")

        return self.ml_client.compute.get(name)

    # --- Environment Management ---

    def create_environment(
        self,
        name: str,
        conda_file: str = None,
        docker_image: str = None,
        pip_requirements: list = None,
    ) -> Environment:
        """Create a training environment."""
        if conda_file:
            env = Environment(
                name=name,
                conda_file=conda_file,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            )
        elif docker_image:
            env = Environment(
                name=name,
                image=docker_image,
            )
        else:
            # Create from pip requirements
            conda_content = f"""
name: {name}
dependencies:
  - python=3.10
  - pip:
    - {chr(10).join(f'    - {pkg}' for pkg in (pip_requirements or ['scikit-learn', 'pandas', 'numpy']))}
"""
            env = Environment(
                name=name,
                conda_file=conda_content,
                image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            )

        env = self.ml_client.environments.create_or_update(env)
        logger.info(f"Created environment: {name}")

        return env

    # --- Data Management ---

    def register_data_asset(
        self,
        name: str,
        path: str,
        asset_type: str = "uri_folder",
        description: str = None,
    ):
        """Register a data asset."""
        from azure.ai.ml.entities import Data

        data = Data(
            name=name,
            path=path,
            type=asset_type,
            description=description,
        )

        data = self.ml_client.data.create_or_update(data)
        logger.info(f"Registered data asset: {name}")

        return data

    # --- Training Jobs ---

    def submit_training_job(
        self,
        name: str,
        script_path: str,
        compute_name: str,
        environment_name: str,
        inputs: Dict[str, str] = None,
        outputs: Dict[str, str] = None,
        arguments: list = None,
    ):
        """Submit a training job."""
        # Build inputs
        job_inputs = {}
        if inputs:
            for key, path in inputs.items():
                job_inputs[key] = Input(type=AssetTypes.URI_FOLDER, path=path)

        # Build outputs
        job_outputs = {}
        if outputs:
            for key, path in outputs.items():
                job_outputs[key] = Output(type=AssetTypes.URI_FOLDER, path=path)

        job = command(
            name=name,
            display_name=name,
            code="./src",
            command=f"python {script_path} " + " ".join(arguments or []),
            environment=f"{environment_name}@latest",
            compute=compute_name,
            inputs=job_inputs,
            outputs=job_outputs,
        )

        returned_job = self.ml_client.jobs.create_or_update(job)
        logger.info(f"Submitted job: {returned_job.name}")

        return returned_job

    def wait_for_job(self, job_name: str) -> str:
        """Wait for a job to complete."""
        from azure.ai.ml.entities import Job

        job = self.ml_client.jobs.get(job_name)

        # Poll until complete
        import time
        while job.status not in ["Completed", "Failed", "Canceled"]:
            time.sleep(30)
            job = self.ml_client.jobs.get(job_name)
            logger.info(f"Job {job_name} status: {job.status}")

        return job.status

    # --- Model Management ---

    def register_model(
        self,
        name: str,
        path: str,
        model_type: str = "custom_model",
        description: str = None,
        properties: Dict[str, str] = None,
    ) -> Model:
        """Register a trained model."""
        model = Model(
            name=name,
            path=path,
            type=model_type,
            description=description,
            properties=properties or {},
        )

        model = self.ml_client.models.create_or_update(model)
        logger.info(f"Registered model: {name} (version {model.version})")

        return model

    def get_model(self, name: str, version: str = None) -> Model:
        """Get a registered model."""
        if version:
            return self.ml_client.models.get(name, version)
        else:
            # Get latest version
            models = list(self.ml_client.models.list(name))
            if not models:
                raise ValueError(f"Model not found: {name}")
            return max(models, key=lambda m: int(m.version))

    # --- Deployment ---

    def create_online_endpoint(
        self,
        name: str,
        auth_mode: str = "key",
    ) -> ManagedOnlineEndpoint:
        """Create an online endpoint for model serving."""
        endpoint = ManagedOnlineEndpoint(
            name=name,
            auth_mode=auth_mode,
        )

        endpoint = self.ml_client.online_endpoints.begin_create_or_update(
            endpoint
        ).result()

        logger.info(f"Created endpoint: {name}")
        return endpoint

    def deploy_model(
        self,
        endpoint_name: str,
        deployment_name: str,
        model_name: str,
        model_version: str = None,
        instance_type: str = "Standard_DS2_v2",
        instance_count: int = 1,
        scoring_script: str = "score.py",
        environment_name: str = None,
    ) -> ManagedOnlineDeployment:
        """Deploy a model to an endpoint."""
        # Get model
        model = self.get_model(model_name, model_version)

        # Create deployment
        deployment = ManagedOnlineDeployment(
            name=deployment_name,
            endpoint_name=endpoint_name,
            model=model,
            code_configuration=CodeConfiguration(
                code="./src",
                scoring_script=scoring_script,
            ),
            environment=f"{environment_name}@latest" if environment_name else None,
            instance_type=instance_type,
            instance_count=instance_count,
        )

        deployment = self.ml_client.online_deployments.begin_create_or_update(
            deployment
        ).result()

        # Route 100% traffic to new deployment
        endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        endpoint.traffic = {deployment_name: 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()

        logger.info(f"Deployed {model_name} to {endpoint_name}/{deployment_name}")
        return deployment

    def invoke_endpoint(
        self,
        endpoint_name: str,
        data: Dict[str, Any],
    ) -> Any:
        """Invoke an online endpoint."""
        import json

        response = self.ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=None,
            request=json.dumps(data),
        )

        return json.loads(response)

    def delete_endpoint(self, endpoint_name: str):
        """Delete an online endpoint."""
        self.ml_client.online_endpoints.begin_delete(endpoint_name).result()
        logger.info(f"Deleted endpoint: {endpoint_name}")


# Scoring script example
SCORING_SCRIPT = """
import json
import joblib
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_data = np.array(data["data"])
        predictions = model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
"""


# Training script example
TRAINING_SCRIPT = """
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()

    # Load data
    df = pd.read_csv(os.path.join(args.data, "train.csv"))
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=args.n_estimators)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Log metrics
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Save model
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)

    mlflow.end_run()

if __name__ == "__main__":
    main()
"""


# Usage
def main():
    ml = AzureMLManager(
        subscription_id="your-subscription-id",
        resource_group="your-resource-group",
        workspace_name="your-workspace",
    )

    # Create compute
    ml.create_compute_cluster("cpu-cluster", max_instances=4)

    # Create environment
    ml.create_environment(
        "sklearn-env",
        pip_requirements=["scikit-learn", "pandas", "numpy", "mlflow"]
    )

    # Submit training job
    job = ml.submit_training_job(
        name="train-classifier",
        script_path="train.py",
        compute_name="cpu-cluster",
        environment_name="sklearn-env",
        inputs={"data": "azureml://datastores/training/paths/data"},
        outputs={"output": "azureml://datastores/models/paths/classifier"},
        arguments=["--data", "${{inputs.data}}", "--output", "${{outputs.output}}"],
    )

    # Wait for completion
    status = ml.wait_for_job(job.name)
    print(f"Job completed with status: {status}")

    # Register and deploy model
    model = ml.register_model(
        "classifier",
        f"azureml://jobs/{job.name}/outputs/output",
    )

    ml.create_online_endpoint("classifier-endpoint")
    ml.deploy_model(
        "classifier-endpoint",
        "v1",
        "classifier",
        instance_type="Standard_DS2_v2",
    )


if __name__ == "__main__":
    main()
''',
                domain="azure",
                subdomain="azure_ml",
                tags=["ml", "training", "deployment", "mlops"],
                difficulty="advanced"
            ),
        ]
