"""
AWS Cloud Domain

Covers:
- AWS SDK (boto3)
- Lambda, S3, DynamoDB, SQS, SNS
- EC2, ECS, EKS
- IAM, CloudWatch
- Step Functions
- CDK (Python)
"""

from typing import List
from .base import BaseDomain, DomainExample


class AWSCloudDomain(BaseDomain):
    """AWS cloud services training examples."""

    def get_name(self) -> str:
        return "AWS Cloud"

    def get_description(self) -> str:
        return "AWS services including Lambda, S3, DynamoDB, ECS, and CDK"

    def get_subdomains(self) -> List[str]:
        return ["lambda", "s3", "dynamodb", "ecs", "cdk", "step_functions"]

    def get_examples(self) -> List[DomainExample]:
        examples = []
        examples.extend(self._lambda_examples())
        examples.extend(self._s3_examples())
        examples.extend(self._dynamodb_examples())
        examples.extend(self._cdk_examples())
        return examples

    def _lambda_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an AWS Lambda function with API Gateway integration",
                code='''import json
import boto3
import logging
from typing import Dict, Any
from functools import wraps
import traceback

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize clients
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("users")


def cors_headers(origin: str = "*") -> Dict[str, str]:
    """Return CORS headers."""
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    }


def response(status_code: int, body: Any, headers: Dict = None) -> Dict:
    """Create API Gateway response."""
    return {
        "statusCode": status_code,
        "headers": {**cors_headers(), **(headers or {})},
        "body": json.dumps(body) if not isinstance(body, str) else body,
    }


def handle_exceptions(func):
    """Decorator to handle exceptions."""
    @wraps(func)
    def wrapper(event, context):
        try:
            return func(event, context)
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return response(400, {"error": str(e)})
        except KeyError as e:
            logger.warning(f"Missing key: {e}")
            return response(400, {"error": f"Missing required field: {e}"})
        except Exception as e:
            logger.error(f"Unexpected error: {e}\\n{traceback.format_exc()}")
            return response(500, {"error": "Internal server error"})
    return wrapper


def validate_user(data: Dict) -> Dict:
    """Validate user data."""
    required_fields = ["email", "name"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    if "@" not in data["email"]:
        raise ValueError("Invalid email format")

    return data


@handle_exceptions
def handler(event: Dict, context) -> Dict:
    """Main Lambda handler."""
    logger.info(f"Event: {json.dumps(event)}")

    # Handle CORS preflight
    if event.get("httpMethod") == "OPTIONS":
        return response(200, "")

    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    path_params = event.get("pathParameters") or {}
    query_params = event.get("queryStringParameters") or {}
    body = json.loads(event.get("body") or "{}") if event.get("body") else {}

    # Route requests
    if path == "/users" and http_method == "GET":
        return list_users(query_params)
    elif path == "/users" and http_method == "POST":
        return create_user(body)
    elif path.startswith("/users/") and http_method == "GET":
        return get_user(path_params.get("user_id"))
    elif path.startswith("/users/") and http_method == "PUT":
        return update_user(path_params.get("user_id"), body)
    elif path.startswith("/users/") and http_method == "DELETE":
        return delete_user(path_params.get("user_id"))
    else:
        return response(404, {"error": "Not found"})


def list_users(params: Dict) -> Dict:
    """List users with pagination."""
    limit = int(params.get("limit", 20))
    last_key = params.get("last_key")

    scan_kwargs = {"Limit": limit}
    if last_key:
        scan_kwargs["ExclusiveStartKey"] = {"user_id": last_key}

    result = table.scan(**scan_kwargs)

    return response(200, {
        "users": result.get("Items", []),
        "last_key": result.get("LastEvaluatedKey", {}).get("user_id"),
    })


def create_user(data: Dict) -> Dict:
    """Create a new user."""
    import uuid
    from datetime import datetime

    validated = validate_user(data)

    user = {
        "user_id": str(uuid.uuid4()),
        "email": validated["email"],
        "name": validated["name"],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    table.put_item(Item=user)
    logger.info(f"Created user: {user['user_id']}")

    return response(201, user)


def get_user(user_id: str) -> Dict:
    """Get user by ID."""
    result = table.get_item(Key={"user_id": user_id})

    if "Item" not in result:
        return response(404, {"error": "User not found"})

    return response(200, result["Item"])


def update_user(user_id: str, data: Dict) -> Dict:
    """Update user."""
    from datetime import datetime

    update_expr = "SET updated_at = :updated_at"
    expr_values = {":updated_at": datetime.utcnow().isoformat()}

    for key, value in data.items():
        if key not in ["user_id", "created_at"]:
            update_expr += f", {key} = :{key}"
            expr_values[f":{key}"] = value

    result = table.update_item(
        Key={"user_id": user_id},
        UpdateExpression=update_expr,
        ExpressionAttributeValues=expr_values,
        ReturnValues="ALL_NEW",
    )

    return response(200, result.get("Attributes"))


def delete_user(user_id: str) -> Dict:
    """Delete user."""
    table.delete_item(Key={"user_id": user_id})
    return response(204, "")''',
                domain="aws",
                subdomain="lambda",
                tags=["api_gateway", "crud", "dynamodb"],
                difficulty="intermediate"
            ),
        ]

    def _s3_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create S3 operations with presigned URLs and multipart upload",
                code='''import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, BinaryIO
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class S3Manager:
    """Manage S3 operations with advanced features."""

    def __init__(self, bucket_name: str, region: str = "us-east-1"):
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client(
            "s3",
            region_name=region,
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"}
            )
        )
        self.s3_resource = boto3.resource("s3", region_name=region)

    def generate_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        method: str = "get_object"
    ) -> str:
        """Generate presigned URL for object access."""
        try:
            url = self.s3_client.generate_presigned_url(
                ClientMethod=method,
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=expiration,
            )
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL: {e}")
            raise

    def generate_presigned_post(
        self,
        key: str,
        expiration: int = 3600,
        max_size: int = 10 * 1024 * 1024,  # 10MB
        content_type: str = None
    ) -> Dict:
        """Generate presigned POST for direct upload from browser."""
        conditions = [
            ["content-length-range", 0, max_size],
        ]

        if content_type:
            conditions.append(["eq", "$Content-Type", content_type])

        fields = {}
        if content_type:
            fields["Content-Type"] = content_type

        try:
            response = self.s3_client.generate_presigned_post(
                Bucket=self.bucket_name,
                Key=key,
                Fields=fields,
                Conditions=conditions,
                ExpiresIn=expiration,
            )
            return response
        except ClientError as e:
            logger.error(f"Error generating presigned POST: {e}")
            raise

    def multipart_upload(
        self,
        key: str,
        file_path: str,
        chunk_size: int = 8 * 1024 * 1024,  # 8MB
        max_workers: int = 4
    ) -> str:
        """Upload large file using multipart upload."""
        file_size = os.path.getsize(file_path)

        # Start multipart upload
        response = self.s3_client.create_multipart_upload(
            Bucket=self.bucket_name,
            Key=key
        )
        upload_id = response["UploadId"]

        try:
            parts = []
            part_number = 1

            with open(file_path, "rb") as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break

                    # Upload part
                    response = self.s3_client.upload_part(
                        Bucket=self.bucket_name,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=data
                    )

                    parts.append({
                        "PartNumber": part_number,
                        "ETag": response["ETag"]
                    })

                    logger.info(f"Uploaded part {part_number}")
                    part_number += 1

            # Complete upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts}
            )

            logger.info(f"Multipart upload completed: {key}")
            return key

        except Exception as e:
            # Abort on failure
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name,
                Key=key,
                UploadId=upload_id
            )
            raise

    def copy_object(self, source_key: str, dest_key: str, dest_bucket: str = None) -> None:
        """Copy object within or across buckets."""
        dest_bucket = dest_bucket or self.bucket_name
        copy_source = {"Bucket": self.bucket_name, "Key": source_key}

        self.s3_client.copy_object(
            CopySource=copy_source,
            Bucket=dest_bucket,
            Key=dest_key
        )

    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
        delimiter: str = None
    ) -> List[Dict]:
        """List objects with pagination."""
        objects = []
        paginator = self.s3_client.get_paginator("list_objects_v2")

        page_config = {
            "Bucket": self.bucket_name,
            "Prefix": prefix,
            "MaxKeys": max_keys
        }

        if delimiter:
            page_config["Delimiter"] = delimiter

        for page in paginator.paginate(**page_config):
            for obj in page.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": obj["LastModified"].isoformat(),
                    "etag": obj["ETag"].strip('"')
                })

        return objects

    def get_object_metadata(self, key: str) -> Dict:
        """Get object metadata without downloading."""
        response = self.s3_client.head_object(
            Bucket=self.bucket_name,
            Key=key
        )

        return {
            "content_type": response.get("ContentType"),
            "content_length": response.get("ContentLength"),
            "last_modified": response.get("LastModified"),
            "etag": response.get("ETag"),
            "metadata": response.get("Metadata", {})
        }

    def set_object_tags(self, key: str, tags: Dict[str, str]) -> None:
        """Set tags on an object."""
        tag_set = [{"Key": k, "Value": v} for k, v in tags.items()]

        self.s3_client.put_object_tagging(
            Bucket=self.bucket_name,
            Key=key,
            Tagging={"TagSet": tag_set}
        )

    def sync_directory(self, local_dir: str, s3_prefix: str) -> int:
        """Sync local directory to S3."""
        uploaded = 0

        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"{s3_prefix}/{relative_path}".replace("\\\\", "/")

                self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
                uploaded += 1
                logger.info(f"Uploaded: {s3_key}")

        return uploaded''',
                domain="aws",
                subdomain="s3",
                tags=["presigned_url", "multipart", "upload"],
                difficulty="intermediate"
            ),
        ]

    def _dynamodb_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create DynamoDB operations with transactions and GSI queries",
                code='''import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any
from decimal import Decimal
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DynamoDBManager:
    """DynamoDB operations with advanced patterns."""

    def __init__(self, table_name: str, region: str = "us-east-1"):
        self.dynamodb = boto3.resource("dynamodb", region_name=region)
        self.client = boto3.client("dynamodb", region_name=region)
        self.table = self.dynamodb.Table(table_name)
        self.table_name = table_name

    def _serialize(self, data: Dict) -> Dict:
        """Convert floats to Decimals for DynamoDB."""
        return json.loads(json.dumps(data), parse_float=Decimal)

    def _deserialize(self, data: Dict) -> Dict:
        """Convert Decimals to floats for JSON."""
        return json.loads(json.dumps(data, default=str))

    def put_item(self, item: Dict, condition: str = None) -> bool:
        """Put item with optional condition."""
        try:
            kwargs = {"Item": self._serialize(item)}

            if condition:
                kwargs["ConditionExpression"] = condition

            self.table.put_item(**kwargs)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning(f"Condition check failed for put_item")
                return False
            raise

    def get_item(self, key: Dict) -> Optional[Dict]:
        """Get item by primary key."""
        response = self.table.get_item(Key=key)
        item = response.get("Item")
        return self._deserialize(item) if item else None

    def query(
        self,
        key_condition: Any,
        index_name: str = None,
        filter_expression: Any = None,
        limit: int = None,
        scan_forward: bool = True,
        projection: List[str] = None
    ) -> List[Dict]:
        """Query with optional GSI and filters."""
        kwargs = {
            "KeyConditionExpression": key_condition,
            "ScanIndexForward": scan_forward,
        }

        if index_name:
            kwargs["IndexName"] = index_name

        if filter_expression:
            kwargs["FilterExpression"] = filter_expression

        if limit:
            kwargs["Limit"] = limit

        if projection:
            kwargs["ProjectionExpression"] = ", ".join(projection)

        items = []
        while True:
            response = self.table.query(**kwargs)
            items.extend(response.get("Items", []))

            if "LastEvaluatedKey" not in response or (limit and len(items) >= limit):
                break

            kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]

        return [self._deserialize(item) for item in items]

    def update_item(
        self,
        key: Dict,
        updates: Dict,
        condition: str = None
    ) -> Dict:
        """Update item with expression builder."""
        update_parts = []
        expr_names = {}
        expr_values = {}

        for i, (field, value) in enumerate(updates.items()):
            placeholder_name = f"#field{i}"
            placeholder_value = f":value{i}"

            update_parts.append(f"{placeholder_name} = {placeholder_value}")
            expr_names[placeholder_name] = field
            expr_values[placeholder_value] = self._serialize({field: value})[field]

        update_expr = "SET " + ", ".join(update_parts)

        kwargs = {
            "Key": key,
            "UpdateExpression": update_expr,
            "ExpressionAttributeNames": expr_names,
            "ExpressionAttributeValues": expr_values,
            "ReturnValues": "ALL_NEW",
        }

        if condition:
            kwargs["ConditionExpression"] = condition

        response = self.table.update_item(**kwargs)
        return self._deserialize(response.get("Attributes", {}))

    def transact_write(self, operations: List[Dict]) -> bool:
        """Execute transactional write."""
        transact_items = []

        for op in operations:
            if op["type"] == "put":
                transact_items.append({
                    "Put": {
                        "TableName": self.table_name,
                        "Item": self._serialize(op["item"]),
                    }
                })
            elif op["type"] == "update":
                transact_items.append({
                    "Update": {
                        "TableName": self.table_name,
                        "Key": op["key"],
                        "UpdateExpression": op["update_expression"],
                        "ExpressionAttributeValues": op.get("expression_values", {}),
                    }
                })
            elif op["type"] == "delete":
                transact_items.append({
                    "Delete": {
                        "TableName": self.table_name,
                        "Key": op["key"],
                    }
                })

        try:
            self.client.transact_write_items(TransactItems=transact_items)
            return True
        except ClientError as e:
            logger.error(f"Transaction failed: {e}")
            return False

    def batch_write(self, items: List[Dict], batch_size: int = 25) -> int:
        """Batch write items."""
        written = 0

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            request_items = {
                self.table_name: [
                    {"PutRequest": {"Item": self._serialize(item)}}
                    for item in batch
                ]
            }

            response = self.dynamodb.meta.client.batch_write_item(
                RequestItems=request_items
            )

            # Handle unprocessed items
            unprocessed = response.get("UnprocessedItems", {})
            while unprocessed:
                response = self.dynamodb.meta.client.batch_write_item(
                    RequestItems=unprocessed
                )
                unprocessed = response.get("UnprocessedItems", {})

            written += len(batch)

        return written


# Example usage patterns
def order_processing_example():
    """Example: Order processing with transactions."""
    db = DynamoDBManager("orders")

    # Create order with inventory update
    order_id = "order-123"
    product_id = "product-456"

    operations = [
        # Create order
        {
            "type": "put",
            "item": {
                "pk": f"ORDER#{order_id}",
                "sk": "METADATA",
                "order_id": order_id,
                "product_id": product_id,
                "quantity": 2,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
            }
        },
        # Decrement inventory
        {
            "type": "update",
            "key": {"pk": f"PRODUCT#{product_id}", "sk": "INVENTORY"},
            "update_expression": "SET quantity = quantity - :qty",
            "expression_values": {":qty": {"N": "2"}},
        }
    ]

    success = db.transact_write(operations)
    return success


def query_gsi_example():
    """Example: Query using GSI."""
    db = DynamoDBManager("orders")

    # Query orders by customer using GSI
    orders = db.query(
        key_condition=Key("gsi1pk").eq("CUSTOMER#cust-123"),
        index_name="gsi1",
        filter_expression=Attr("status").eq("completed"),
        limit=10
    )

    return orders''',
                domain="aws",
                subdomain="dynamodb",
                tags=["transactions", "gsi", "query"],
                difficulty="advanced"
            ),
        ]

    def _cdk_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an AWS CDK stack for a serverless API with Lambda and DynamoDB",
                code='''from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_lambda as lambda_,
    aws_dynamodb as dynamodb,
    aws_apigateway as apigw,
    aws_iam as iam,
    aws_logs as logs,
    aws_sqs as sqs,
    aws_lambda_event_sources as event_sources,
)
from constructs import Construct


class ServerlessApiStack(Stack):
    """CDK Stack for serverless API."""

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # DynamoDB Table
        table = dynamodb.Table(
            self, "UsersTable",
            table_name="users",
            partition_key=dynamodb.Attribute(
                name="pk",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="sk",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY,
            point_in_time_recovery=True,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
        )

        # GSI for querying by email
        table.add_global_secondary_index(
            index_name="gsi1",
            partition_key=dynamodb.Attribute(
                name="gsi1pk",
                type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="gsi1sk",
                type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # Lambda Layer for shared code
        shared_layer = lambda_.LayerVersion(
            self, "SharedLayer",
            code=lambda_.Code.from_asset("layers/shared"),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_11],
            description="Shared utilities and dependencies",
        )

        # Lambda Function
        api_handler = lambda_.Function(
            self, "ApiHandler",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.handler",
            code=lambda_.Code.from_asset("lambda/api"),
            layers=[shared_layer],
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "TABLE_NAME": table.table_name,
                "LOG_LEVEL": "INFO",
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.ONE_WEEK,
        )

        # Grant DynamoDB permissions
        table.grant_read_write_data(api_handler)

        # API Gateway
        api = apigw.RestApi(
            self, "UsersApi",
            rest_api_name="Users Service",
            description="User management API",
            deploy_options=apigw.StageOptions(
                stage_name="prod",
                throttling_rate_limit=1000,
                throttling_burst_limit=500,
                logging_level=apigw.MethodLoggingLevel.INFO,
                data_trace_enabled=True,
            ),
            default_cors_preflight_options=apigw.CorsOptions(
                allow_origins=apigw.Cors.ALL_ORIGINS,
                allow_methods=apigw.Cors.ALL_METHODS,
            ),
        )

        # Lambda integration
        integration = apigw.LambdaIntegration(
            api_handler,
            request_templates={"application/json": \'{ "statusCode": "200" }\'}
        )

        # API Resources and Methods
        users = api.root.add_resource("users")
        users.add_method("GET", integration)
        users.add_method("POST", integration)

        user = users.add_resource("{user_id}")
        user.add_method("GET", integration)
        user.add_method("PUT", integration)
        user.add_method("DELETE", integration)

        # Dead Letter Queue
        dlq = sqs.Queue(
            self, "DLQ",
            queue_name="users-dlq",
            retention_period=Duration.days(14),
        )

        # Stream processor Lambda
        stream_processor = lambda_.Function(
            self, "StreamProcessor",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="stream.handler",
            code=lambda_.Code.from_asset("lambda/stream"),
            timeout=Duration.seconds(60),
            environment={
                "TABLE_NAME": table.table_name,
            },
            dead_letter_queue=dlq,
        )

        # DynamoDB Stream trigger
        stream_processor.add_event_source(
            event_sources.DynamoEventSource(
                table,
                starting_position=lambda_.StartingPosition.TRIM_HORIZON,
                batch_size=100,
                bisect_batch_on_error=True,
                retry_attempts=3,
            )
        )

        # Outputs
        CfnOutput(self, "ApiUrl", value=api.url)
        CfnOutput(self, "TableName", value=table.table_name)''',
                domain="aws",
                subdomain="cdk",
                tags=["infrastructure", "serverless", "iac"],
                difficulty="advanced"
            ),
        ]
