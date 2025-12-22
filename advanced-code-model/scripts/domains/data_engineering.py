"""
Data Engineering Domain

Covers:
- PySpark & SparkSQL
- AWS Glue
- Apache Airflow
- ETL Pipelines
- Data Lakes & Warehouses
- Stream Processing (Kafka, Kinesis)
- Data Quality & Validation
"""

from typing import List, Optional
from .base import BaseDomain, DomainExample


class DataEngineeringDomain(BaseDomain):
    """Data Engineering training examples."""

    def __init__(self, focus: Optional[str] = None):
        super().__init__()
        self.focus = focus  # pyspark, glue, bigdata, etl, or None for all

    def get_name(self) -> str:
        if self.focus:
            return f"Data Engineering ({self.focus})"
        return "Data Engineering"

    def get_description(self) -> str:
        return "PySpark, AWS Glue, ETL pipelines, data lakes, and big data processing"

    def get_subdomains(self) -> List[str]:
        return ["pyspark", "glue", "airflow", "etl", "streaming", "data_quality"]

    def get_examples(self) -> List[DomainExample]:
        examples = []

        if not self.focus or self.focus == "pyspark":
            examples.extend(self._pyspark_examples())
        if not self.focus or self.focus == "glue":
            examples.extend(self._glue_examples())
        if not self.focus or self.focus in ["bigdata", "etl"]:
            examples.extend(self._etl_examples())
        if not self.focus or self.focus == "streaming":
            examples.extend(self._streaming_examples())

        examples.extend(self._data_quality_examples())

        return examples

    def _pyspark_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a PySpark ETL job to process large CSV files with transformations",
                code='''from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
from pyspark.sql.window import Window
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name: str = "ETL Job") -> SparkSession:
    """Create optimized Spark session."""
    return (SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.broadcastTimeout", "600")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate())


def define_schema() -> StructType:
    """Define explicit schema for better performance."""
    return StructType([
        StructField("id", IntegerType(), False),
        StructField("user_id", IntegerType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("price", DoubleType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("category", StringType(), True),
        StructField("region", StringType(), True),
    ])


def read_data(spark: SparkSession, path: str, schema: StructType = None):
    """Read CSV data with schema inference or explicit schema."""
    reader = spark.read.format("csv").option("header", "true")

    if schema:
        return reader.schema(schema).load(path)
    else:
        return reader.option("inferSchema", "true").load(path)


def clean_data(df):
    """Clean and validate data."""
    return (df
        # Remove nulls in critical columns
        .dropna(subset=["id", "user_id", "product_id"])
        # Remove duplicates
        .dropDuplicates(["id"])
        # Filter invalid values
        .filter(F.col("quantity") > 0)
        .filter(F.col("price") > 0)
        # Trim strings
        .withColumn("category", F.trim(F.col("category")))
        .withColumn("region", F.trim(F.col("region")))
        # Fill nulls
        .fillna({"category": "unknown", "region": "unknown"})
    )


def add_derived_columns(df):
    """Add calculated columns."""
    return (df
        .withColumn("total_amount", F.col("quantity") * F.col("price"))
        .withColumn("date", F.to_date(F.col("timestamp")))
        .withColumn("hour", F.hour(F.col("timestamp")))
        .withColumn("day_of_week", F.dayofweek(F.col("timestamp")))
        .withColumn("is_weekend", F.when(F.col("day_of_week").isin(1, 7), True).otherwise(False))
        .withColumn("price_bucket", F.when(F.col("price") < 10, "low")
                                     .when(F.col("price") < 50, "medium")
                                     .otherwise("high"))
    )


def add_window_features(df):
    """Add window-based features."""
    # User window
    user_window = Window.partitionBy("user_id").orderBy("timestamp")
    user_window_all = Window.partitionBy("user_id")

    # Running calculations
    df = (df
        .withColumn("user_order_number", F.row_number().over(user_window))
        .withColumn("user_running_total", F.sum("total_amount").over(user_window))
        .withColumn("user_avg_order", F.avg("total_amount").over(user_window_all))
        .withColumn("days_since_last_order",
            F.datediff(F.col("date"),
                       F.lag("date").over(user_window)))
    )

    return df


def aggregate_metrics(df):
    """Create aggregated metrics."""
    # Daily metrics
    daily_metrics = (df
        .groupBy("date", "category", "region")
        .agg(
            F.count("*").alias("order_count"),
            F.sum("total_amount").alias("revenue"),
            F.avg("total_amount").alias("avg_order_value"),
            F.countDistinct("user_id").alias("unique_customers"),
            F.sum("quantity").alias("items_sold"),
            F.approx_count_distinct("product_id").alias("unique_products")
        )
    )

    return daily_metrics


def write_output(df, path: str, partition_cols: List[str] = None):
    """Write data in optimized format."""
    writer = (df.write
        .format("parquet")
        .mode("overwrite")
        .option("compression", "snappy"))

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    writer.save(path)
    logger.info(f"Written {df.count()} rows to {path}")


def main():
    logger.info("Starting ETL job")

    # Create session
    spark = create_spark_session("Sales ETL")

    try:
        # Read
        schema = define_schema()
        raw_df = read_data(spark, "s3://bucket/raw/sales/", schema)
        logger.info(f"Read {raw_df.count()} rows")

        # Transform
        cleaned_df = clean_data(raw_df)
        enriched_df = add_derived_columns(cleaned_df)
        featured_df = add_window_features(enriched_df)

        # Cache for multiple outputs
        featured_df.cache()

        # Write detailed data
        write_output(
            featured_df,
            "s3://bucket/processed/sales_detailed/",
            partition_cols=["date", "region"]
        )

        # Write aggregated data
        metrics_df = aggregate_metrics(featured_df)
        write_output(
            metrics_df,
            "s3://bucket/processed/sales_metrics/",
            partition_cols=["date"]
        )

        logger.info("ETL job completed successfully")

    except Exception as e:
        logger.error(f"ETL job failed: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()''',
                domain="data_engineering",
                subdomain="pyspark",
                tags=["etl", "spark", "transformations"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Create PySpark DataFrame operations with complex joins and UDFs",
                code='''from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, ArrayType, MapType, StructType, StructField
from pyspark.sql.window import Window
import json
from typing import List, Dict


def create_spark_session() -> SparkSession:
    return SparkSession.builder.appName("Complex Operations").getOrCreate()


# ============================================================
# User Defined Functions (UDFs)
# ============================================================

@F.udf(returnType=StringType())
def parse_user_agent(ua: str) -> str:
    """Parse user agent to extract browser."""
    if not ua:
        return "unknown"
    ua_lower = ua.lower()
    if "chrome" in ua_lower:
        return "chrome"
    elif "firefox" in ua_lower:
        return "firefox"
    elif "safari" in ua_lower:
        return "safari"
    elif "edge" in ua_lower:
        return "edge"
    return "other"


@F.udf(returnType=ArrayType(StringType()))
def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text."""
    if not text:
        return []
    import re
    return re.findall(r"#(\\w+)", text)


@F.udf(returnType=MapType(StringType(), StringType()))
def parse_json_safely(json_str: str) -> Dict[str, str]:
    """Safely parse JSON string."""
    try:
        return json.loads(json_str) if json_str else {}
    except:
        return {}


# Pandas UDF for better performance
from pyspark.sql.functions import pandas_udf
import pandas as pd

@pandas_udf("double")
def calculate_zscore(values: pd.Series) -> pd.Series:
    """Calculate z-score using Pandas UDF."""
    mean = values.mean()
    std = values.std()
    return (values - mean) / std if std > 0 else pd.Series([0] * len(values))


# ============================================================
# Complex Joins
# ============================================================

def perform_complex_joins(spark: SparkSession):
    """Demonstrate various join types."""

    # Sample data
    orders = spark.createDataFrame([
        (1, 101, "2024-01-01", 100.0),
        (2, 102, "2024-01-02", 150.0),
        (3, 101, "2024-01-03", 200.0),
        (4, 103, "2024-01-04", 75.0),
    ], ["order_id", "customer_id", "order_date", "amount"])

    customers = spark.createDataFrame([
        (101, "Alice", "premium"),
        (102, "Bob", "standard"),
        (104, "David", "premium"),
    ], ["customer_id", "name", "tier"])

    products = spark.createDataFrame([
        (1, 1001), (1, 1002), (2, 1001), (3, 1003),
    ], ["order_id", "product_id"])

    # Inner join
    inner_join = orders.join(customers, "customer_id", "inner")

    # Left outer join
    left_join = orders.join(customers, "customer_id", "left")

    # Full outer join
    full_join = orders.join(customers, "customer_id", "full")

    # Left anti join (orders without matching customers)
    orders_no_customer = orders.join(customers, "customer_id", "left_anti")

    # Left semi join (orders with matching customers, but only order columns)
    orders_with_customer = orders.join(customers, "customer_id", "left_semi")

    # Multiple joins
    full_order_info = (orders
        .join(customers, "customer_id", "left")
        .join(products, "order_id", "left")
    )

    # Broadcast join for small tables
    broadcast_join = orders.join(
        F.broadcast(customers),
        "customer_id",
        "left"
    )

    # Self join (find orders on same day)
    orders_alias = orders.alias("o1")
    orders_alias2 = orders.alias("o2")
    same_day_orders = (orders_alias
        .join(orders_alias2,
              (F.col("o1.order_date") == F.col("o2.order_date")) &
              (F.col("o1.order_id") < F.col("o2.order_id")),
              "inner")
        .select(
            F.col("o1.order_id").alias("order1"),
            F.col("o2.order_id").alias("order2"),
            F.col("o1.order_date")
        )
    )

    return full_order_info


# ============================================================
# Advanced Aggregations
# ============================================================

def advanced_aggregations(df):
    """Demonstrate advanced aggregation patterns."""

    # Multiple aggregations
    summary = df.groupBy("category").agg(
        F.count("*").alias("count"),
        F.sum("amount").alias("total"),
        F.avg("amount").alias("average"),
        F.stddev("amount").alias("std_dev"),
        F.min("amount").alias("min"),
        F.max("amount").alias("max"),
        F.expr("percentile_approx(amount, 0.5)").alias("median"),
        F.expr("percentile_approx(amount, array(0.25, 0.5, 0.75))").alias("quartiles"),
        F.collect_list("product_id").alias("products"),
        F.collect_set("customer_id").alias("unique_customers"),
    )

    # Pivot table
    pivot_df = df.groupBy("region").pivot("category").agg(F.sum("amount"))

    # Cube (all combinations of grouping)
    cube_df = df.cube("region", "category").agg(F.sum("amount").alias("total"))

    # Rollup (hierarchical aggregation)
    rollup_df = df.rollup("region", "category").agg(F.sum("amount").alias("total"))

    return summary


# ============================================================
# Window Functions
# ============================================================

def advanced_window_operations(df):
    """Demonstrate advanced window functions."""

    # Define windows
    customer_window = Window.partitionBy("customer_id").orderBy("order_date")
    category_window = Window.partitionBy("category").orderBy("order_date")
    rolling_window = (Window.partitionBy("customer_id")
                     .orderBy(F.col("order_date").cast("timestamp").cast("long"))
                     .rangeBetween(-7 * 86400, 0))  # 7 days

    result = (df
        # Row number and rank
        .withColumn("customer_order_num", F.row_number().over(customer_window))
        .withColumn("category_rank", F.rank().over(category_window.orderBy(F.desc("amount"))))
        .withColumn("dense_rank", F.dense_rank().over(category_window.orderBy(F.desc("amount"))))
        .withColumn("percent_rank", F.percent_rank().over(category_window.orderBy("amount")))
        .withColumn("ntile", F.ntile(4).over(category_window.orderBy("amount")))

        # Lag and lead
        .withColumn("prev_amount", F.lag("amount", 1).over(customer_window))
        .withColumn("next_amount", F.lead("amount", 1).over(customer_window))
        .withColumn("first_order", F.first("amount").over(customer_window))
        .withColumn("last_order", F.last("amount").over(customer_window))

        # Running calculations
        .withColumn("running_total", F.sum("amount").over(customer_window))
        .withColumn("running_avg", F.avg("amount").over(customer_window))
        .withColumn("running_count", F.count("*").over(customer_window))

        # Rolling window
        .withColumn("rolling_7d_sum", F.sum("amount").over(rolling_window))
        .withColumn("rolling_7d_avg", F.avg("amount").over(rolling_window))
    )

    return result


# ============================================================
# Main
# ============================================================

def main():
    spark = create_spark_session()

    # Example usage
    df = spark.read.parquet("s3://bucket/data/")

    # Apply UDFs
    df = df.withColumn("browser", parse_user_agent(F.col("user_agent")))
    df = df.withColumn("hashtags", extract_hashtags(F.col("text")))

    # Complex operations
    result = advanced_window_operations(df)
    result.show()

    spark.stop()


if __name__ == "__main__":
    main()''',
                domain="data_engineering",
                subdomain="pyspark",
                tags=["udf", "joins", "window_functions"],
                difficulty="advanced"
            ),
        ]

    def _glue_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an AWS Glue ETL job with DynamicFrames",
                code='''import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get job parameters
args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "source_database",
    "source_table",
    "target_path",
    "target_database",
    "target_table"
])

# Initialize Glue context
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)


def read_from_catalog(database: str, table: str) -> DynamicFrame:
    """Read data from Glue Data Catalog."""
    logger.info(f"Reading from {database}.{table}")

    dyf = glueContext.create_dynamic_frame.from_catalog(
        database=database,
        table_name=table,
        transformation_ctx="source_data",
        push_down_predicate="year >= 2024"  # Partition pruning
    )

    logger.info(f"Read {dyf.count()} records")
    return dyf


def clean_data(dyf: DynamicFrame) -> DynamicFrame:
    """Clean and transform data using DynamicFrame."""

    # Apply mapping to rename and cast columns
    mapped = ApplyMapping.apply(
        frame=dyf,
        mappings=[
            ("id", "long", "order_id", "long"),
            ("customer_id", "long", "customer_id", "long"),
            ("product_id", "long", "product_id", "long"),
            ("amount", "string", "amount", "double"),
            ("order_date", "string", "order_date", "date"),
            ("status", "string", "status", "string"),
        ],
        transformation_ctx="mapped_data"
    )

    # Resolve choice types (handle schema inconsistencies)
    resolved = ResolveChoice.apply(
        frame=mapped,
        choice="match_catalog",
        database=args["target_database"],
        table_name=args["target_table"],
        transformation_ctx="resolved_data"
    )

    # Drop null fields
    dropped = DropNullFields.apply(
        frame=resolved,
        transformation_ctx="dropped_nulls"
    )

    # Filter using relationalize for nested data
    # relationalized = Relationalize.apply(
    #     frame=dropped,
    #     staging_path="s3://bucket/temp/",
    #     name="relationalized",
    #     transformation_ctx="relationalized"
    # )

    return dropped


def transform_with_spark(dyf: DynamicFrame) -> DynamicFrame:
    """Convert to Spark DataFrame for complex transformations."""

    df = dyf.toDF()

    # Complex transformations
    transformed = (df
        .withColumn("amount", F.col("amount").cast("double"))
        .withColumn("order_month", F.date_format(F.col("order_date"), "yyyy-MM"))
        .withColumn("is_high_value", F.when(F.col("amount") > 1000, True).otherwise(False))
        .filter(F.col("status").isin(["completed", "shipped"]))
        .dropDuplicates(["order_id"])
    )

    # Convert back to DynamicFrame
    return DynamicFrame.fromDF(transformed, glueContext, "transformed")


def write_to_s3(dyf: DynamicFrame, path: str, partition_keys: list = None):
    """Write data to S3 in Parquet format."""

    logger.info(f"Writing {dyf.count()} records to {path}")

    sink = glueContext.getSink(
        path=path,
        connection_type="s3",
        updateBehavior="UPDATE_IN_DATABASE",
        partitionKeys=partition_keys or [],
        compression="snappy",
        enableUpdateCatalog=True,
        transformation_ctx="sink"
    )

    sink.setCatalogInfo(
        catalogDatabase=args["target_database"],
        catalogTableName=args["target_table"]
    )

    sink.setFormat("glueparquet")
    sink.writeFrame(dyf)


def write_to_redshift(dyf: DynamicFrame, connection: str, table: str):
    """Write data to Redshift."""

    glueContext.write_dynamic_frame.from_jdbc_conf(
        frame=dyf,
        catalog_connection=connection,
        connection_options={
            "dbtable": table,
            "database": "warehouse"
        },
        redshift_tmp_dir="s3://bucket/temp/redshift/",
        transformation_ctx="redshift_sink"
    )


def main():
    try:
        # Read source data
        source_dyf = read_from_catalog(
            args["source_database"],
            args["source_table"]
        )

        # Clean and transform
        cleaned_dyf = clean_data(source_dyf)
        transformed_dyf = transform_with_spark(cleaned_dyf)

        # Write to target
        write_to_s3(
            transformed_dyf,
            args["target_path"],
            partition_keys=["order_month"]
        )

        job.commit()
        logger.info("Job completed successfully")

    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise


if __name__ == "__main__":
    main()''',
                domain="data_engineering",
                subdomain="glue",
                tags=["aws", "etl", "dynamicframe"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Create AWS Glue job with bookmarks and incremental processing",
                code='''import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from pyspark.sql import functions as F
from datetime import datetime, timedelta
import boto3


args = getResolvedOptions(sys.argv, [
    "JOB_NAME",
    "source_bucket",
    "source_prefix",
    "target_bucket",
    "target_prefix",
    "bookmark_key"
])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)


class BookmarkManager:
    """Manage job bookmarks for incremental processing."""

    def __init__(self, table_name: str = "glue_bookmarks"):
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table(table_name)

    def get_bookmark(self, job_key: str) -> dict:
        """Get last processed bookmark."""
        try:
            response = self.table.get_item(Key={"job_key": job_key})
            return response.get("Item", {})
        except Exception:
            return {}

    def set_bookmark(self, job_key: str, bookmark_value: str):
        """Save bookmark."""
        self.table.put_item(Item={
            "job_key": job_key,
            "bookmark_value": bookmark_value,
            "updated_at": datetime.utcnow().isoformat()
        })


def read_incremental_data(source_path: str, bookmark_manager: BookmarkManager, job_key: str) -> DynamicFrame:
    """Read only new/updated data based on bookmark."""

    # Get last bookmark
    bookmark = bookmark_manager.get_bookmark(job_key)
    last_processed = bookmark.get("bookmark_value", "1970-01-01T00:00:00")

    # Read with push-down predicate
    dyf = glueContext.create_dynamic_frame.from_options(
        connection_type="s3",
        connection_options={
            "paths": [source_path],
            "recurse": True,
            # Use bookmark for filtering
            "groupFiles": "inPartition",
            "groupSize": "134217728"  # 128MB
        },
        format="parquet",
        transformation_ctx="source",
        # Enable job bookmarks
        additional_options={
            "jobBookmarkKeys": ["modified_date"],
            "jobBookmarkKeysSortOrder": "asc"
        }
    )

    # Additional filtering in Spark
    df = dyf.toDF()
    filtered_df = df.filter(F.col("modified_date") > last_processed)

    return DynamicFrame.fromDF(filtered_df, glueContext, "filtered_source")


def process_with_deduplication(dyf: DynamicFrame) -> DynamicFrame:
    """Process data with deduplication for upserts."""

    df = dyf.toDF()

    # Deduplicate keeping latest record
    from pyspark.sql.window import Window
    window = Window.partitionBy("id").orderBy(F.desc("modified_date"))

    deduped = (df
        .withColumn("row_num", F.row_number().over(window))
        .filter(F.col("row_num") == 1)
        .drop("row_num")
    )

    return DynamicFrame.fromDF(deduped, glueContext, "deduped")


def merge_with_existing(new_dyf: DynamicFrame, existing_path: str) -> DynamicFrame:
    """Merge new data with existing data (CDC pattern)."""

    # Read existing data
    try:
        existing_df = spark.read.parquet(existing_path)
    except Exception:
        # No existing data
        return new_dyf

    new_df = new_dyf.toDF()

    # Identify records to update vs insert
    existing_ids = existing_df.select("id").distinct()
    new_ids = new_df.select("id").distinct()

    # Records to update (exist in both)
    updates = new_df.join(existing_ids, "id", "inner")

    # Records to insert (only in new)
    inserts = new_df.join(existing_ids, "id", "left_anti")

    # Unchanged records (only in existing, not in new)
    unchanged_ids = existing_ids.join(new_ids, "id", "left_anti")
    unchanged = existing_df.join(unchanged_ids, "id", "inner")

    # Combine all
    merged = unchanged.union(updates).union(inserts)

    return DynamicFrame.fromDF(merged, glueContext, "merged")


def write_with_compaction(dyf: DynamicFrame, target_path: str, partition_keys: list):
    """Write data with small file compaction."""

    df = dyf.toDF()

    # Coalesce to reduce small files
    num_partitions = max(1, df.count() // 100000)  # ~100K rows per file
    coalesced = df.coalesce(num_partitions)

    # Write with dynamic partition overwrite
    (coalesced.write
        .mode("overwrite")
        .partitionBy(*partition_keys)
        .option("partitionOverwriteMode", "dynamic")
        .parquet(target_path))


def main():
    bookmark_manager = BookmarkManager()
    source_path = f"s3://{args['source_bucket']}/{args['source_prefix']}"
    target_path = f"s3://{args['target_bucket']}/{args['target_prefix']}"

    # Read incremental data
    source_dyf = read_incremental_data(
        source_path,
        bookmark_manager,
        args["bookmark_key"]
    )

    if source_dyf.count() == 0:
        print("No new data to process")
        job.commit()
        return

    # Process
    deduped_dyf = process_with_deduplication(source_dyf)
    merged_dyf = merge_with_existing(deduped_dyf, target_path)

    # Write
    write_with_compaction(
        merged_dyf,
        target_path,
        partition_keys=["year", "month"]
    )

    # Update bookmark
    max_date = merged_dyf.toDF().agg(F.max("modified_date")).collect()[0][0]
    bookmark_manager.set_bookmark(args["bookmark_key"], str(max_date))

    job.commit()


if __name__ == "__main__":
    main()''',
                domain="data_engineering",
                subdomain="glue",
                tags=["aws", "incremental", "cdc", "bookmarks"],
                difficulty="advanced"
            ),
        ]

    def _etl_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an Apache Airflow DAG for data pipeline orchestration",
                code='''from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.operators.glue import GlueJobOperator
from airflow.providers.amazon.aws.operators.s3 import S3DeleteObjectsOperator
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from datetime import datetime, timedelta
import json


# Default arguments
default_args = {
    "owner": "data-team",
    "depends_on_past": False,
    "email": ["data-team@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
}


# DAG definition
with DAG(
    dag_id="sales_data_pipeline",
    default_args=default_args,
    description="Daily sales data ETL pipeline",
    schedule_interval="0 6 * * *",  # 6 AM daily
    start_date=days_ago(1),
    catchup=False,
    tags=["sales", "etl", "production"],
    max_active_runs=1,
    doc_md="""
    ## Sales Data Pipeline

    This DAG processes daily sales data:
    1. Waits for source files
    2. Validates data quality
    3. Runs Glue ETL jobs
    4. Updates data catalog
    5. Sends notifications
    """,
) as dag:

    # ============================================================
    # Helper functions
    # ============================================================

    def check_data_quality(**context):
        """Validate data quality before processing."""
        s3_hook = S3Hook(aws_conn_id="aws_default")
        bucket = Variable.get("raw_bucket")
        prefix = f"sales/{context['ds']}/"

        # List files
        files = s3_hook.list_keys(bucket, prefix=prefix)
        if not files:
            raise ValueError(f"No files found for {context['ds']}")

        # Check file sizes
        for key in files:
            metadata = s3_hook.head_object(key, bucket)
            if metadata["ContentLength"] < 1000:
                raise ValueError(f"File too small: {key}")

        context["ti"].xcom_push(key="file_count", value=len(files))
        return "data_valid"

    def decide_processing_path(**context):
        """Branch based on data volume."""
        file_count = context["ti"].xcom_pull(key="file_count")

        if file_count > 100:
            return "heavy_processing"
        else:
            return "light_processing"

    def send_success_notification(**context):
        """Send success notification."""
        stats = context["ti"].xcom_pull(task_ids="aggregate_stats")
        message = f"""
        :white_check_mark: Sales Pipeline Complete

        Date: {context['ds']}
        Records processed: {stats.get('record_count', 'N/A')}
        Duration: {stats.get('duration', 'N/A')}
        """
        return message

    def handle_failure(context):
        """Callback for task failures."""
        task_instance = context["task_instance"]
        message = f"""
        :x: Sales Pipeline Failed

        DAG: {context['dag'].dag_id}
        Task: {task_instance.task_id}
        Error: {context['exception']}
        Log URL: {task_instance.log_url}
        """
        # Send alert
        SlackWebhookOperator(
            task_id="failure_alert",
            slack_webhook_conn_id="slack_webhook",
            message=message,
        ).execute(context)

    # ============================================================
    # Tasks
    # ============================================================

    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    # Wait for source data
    wait_for_data = S3KeySensor(
        task_id="wait_for_source_data",
        bucket_name="{{ var.value.raw_bucket }}",
        bucket_key="sales/{{ ds }}/",
        wildcard_match=True,
        timeout=3600,
        poke_interval=60,
        mode="reschedule",
    )

    # Validate data
    validate_data = PythonOperator(
        task_id="validate_data_quality",
        python_callable=check_data_quality,
        on_failure_callback=handle_failure,
    )

    # Branch based on volume
    branch_decision = BranchPythonOperator(
        task_id="decide_processing",
        python_callable=decide_processing_path,
    )

    # Processing paths
    with TaskGroup("heavy_processing") as heavy_processing:
        extract_heavy = GlueJobOperator(
            task_id="extract_large_dataset",
            job_name="sales-extract-heavy",
            script_location="s3://scripts/extract_heavy.py",
            concurrent_run_limit=2,
            aws_conn_id="aws_default",
        )

        transform_heavy = GlueJobOperator(
            task_id="transform_large_dataset",
            job_name="sales-transform-heavy",
            script_location="s3://scripts/transform_heavy.py",
            aws_conn_id="aws_default",
        )

        extract_heavy >> transform_heavy

    with TaskGroup("light_processing") as light_processing:
        extract_light = GlueJobOperator(
            task_id="extract_small_dataset",
            job_name="sales-extract-light",
            script_location="s3://scripts/extract_light.py",
            aws_conn_id="aws_default",
        )

        transform_light = GlueJobOperator(
            task_id="transform_small_dataset",
            job_name="sales-transform-light",
            script_location="s3://scripts/transform_light.py",
            aws_conn_id="aws_default",
        )

        extract_light >> transform_light

    # Join paths
    join_paths = EmptyOperator(
        task_id="join_paths",
        trigger_rule="none_failed_min_one_success",
    )

    # Load to warehouse
    load_to_warehouse = GlueJobOperator(
        task_id="load_to_warehouse",
        job_name="sales-load-warehouse",
        script_location="s3://scripts/load_warehouse.py",
        aws_conn_id="aws_default",
    )

    # Aggregate statistics
    aggregate_stats = PythonOperator(
        task_id="aggregate_stats",
        python_callable=lambda **ctx: {
            "record_count": 10000,
            "duration": "15m"
        },
    )

    # Cleanup temp files
    cleanup = S3DeleteObjectsOperator(
        task_id="cleanup_temp_files",
        bucket="{{ var.value.temp_bucket }}",
        prefix="temp/{{ ds }}/",
        aws_conn_id="aws_default",
    )

    # Notification
    notify_success = SlackWebhookOperator(
        task_id="notify_success",
        slack_webhook_conn_id="slack_webhook",
        message="{{ ti.xcom_pull(task_ids='aggregate_stats') }}",
    )

    # ============================================================
    # Dependencies
    # ============================================================

    start >> wait_for_data >> validate_data >> branch_decision
    branch_decision >> [heavy_processing, light_processing] >> join_paths
    join_paths >> load_to_warehouse >> aggregate_stats >> [cleanup, notify_success] >> end''',
                domain="data_engineering",
                subdomain="airflow",
                tags=["orchestration", "dag", "workflow"],
                difficulty="advanced"
            ),
        ]

    def _streaming_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a Spark Structured Streaming job for real-time data processing",
                code='''from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
from pyspark.sql.streaming import StreamingQuery
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session() -> SparkSession:
    """Create Spark session for streaming."""
    return (SparkSession.builder
        .appName("Real-time Analytics")
        .config("spark.sql.streaming.checkpointLocation", "/checkpoints")
        .config("spark.sql.streaming.schemaInference", "true")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate())


def define_event_schema() -> StructType:
    """Define schema for streaming events."""
    return StructType([
        StructField("event_id", StringType(), False),
        StructField("event_type", StringType(), False),
        StructField("user_id", StringType(), True),
        StructField("timestamp", TimestampType(), False),
        StructField("value", DoubleType(), True),
        StructField("properties", StringType(), True),
    ])


def read_from_kafka(spark: SparkSession, topic: str, bootstrap_servers: str):
    """Read stream from Kafka."""
    return (spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 10000)
        .load())


def parse_kafka_messages(df, schema: StructType):
    """Parse Kafka messages to structured data."""
    return (df
        .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)", "timestamp AS kafka_timestamp")
        .select(
            F.from_json(F.col("value"), schema).alias("data"),
            F.col("kafka_timestamp")
        )
        .select("data.*", "kafka_timestamp")
    )


def enrich_events(df):
    """Enrich streaming events."""
    return (df
        .withColumn("event_hour", F.hour(F.col("timestamp")))
        .withColumn("event_date", F.to_date(F.col("timestamp")))
        .withColumn("processing_time", F.current_timestamp())
        .withColumn("latency_ms",
            (F.col("processing_time").cast("long") - F.col("timestamp").cast("long")) * 1000)
        .withWatermark("timestamp", "10 minutes")
    )


def aggregate_windowed(df):
    """Create windowed aggregations."""
    return (df
        .groupBy(
            F.window(F.col("timestamp"), "5 minutes", "1 minute"),
            "event_type"
        )
        .agg(
            F.count("*").alias("event_count"),
            F.sum("value").alias("total_value"),
            F.avg("value").alias("avg_value"),
            F.approx_count_distinct("user_id").alias("unique_users"),
            F.max("latency_ms").alias("max_latency")
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "*"
        )
        .drop("window")
    )


def detect_anomalies(df):
    """Detect anomalies in streaming data."""
    # Calculate rolling stats
    return (df
        .withColumn("is_anomaly",
            F.when(
                (F.col("value") > F.lit(1000)) |
                (F.col("latency_ms") > F.lit(5000)),
                True
            ).otherwise(False))
    )


def write_to_console(df, output_mode: str = "update") -> StreamingQuery:
    """Write to console for debugging."""
    return (df.writeStream
        .outputMode(output_mode)
        .format("console")
        .option("truncate", "false")
        .trigger(processingTime="10 seconds")
        .start())


def write_to_kafka(df, topic: str, bootstrap_servers: str) -> StreamingQuery:
    """Write processed data back to Kafka."""
    return (df
        .selectExpr("CAST(event_id AS STRING) AS key", "to_json(struct(*)) AS value")
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", topic)
        .option("checkpointLocation", f"/checkpoints/{topic}")
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .start())


def write_to_delta(df, path: str, partition_cols: list = None) -> StreamingQuery:
    """Write to Delta Lake for real-time analytics."""
    writer = (df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", f"{path}/_checkpoints")
        .trigger(processingTime="1 minute"))

    if partition_cols:
        writer = writer.partitionBy(*partition_cols)

    return writer.start(path)


def write_aggregates_to_delta(df, path: str) -> StreamingQuery:
    """Write aggregated data with update mode."""
    return (df.writeStream
        .format("delta")
        .outputMode("complete")
        .option("checkpointLocation", f"{path}/_checkpoints")
        .trigger(processingTime="1 minute")
        .start(path))


def main():
    spark = create_spark_session()
    schema = define_event_schema()

    # Read from Kafka
    raw_stream = read_from_kafka(
        spark,
        topic="events",
        bootstrap_servers="kafka:9092"
    )

    # Parse and enrich
    parsed = parse_kafka_messages(raw_stream, schema)
    enriched = enrich_events(parsed)

    # Detect anomalies
    with_anomalies = detect_anomalies(enriched)

    # Create aggregations
    windowed_aggs = aggregate_windowed(enriched)

    # Multiple outputs
    # 1. Raw events to Delta
    raw_query = write_to_delta(
        with_anomalies,
        "/datalake/events/raw",
        partition_cols=["event_date", "event_type"]
    )

    # 2. Aggregations to Delta
    agg_query = write_aggregates_to_delta(
        windowed_aggs,
        "/datalake/events/aggregated"
    )

    # 3. Anomalies to Kafka for alerting
    anomalies = with_anomalies.filter(F.col("is_anomaly") == True)
    alert_query = write_to_kafka(
        anomalies,
        topic="anomaly-alerts",
        bootstrap_servers="kafka:9092"
    )

    # Wait for all queries
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()''',
                domain="data_engineering",
                subdomain="streaming",
                tags=["spark", "kafka", "real_time"],
                difficulty="advanced"
            ),
        ]

    def _data_quality_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a data quality validation framework with Great Expectations",
                code='''import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint import Checkpoint
import pandas as pd
from typing import Dict, List, Any
import json
from datetime import datetime


class DataQualityValidator:
    """Data quality validation framework using Great Expectations."""

    def __init__(self, context_root_dir: str = "great_expectations"):
        self.context = gx.get_context(context_root_dir=context_root_dir)

    def create_expectation_suite(self, suite_name: str, expectations: List[Dict]) -> None:
        """Create an expectation suite from a list of expectations."""

        suite = self.context.add_expectation_suite(suite_name)

        for exp in expectations:
            expectation_type = exp["expectation_type"]
            kwargs = exp.get("kwargs", {})

            suite.add_expectation(
                expectation_configuration=gx.core.ExpectationConfiguration(
                    expectation_type=expectation_type,
                    kwargs=kwargs
                )
            )

        self.context.save_expectation_suite(suite)

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        suite_name: str,
        data_asset_name: str = "dataframe"
    ) -> Dict[str, Any]:
        """Validate a pandas DataFrame against an expectation suite."""

        # Create batch request
        batch_request = RuntimeBatchRequest(
            datasource_name="runtime_datasource",
            data_connector_name="runtime_data_connector",
            data_asset_name=data_asset_name,
            runtime_parameters={"batch_data": df},
            batch_identifiers={"default_identifier_name": "default_identifier"},
        )

        # Run validation
        validator = self.context.get_validator(
            batch_request=batch_request,
            expectation_suite_name=suite_name,
        )

        results = validator.validate()

        return {
            "success": results.success,
            "statistics": results.statistics,
            "results": [
                {
                    "expectation": r.expectation_config.expectation_type,
                    "success": r.success,
                    "observed_value": r.result.get("observed_value"),
                }
                for r in results.results
            ]
        }

    def create_checkpoint(
        self,
        checkpoint_name: str,
        suite_name: str,
        datasource_name: str,
        data_asset_name: str
    ) -> None:
        """Create a checkpoint for automated validation."""

        checkpoint_config = {
            "name": checkpoint_name,
            "config_version": 1.0,
            "class_name": "Checkpoint",
            "run_name_template": "%Y%m%d-%H%M%S",
            "validations": [
                {
                    "batch_request": {
                        "datasource_name": datasource_name,
                        "data_connector_name": "default_inferred_data_connector_name",
                        "data_asset_name": data_asset_name,
                    },
                    "expectation_suite_name": suite_name,
                }
            ],
            "action_list": [
                {
                    "name": "store_validation_result",
                    "action": {"class_name": "StoreValidationResultAction"},
                },
                {
                    "name": "update_data_docs",
                    "action": {"class_name": "UpdateDataDocsAction"},
                },
            ],
        }

        self.context.add_or_update_checkpoint(**checkpoint_config)


def define_sales_expectations() -> List[Dict]:
    """Define expectations for sales data."""
    return [
        # Column existence
        {
            "expectation_type": "expect_table_columns_to_match_ordered_list",
            "kwargs": {
                "column_list": ["order_id", "customer_id", "product_id", "quantity", "price", "order_date"]
            }
        },

        # Primary key uniqueness
        {
            "expectation_type": "expect_column_values_to_be_unique",
            "kwargs": {"column": "order_id"}
        },

        # Not null constraints
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "order_id"}
        },
        {
            "expectation_type": "expect_column_values_to_not_be_null",
            "kwargs": {"column": "customer_id"}
        },

        # Value ranges
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "quantity", "min_value": 1, "max_value": 1000}
        },
        {
            "expectation_type": "expect_column_values_to_be_between",
            "kwargs": {"column": "price", "min_value": 0.01, "max_value": 100000}
        },

        # Date validity
        {
            "expectation_type": "expect_column_values_to_be_dateutil_parseable",
            "kwargs": {"column": "order_date"}
        },

        # Categorical values
        {
            "expectation_type": "expect_column_values_to_be_in_set",
            "kwargs": {"column": "status", "value_set": ["pending", "completed", "cancelled", "refunded"]}
        },

        # Row count
        {
            "expectation_type": "expect_table_row_count_to_be_between",
            "kwargs": {"min_value": 1, "max_value": 10000000}
        },

        # Freshness (custom)
        {
            "expectation_type": "expect_column_max_to_be_between",
            "kwargs": {
                "column": "order_date",
                "min_value": (datetime.now() - pd.Timedelta(days=1)).isoformat(),
                "parse_strings_as_datetimes": True
            }
        },
    ]


def validate_sales_data(df: pd.DataFrame) -> Dict:
    """Validate sales data with comprehensive checks."""

    validator = DataQualityValidator()

    # Create suite
    validator.create_expectation_suite(
        suite_name="sales_expectations",
        expectations=define_sales_expectations()
    )

    # Run validation
    results = validator.validate_dataframe(
        df=df,
        suite_name="sales_expectations",
        data_asset_name="sales_data"
    )

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall_success": results["success"],
        "statistics": results["statistics"],
        "failed_expectations": [
            r for r in results["results"] if not r["success"]
        ]
    }

    return report


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("sales_data.csv")
    report = validate_sales_data(df)

    if not report["overall_success"]:
        print("Data quality check FAILED!")
        print(json.dumps(report["failed_expectations"], indent=2))
        exit(1)
    else:
        print("Data quality check PASSED!")''',
                domain="data_engineering",
                subdomain="data_quality",
                tags=["great_expectations", "validation", "testing"],
                difficulty="intermediate"
            ),
        ]
