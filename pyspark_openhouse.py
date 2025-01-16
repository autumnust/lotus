from pyspark.sql import SparkSession
from examples.op_examples.openhouse_connector import SparkConnector, OpenHouse
import pandas as pd

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("OpenHouse-Example") \
    .config("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
    .config("spark.sql.catalog.spark_catalog.type", "hadoop") \
    .config("spark.sql.catalog.spark_catalog.warehouse", "$PWD/warehouse") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", "$PWD/local/warehouse") \
    .getOrCreate()

# Create `employees` table in Spark
spark.sql("CREATE TABLE IF NOT EXISTS local.db.employees (id INT, name STRING, age INT) USING iceberg")

# Initialize OpenHouse with SparkConnector
openhouse = OpenHouse(SparkConnector(spark))

# Create sample data as a pandas DataFrame
data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Emma", "Michael", "Sophia", "James", "Olivia"],
    "age": [27, 33, 29, 45, 31]
})

# trivial example, skip index_cols
# Note that this thing cannot handle the case where table don't exist.
openhouse.table("local.db.employees").write(data)

# Get the table using OpenHouse
employees = openhouse.table("local.db.employees")

# Demonstrate various operations
# 1. View first few rows
print("First 3 rows:")
print(employees.head(3))

# 2. View last few rows
print("\nLast 2 rows:")
print(employees.tail(2))

# 3. Get basic statistics
print("\nDescriptive statistics:")
print(employees.describe())

# Clean up
spark.stop()