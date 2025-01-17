from pyspark.sql import SparkSession
from openhouse_connector import SparkConnector, OpenHouse
import pandas as pd
import os

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("OpenHouse-Example") \
    .config("spark.jars.packages", "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1") \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkSessionCatalog") \
    .config("spark.sql.catalog.spark_catalog.type", "hadoop") \
    .config("spark.sql.catalog.spark_catalog.warehouse", f"{os.getcwd()}/warehouse") \
    .config("spark.sql.catalog.local", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.local.type", "hadoop") \
    .config("spark.sql.catalog.local.warehouse", f"{os.getcwd()}/local/warehouse") \
    .getOrCreate()

# Create `employees` table in Spark
spark.sql("CREATE DATABASE IF NOT EXISTS db")
spark.sql("DROP TABLE IF EXISTS db.employees")
spark.sql("CREATE TABLE IF NOT EXISTS db.employees (id INT, name STRING, age INT) USING iceberg")

# Initialize OpenHouse with SparkConnector
openhouse = OpenHouse(SparkConnector(spark))

# Create sample data as a pandas DataFrame
data = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "name": ["Emma", "Michael", "Sophia", "James", "Olivia"],
    "age": [27, 33, 29, 45, 31]
})

data2 = pd.DataFrame({
    "id": [6, 7, 8, 9, 10],
    "name": ["Lisa", "Omar", "Nina", "Peter", "Rachel"],
    "age": [31, 45, 28, 39, 33]
})

table = openhouse.table("db.employees")
table.write(data, index_cols=["name"])
search_df = table.load().sem_search("name", "Who's emma?", K=2, n_rerank=4)
print(search_df)

# Clean up
spark.stop()