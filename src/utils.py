import sys
sys.setrecursionlimit(10000)
from pyspark.sql import SparkSession
from config import SPARK_APP_NAME, SPARK_MASTER

def get_spark_session():
    """
    Windows-safe Spark session - no Hadoop native.
    """
    spark = SparkSession.getActiveSession()
    if spark:
        spark.stop()
    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master(SPARK_MASTER) \
        .config("spark.hadoop.fs.defaultFS", "file:///") \
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
        .config("spark.hadoop.io.native.lib.available", "false") \
        .config("spark.sql.parquet.enableVectorizedReader", "false") \
        .config("spark.sql.parquet.enableDictionary", "false") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.adaptive.enabled", "false") \
        .config("spark.sql.legacy.parquet.datetimeRebaseModeInWrite", "false") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.files.maxPartitionBytes", "33554432") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
        .config("spark.sql.warehouse.dir", "file:///C:/tmp/spark-warehouse") \
        .getOrCreate()
    return spark

