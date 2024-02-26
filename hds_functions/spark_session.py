"""
Module name: spark_session.py

This module provides functions to manage SparkSessions for PySpark applications.

"""
from pyspark.sql import SparkSession

def get_spark_session():
    """
    Creates or retrieves a SparkSession object.

    This function initializes a SparkSession with the specified app name if it does
    not exist; otherwise, it retrieves the existing SparkSession.

    Returns:
    - spark_session (SparkSession): A SparkSession object.

    Example:
        >>> spark = get_spark_session()
    """
    spark_session = (
        SparkSession.builder
        .appName('SparkSession')
        .getOrCreate()
    )

    return spark_session
