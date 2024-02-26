"""
Module name: csv_utils.py

Description:
    This module provides utilities for reading and writing CSV files in Python, with support for Spark DataFrame.
"""
import pandas as pd
from pyspark.sql import DataFrame
from .spark_session import get_spark_session

def read_csv_file(path: str, keep_default_na=False, **kwargs) -> DataFrame:
    """
    Read .csv file and return as Spark DataFrame
    
    Args:
    - path (str): Path to the CSV file
    - keep_default_na (boolean): Whether or not to include the default NaN values when parsing the data.
    - **kwargs: Additional keyword arguments to be passed to pd.read_csv()
    
    Returns:
    - spark_df (DataFrame): Spark DataFrame containing the data read from the CSV file
    """
    # Read the CSV file into a Pandas DataFrame, passing additional arguments
    pandas_df = pd.read_csv(path, keep_default_na=keep_default_na, **kwargs)
    
    # Get spark session
    spark = get_spark_session()

    # Convert the Pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)
    
    return spark_df


def write_csv_file(df, path, index=False, max_rows_threshold=1000, **kwargs) -> None:
    """
    Write a Spark DataFrame to a CSV file.

    Args:
    - df (DataFrame): The Spark DataFrame to be written to a CSV file.
    - path (str): The path where the CSV file will be saved.
    - index (bool): Whether to include the index (row numbers) in the CSV file. Default is False.
    - max_rows_threshold (int): The maximum number of rows allowed in the DataFrame before raising an error. Default is 1000.
    - **kwargs: Additional keyword arguments to be passed to pd.DataFrame.to_csv().

    Returns:
    - None

    Raises:
    - ValueError: If the DataFrame exceeds the maximum rows threshold.
    - IOError: If there is an issue writing the CSV file.

    Example:
        >>> write_csv_file(spark_df, '/path/to/output.csv')
    """
    # Check if DataFrame exceeds the maximum rows threshold
    if df.count() > max_rows_threshold:
        raise ValueError(f"DataFrame exceeds maximum rows threshold of {max_rows_threshold}. "
                         "This function is not meant for writing large datasets. "
                         "Consider using save_table() function to save to the database.")

    # Check if DataFrame is empty
    if df.count() == 0:
        raise ValueError("DataFrame is empty")

    try:
        # Convert the Spark DataFrame to a Pandas DataFrame and write it to a CSV file
        df.toPandas().to_csv(path, index=index, **kwargs)
    except Exception as e:
        raise IOError(f"Error writing DataFrame to CSV file: {str(e)}")
