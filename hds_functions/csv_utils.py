"""
Module name: csv_utils.py

Description:
    This module provides utilities for reading and writing CSV files in Python, with support for Spark DataFrame.
"""
import os
import pandas as pd
from pyspark.sql import DataFrame
from .environment_utils import get_spark_session
from .environment_utils import resolve_path

def read_csv_file(path: str, repo: str = None, keep_default_na: bool =False, **kwargs) -> DataFrame:
    """
    Read .csv file and return as Spark DataFrame
    
    Args:
    - path (str): Path to the CSV file. Can be an absolute path, a relative path with './', or a path within a repo.
    - repo (str): Name of the repo if the path is relative within a repo.
    - keep_default_na (boolean): Whether or not to include the default NaN values when parsing the data.
    - **kwargs: Additional keyword arguments to be passed to pd.read_csv()
    
    Returns:
    - spark_df (DataFrame): Spark DataFrame containing the data read from the CSV file

    Example:
        >>> read_csv_file('./relative/path/in/project.csv')
        >>> read_csv_file('/Workspace/absolute/path.csv')
        >>> read_csv_file(path='path/in/repo.csv', repo='common_repo')
    """
    # Resolve the path
    resolved_path = resolve_path(path, repo)
    
    # Read the CSV file into a Pandas DataFrame, passing additional arguments
    pandas_df = pd.read_csv(resolved_path, keep_default_na=keep_default_na, **kwargs)
    
    # Get spark session
    spark = get_spark_session()

    # Convert the Pandas DataFrame to a Spark DataFrame
    spark_df = spark.createDataFrame(pandas_df)
    
    return spark_df


def write_csv_file(df: DataFrame, path: str, repo: str = None, index: bool = False, max_rows_threshold: int = 1000, **kwargs) -> None:
    """
    Write a Spark DataFrame to a CSV file.

    Args:
    - df (DataFrame): The Spark DataFrame to be written to a CSV file.
    - path (str): Path to the CSV file. Can be an absolute path, a relative path with './', or a path within a repo.
    - repo (str): Name of the repo if the path is relative within a repo.
    - index (bool): Whether to include the index (row numbers) in the CSV file. Default is False.
    - max_rows_threshold (int): The maximum number of rows allowed in the DataFrame before raising an error. Default is 1000.
    - **kwargs: Additional keyword arguments to be passed to pd.DataFrame.to_csv().

    Returns:
    - None

    Raises:
    - ValueError: If the DataFrame exceeds the maximum rows threshold, the directory does not exist, or the DataFrame is empty.
    - IOError: If there is an issue writing the CSV file.

    Example:
        >>> write_csv_file(spark_df, './relative/path/in/project.csv')
        >>> write_csv_file(spark_df, '/Workspace/absolute/path.csv')
        >>> write_csv_file(spark_df, path='path/in/repo.csv', repo='common_repo')
    """
    # Check if DataFrame exceeds the maximum rows threshold
    if df.count() > max_rows_threshold:
        raise ValueError(f"DataFrame exceeds maximum rows threshold of {max_rows_threshold}. "
                         "This function is not meant for writing large datasets. "
                         "Consider using save_table() function to save to the database.")

    # Resolve the path
    resolved_path = resolve_path(path, repo)

    # Check if the directory exists
    directory = os.path.dirname(resolved_path)
    if not os.path.exists(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")

    # Check if DataFrame is empty
    if df.count() == 0:
        raise ValueError("DataFrame is empty")

    try:
        # Convert the Spark DataFrame to a Pandas DataFrame and write it to a CSV file
        df.toPandas().to_csv(resolved_path, index=index, **kwargs)
    except Exception as e:
        raise IOError(f"Error writing DataFrame to CSV file: {str(e)}")
