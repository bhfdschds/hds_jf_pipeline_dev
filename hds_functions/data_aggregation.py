"""
Module name: data_aggregation.py

Description:
A module for handling data aggregation tasks that may not be directly supported by PySpark's built-in functions

Functions:
- first_row: Selects the first 'n' rows for each partition in a PySpark DataFrame.
"""

from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from pyspark.sql import Window

def first_row(
    df, n = 1, partition_by = None, order_by = None, return_row_index = False, row_index_name = 'row_index'
) -> DataFrame:
    """
    Select the first 'n' rows for each partition in a PySpark DataFrame, optionally retaining the row indices.

    Args:
        df (pyspark.sql.DataFrame): The PySpark DataFrame to be processed.
        n (int): The number of rows to retain for each partition.
        partition_by (list, optional): A list of column names to partition the data. Default is None.
        order_by (list, optional): A list of column names to order the data within each partition. Default is None.
        return_row_index (bool, optional): Whether to include the row index column in the output DataFrame. Default is False.
        row_index_name (str, optional): The name of the row index column. Default is 'row_index'.

    Note:
        - If the order_by columns contain null values, it's important to specify the treatment of nulls explicitly. By default, null values are
            treated as smallest when using orderBy. To specify whether nulls should come first or last, consider using asc_nulls_last()
            or desc_nulls_first(), or a combination of asc(), desc(), nulls_first(), and nulls_last() functions.

    Returns:
        pyspark.sql.DataFrame: A new PySpark DataFrame containing the first 'n' rows for each partition.

    Example:
        >>> df = spark.createDataFrame([("A", 1), ("A", 2), ("A", 3), ("B", 4), ("B", 5), ("B", 6)], ["Group", "Value"])
        >>> result = first_row(df, n=2, partition_by=["Group"], order_by=["Value"])
        >>> result.show()
        +-----+-----+
        |Group|Value|
        +-----+-----+
        |    A|    1|
        |    A|    2|
        |    B|    4|
        |    B|    5|
        +-----+-----+
    """

    # Add input validation for n
    assert isinstance(n, int) and n > 0, 'n must be a positive, non-zero integer'

    # Add '_dummy_column' if partition_by is not provided
    if partition_by is None:
        partition_by = ['_dummy_column']
        df = df.withColumn('_dummy_column', f.lit(1))

    # Adjust the window specification for ordering in ascending order
    window_spec = Window.partitionBy(*partition_by)
    if order_by is not None:
        window_spec = window_spec.orderBy(*order_by)

    # Add row number column based on the window specification
    df = df.withColumn(row_index_name, f.row_number().over(window_spec))

    # Filter for the first 'n' rows after ordering in ascending order
    df = df.filter(f.col(row_index_name) <= n)

    # Drop unnecessary columns
    if not return_row_index:
        df = df.drop(row_index_name)
    if partition_by == ['_dummy_column']:
        df = df.drop('_dummy_column')

    return df