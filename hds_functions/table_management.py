"""
Module name: table_management.py

This module provides functions for managing tables within a PySpark environment.

Functions:
- load_table: Load a table from Spark, optionally filtering by archive date and standardising the columns.
- save_table: Save a DataFrame to a database table.
- standardise_table: Standardise the DataFrame according to the specified method, including renaming person ID variables,
    formatting dates into yyyy-MM-dd format, and cleaning column names.
"""
import pyspark.sql.functions as f
from pyspark.sql import DataFrame
from .json_utils import read_json_file
from .spark_session import get_spark_session
from .data_wrangling import clean_column_names

def load_table(table: str, table_directory: str = None, method: str = None) -> DataFrame:
    """
    Load a table from Spark, optionally filtering by archive date and standardising the data.

    Args:
        table (str): The name of the table to load.
        table_directory (str): Path to the JSON file containing table directories.
        method (str): The method used to standardise the table for column names (e.g. `person_id`).
            See `standardise_table` function for description.

    Returns:
        pyspark.sql.DataFrame: The loaded and optionally standardised DataFrame.

    Example:
        # Load table GDPPR with column names standardised
        df = load_table('gdppr', method='gdppr')
    """

    # Load table directory from JSON file
    if table_directory is None:
        table_directory = read_json_file('config/table_directory.json')
    elif isinstance(table_directory, str):
        table_directory = read_json_file(table_directory)
    else:
        raise ValueError("table_directory should be a string or None.")

    # Check table key exists
    assert table in table_directory.keys(), f"Table key '{table}' not found in table_directory"

    # Read parameters from table_directory
    database = table_directory[table]['database']
    table_name = table_directory[table]['table_name']
    archive_date = table_directory[table]['archive_date']

    # Get spark session
    spark = get_spark_session() 

    # Load table
    df = spark.table(f"{database}.{table_name}")

    # Filter for archive date
    if archive_date == 'latest':
        max_archive_date = (
            df
            .agg(f.max('archived_on').alias('_max_archive_date'))
            .collect()[0][0]
        )
        df = df.filter(f.col('archived_on') == f.lit(max_archive_date))
    elif archive_date is not None:
        df = df.filter(f.col('archived_on') == f.lit(archive_date))

    # Standardise table
    if method is not None:
        df = standardise_table(df, method=method)

    return df


def save_table(df, table:str, table_directory = None) -> None:
    """
    Saves a DataFrame to a database table.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame to be saved.
        table (str): The name of the table to save.
        table_directory (str): Path to the JSON file containing table directories.

    Returns:
        None
    """
    # Load table directory from JSON file
    if table_directory is None:
        table_directory = read_json_file('config/table_directory.json')
    elif isinstance(table_directory, str):
        table_directory = read_json_file(table_directory)
    else:
        raise ValueError("table_directory should be a string or None.")

    # Check table key exists
    assert table in table_directory.keys(), f"Table key '{table}' not found in table_directory"

    # Read parameters from table_directory
    database = table_directory[table]['database']
    table_name = table_directory[table]['table_name']

    # Write table to database
    df.write.mode('overwrite').option('overwriteSchema', 'True').saveAsTable(f"{database}.{table_name}")


def standardise_table(df, method):
    """
    Standardise the DataFrame according to the specified method. In general, these include the following steps:
    - Renaming Person ID Variables: Rename variables representing person IDs to 'person_id' for consistency across datasets.
    - Date Formatting: Format date variables into yyyy-MM-dd format to ensure uniformity and facilitate analysis.
    - Cleaning Column Names: Standardise column names by converting them to lowercase and replacing spaces with underscores.

    Args:
        df (pyspark.sql.DataFrame): The DataFrame to be standardized.
        method (str): The method used for standardization. Valid methods include: 
                      'deaths', 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'.

    Returns:
        pyspark.sql.DataFrame: The standardized DataFrame.
    """
    if method == 'deaths':
        return(standardise_deaths_table(df))
    elif method == 'gdppr':
        return(standardise_gdppr_table(df))
    elif method == 'hes_apc':
        return(standardise_hes_apc_table(df))
    elif method == 'hes_op':
        return(standardise_hes_op_table(df))
    elif method == 'hes_ae':
        return(standardise_hes_ae_table(df))
    elif method == 'ssnap':
        return(standardise_ssnap_table(df))
    else:
        raise ValueError(
            f"'{method}' is not a recognized standardize_table method. "
            f"Available methods: deaths, gdppr, hes_apc, hes_op, hes_ae, ssnap"
        )


def standardise_deaths_table(df):
    return(
        df
        .withColumnRenamed('DEC_CONF_NHS_NUMBER_CLEAN_DEID', 'person_id')
        .withColumnRenamed('REG_DATE_OF_DEATH', 'date_of_death')
        .transform(clean_column_names)
        .withColumn('reg_date', f.to_date(f.col('REG_DATE'), 'yyyyMMdd'))
        .withColumn(
            'date_of_death',
            f.when(
                f.col('date_of_death').rlike('\d{8}'),
                f.to_date(f.col('date_of_death'), 'yyyyMMdd')
            )
        )
    )

def standardise_gdppr_table(df):
    return(
        df
        .withColumnRenamed('NHS_NUMBER_DEID', 'person_id')
        .transform(clean_column_names)
    )

def standardise_hes_apc_table(df):
    return(
        df
        .withColumnRenamed('PERSON_ID_DEID', 'person_id')
        .transform(clean_column_names)
    )

def standardise_hes_op_table(df):
    return(
        df
        .withColumnRenamed('PERSON_ID_DEID', 'person_id')
        .transform(clean_column_names)
    )

def standardise_hes_ae_table(df):
    return(
        df
        .withColumnRenamed('PERSON_ID_DEID', 'person_id')
        .transform(clean_column_names)
    )

def standardise_ssnap_table(df):
    return(
        df
        .withColumnRenamed('Person_ID_DEID', 'person_id')
        .transform(clean_column_names)
    )
