from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from typing import List, Dict
from .table_management import load_table
from .table_management import save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row

def create_sex_multisource(table_multisource: str = 'sex_multisource', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing sex data from multiple sources and save it to a table.

    Args:
        table_multisource (str, optional): The name of the table to save the consolidated data. Defaults to 'sex_multisource'.
        extraction_methods (List[str], optional): List of methods for extracting sex data. Defaults to None.

    Returns:
        None
    """
    if extraction_methods is None:
        extraction_methods = ['gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap']

    # Extract sex data from multiple sources
    sex_from_sources = [extract_sex(method) for method in extraction_methods]
    sex_multisource = functools.reduce(DataFrame.unionByName, sex_from_sources)

    # Save the consolidated data to a table
    save_table(sex_multisource, table_multisource)


def extract_sex(data_source: str) -> DataFrame:
    """
    Extract sex data based on the specified data source.

    Args:
        data_source (str): The data source to extract sex data from.
            Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'.

    Returns:
        DataFrame: DataFrame containing the extracted sex data from the selected source.
            The DataFrame includes columns for person ID, record date, sex, sex code and data source.

    """

    extraction_functions = {
        'gdppr': gdppr_sex,
        'hes_apc': hes_apc_sex,
        'hes_op': hes_op_sex,
        'hes_ae': hes_ae_sex,
        'ssnap': ssnap_sex
    }

    if data_source not in extraction_functions:
        raise ValueError(f"Invalid data source: {data_source}. Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'.")

    return extraction_functions[data_source](load_table(data_source, method=data_source))


def gdppr_sex(gdppr: DataFrame) -> DataFrame:
    """
    Process the sex data from the GDPPR table, ensuring distinct records and mapping sex codes to categories.

    Args:
        gdppr (DataFrame): DataFrame containing the GDPPR table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    sex_gdppr = (
        gdppr
        .select(
            'person_id',
            f.col('reporting_period_end_date').alias('record_date'),
            f.col('sex').alias('sex_code')
        )
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                '0': None,
                '1': 'M',
                '2': 'F',
                '9': 'I'
            },
            subset = ['sex']
        )
        .withColumn('data_source', f.lit('gdppr'))
    )

    return sex_gdppr


def hes_apc_sex(hes_apc: DataFrame) -> DataFrame:
    """
    Process the sex data from the HES-APC (Admitted Patient Care) table, ensuring distinct records and mapping
    sex codes to categories.

    Args:
        hes_apc (DataFrame): DataFrame containing the HES-APC table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    sex_hes_apc = (
        hes_apc
        .select(
            'person_id',
            f.col('epistart').alias('record_date'),
            f.col('sex').alias('sex_code')
        )
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                '0': None,
                '1': 'M',
                '2': 'F',
                '3': 'I',
                '9': 'I'
            },
            subset = ['sex']
        )
        .withColumn('data_source', f.lit('hes_apc'))
    )

    return sex_hes_apc


def hes_op_sex(hes_op: DataFrame) -> DataFrame:
    """
    Process the sex data from the HES-OP (Outpatients) table, ensuring distinct records and mapping
    sex codes to categories.

    Args:
        hes_op (DataFrame): DataFrame containing the HES-OP table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    sex_hes_op = (
        hes_op
        .select(
            'person_id',
            f.col('apptdate').alias('record_date'),
            f.col('sex').alias('sex_code')
        )
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                '0': None,
                '1': 'M',
                '2': 'F',
                '9': 'I'
            },
            subset = ['sex']
        )
        .withColumn('data_source', f.lit('hes_op'))
    )

    return sex_hes_op


def hes_ae_sex(hes_ae: DataFrame) -> DataFrame:
    """
    Process the sex data from the HES-A&E (Accident and Emergency) table, ensuring distinct records and mapping
    sex codes to categories.

    Args:
        hes_ae (DataFrame): DataFrame containing the HES-AE table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    sex_hes_ae = (
        hes_ae
        .select(
            'person_id',
            f.col('arrivaldate').alias('record_date'),
            f.col('sex').alias('sex_code')
        )
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                '0': None,
                '1': 'M',
                '2': 'F',
                '3': 'I',
                '9': 'I'
            },
            subset = ['sex']
        )
        .withColumn('data_source', f.lit('hes_ae'))
    )

    return sex_hes_ae


def ssnap_sex(ssnap: DataFrame) -> DataFrame:
    """
    Process the sex data from the SSNAP (Sentinel Stroke National Audit Programme) table, ensuring distinct
    records and mapping sex codes to categories.

    Args:
        ssnap (DataFrame): DataFrame containing the SSNAP table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    sex_ssnap = (
        ssnap
        .select(
            'person_id',
            f.to_date('s1firstarrivaldatetime').alias('record_date'),
            f.col('s1gender').alias('sex_code')
        )
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                'M': 'M',
                'F': 'F',
            },
            subset = ['sex']
        )
        .withColumn('data_source', f.lit('ssnap'))
    )

    return sex_ssnap

