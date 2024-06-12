from pyspark.sql import functions as f, DataFrame
from functools import reduce
from typing import List, Dict
from .table_management import load_table, save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row, first_dense_rank
from .data_wrangling import map_column_values

def create_lsoa_multisource(table_multisource: str = 'lsoa_multisource', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing LSOA data from multiple sources and save it to a table.

    LSOA sourced from gdppr_all_versions uses all archived versions.

    Args:
        table_multisource (str, optional): The name of the table to save the consolidated data. Defaults to 'lsoa_multisource'.
        extraction_methods (List[str], optional): List of methods for extracting LSOA data. Defaults to None.

    Returns:
        None
    """
    if extraction_methods is None:
        extraction_methods = ['gdppr_all_versions', 'hes_apc', 'hes_op', 'hes_ae', 'vaccine_status']

    # Extract LSOA data from multiple sources
    lsoa_from_sources = [extract_lsoa(method) for method in extraction_methods]
    lsoa_multisource = reduce(DataFrame.unionByName, lsoa_from_sources)

    # Save the consolidated data to a table
    save_table(lsoa_multisource, table_multisource)


def extract_lsoa(data_source: str) -> DataFrame:
    """
    Extract LSOA data based on the specified data source.

    Args:
        data_source (str): The data source to extract LSOA data from.
            Allowed values are: 'gdppr_all_versions', 'hes_apc', 'hes_op', 'hes_ae', 'vaccine_status'.

    Returns:
        DataFrame: DataFrame containing the extracted LSOA data from the selected source.
            The DataFrame includes columns for person ID, record date, LSOA and data source.

    """

    extraction_methods = {
        'gdppr': {
            'extraction_function': gdppr_lsoa,
            'data_source': 'gdppr_demographics',
            'load_method': None
        },
        'hes_apc': {
            'extraction_function': hes_apc_lsoa,
            'data_source': 'hes_apc',
            'load_method': 'hes_apc'
        },
        'hes_op': {
            'extraction_function': hes_op_lsoa,
            'data_source': 'hes_op',
            'load_method': 'hes_op'
        },
        'hes_ae': {
            'extraction_function': hes_ae_lsoa,
            'data_source': 'hes_ae',
            'load_method': 'hes_ae'
        },
        'ssnap': {
            'extraction_function': ssnap_lsoa,
            'data_source': 'ssnap',
            'load_method': 'ssnap'
        },
        'vaccine_status': {
            'extraction_function': vaccine_status_lsoa,
            'data_source': 'vaccine_status',
            'load_method': 'vaccine_status'
        }
    }

    if extract_method not in extraction_methods:
    raise ValueError(
        f"Invalid extract_method: {extract_method}. Allowed values are: "
        "'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap', 'vaccine_status'."
    )

    return extraction_methods[extract_method]['extraction_function'](
        load_table(
            table=extraction_methods[extract_method]['data_source'],
            method=extraction_methods[extract_method]['load_method']
        )
    )


def gdppr_lsoa(gdppr_demographics: DataFrame) -> DataFrame:
    """
    Process the LSOA data from the GDPPR demographics table, ensuring distinct records.

    Args:
        gdppr_demographics (DataFrame): DataFrame containing the gdppr_demographics table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    lsoa_gdppr = (
        gdppr_demographics
        .select(
            'person_id',
            f.col('reporting_period_end_date').alias('record_date'),
            'lsoa'
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (lsoa IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('gdppr'))
    )

    return lsoa_gdppr


def hes_apc_lsoa(hes_apc: DataFrame) -> DataFrame:
    """
    Process the LSOA data from the HES-APC (Admitted Patient Care) table, ensuring distinct records.

    Args:
        hes_apc (DataFrame): DataFrame containing the HES-APC table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    lsoa_hes_apc = (
        hes_apc
        .select(
            'person_id',
            f.col('epistart').alias('record_date'),
            f.col('lsoa11').alias('lsoa'),
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (lsoa IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('hes_apc'))
    )

    return lsoa_hes_apc


def hes_op_lsoa(hes_op: DataFrame) -> DataFrame:
    """
    Process the LSOA data from the HES-OP (Outpatients) table, ensuring distinct records.

    Args:
        hes_op (DataFrame): DataFrame containing the HES-OP table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    lsoa_hes_op = (
        hes_op
        .select(
            'person_id',
            f.col('apptdate').alias('record_date'),
            f.col('lsoa11').alias('lsoa')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (lsoa IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('hes_op'))
    )

    return lsoa_hes_op


def hes_ae_lsoa(hes_ae: DataFrame) -> DataFrame:
    """
    Process the LSOA data from the HES-A&E (Accident and Emergency) table, ensuring distinct records.

    Args:
        hes_ae (DataFrame): DataFrame containing the HES-AE table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    lsoa_hes_ae = (
        hes_ae
        .select(
            'person_id',
            f.col('arrivaldate').alias('record_date'),
            f.col('lsoa11').alias('lsoa')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (lsoa IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('hes_ae'))
    )

    return lsoa_hes_ae

def ssnap_lsoa(ssnap: DataFrame) -> DataFrame:
    """
    Process the LSOA data from the SSNAP (Sentinel Stroke National Audit Programme) table, ensuring distinct
    records.

    Args:
        ssnap (DataFrame): DataFrame containing the ssnap table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    lsoa_ssnap = (
        ssnap
        .select(
            'person_id',
            f.to_date('s1firstarrivaldatetime').alias('record_date'),
            f.col('lsoa_of_residence').alias('lsoa')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (lsoa IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('ssnap'))
    )

    return lsoa_ssnap


def vaccine_status_lsoa(vaccine_status: DataFrame) -> DataFrame:
    """
    Process the LSOA data from the vaccine_status (COVID-19 vaccination status) table, ensuring distinct
    records.

    Args:
        vaccine_status (DataFrame): DataFrame containing the vaccine_status table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    lsoa_vaccine_status = (
        vaccine_status
        .select(
            'person_id',
            f.col('recorded_date').alias('record_date'),
            f.col('lsoa').alias('lsoa')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (lsoa IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('vaccine_status'))
    )

    return lsoa_vaccine_status

