from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from typing import List, Dict
from .table_management import load_table
from .table_management import save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row

def create_ethnicity_multisource(table_multisource: str = 'ethnicity_multisource', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing ethnicity data from multiple sources and save it to a table.

    Args:
        table_multisource (str, optional): The name of the table to save the consolidated data. Defaults to 'ethnicity_multisource'.
        extraction_methods (List[str], optional): List of methods for extracting ethnicity data. Defaults to None.

    Returns:
        None
    """
    if extraction_methods is None:
        extraction_methods = ['gdppr_snomed', 'gdppr', 'hes_apc', 'hes_op', 'hes_ae']

    # Extract ethnicity data from multiple sources
    ethnicity_from_sources = [extract_ethnicity(method) for method in extraction_methods]
    ethnicity_multisource = functools.reduce(DataFrame.unionByName, ethnicity_from_sources)

    # Save the consolidated data to a table
    save_table(ethnicity_multisource, table_multisource)


def extract_ethnicity(extract_method: str) -> DataFrame:
    """
    Extract ethnicity data based on the specified data source.

    Args:
        extract_method (str): The method to extract ethnicity data.
            Allowed values are: 'gdppr_snomed', 'gdppr', 'hes_apc', 'hes_op', 'hes_ae'.

    Returns:
        DataFrame: DataFrame containing the extracted ethnicity data from the selected method.
            The DataFrame includes columns for person ID, record date, ethnicity raw code (as seen in the dataset),
            description of raw code, ethnicity cateogry code (harmonised), category description, broad ethnicity grouping
            and data source.

    """

    extraction_methods = {
        'gdppr_snomed': {
            'extraction_function': gdppr_snomed_ethnicity,
            'data_source': 'gdppr',
            'load_method': 'gdppr'
        },
        'gdppr': {
            'extraction_function': gdppr_ethnicity,
            'data_source': 'gdppr',
            'load_method': 'gdppr'
        },
        'hes_apc': {
            'extraction_function': hes_apc_ethnicity,
            'data_source': 'hes_apc',
            'load_method': 'hes_apc'
        },
        'hes_op': {
            'extraction_function': hes_op_ethnicity,
            'data_source': 'hes_op',
            'load_method': 'hes_op'
        },
        'hes_ae': {
            'extraction_function': hes_ae_ethnicity,
            'data_source': 'hes_ae',
            'load_method': 'hes_ae'
        }
    }

    if extract_method not in extraction_methods:
        raise ValueError(f"Invalid extract_method: {extract_method}. Allowed values are: 'gdppr_snomed', 'gdppr', 'hes_apc', 'hes_op', 'hes_ae'.")

    return extraction_methods[extract_method]['extraction_function'](
        load_table(
            table=extraction_methods[extract_method]['data_source'],
            method=extraction_methods[extract_method]['load_method']
        )
    )


def gdppr_ethnicity(gdppr: DataFrame) -> DataFrame:
    """
    Process the ethnicity data from the GDPPR table, ensuring distinct records and mapping ethnicity codes to categories.

    Args:
        gdppr (DataFrame): DataFrame containing the GDPPR table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    dict_category_description = create_dict_from_csv(
        path='ethnicity_categories.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_broad_group = create_dict_from_csv(
        path='ethnicity_categories.csv', key_column='code', value_columns='ethnic_broad_group', repo='hds_reference_data'
    )

    ethnicity_gdppr = (
        gdppr
        .select(
            'person_id',
            f.col('reporting_period_end_date').alias('record_date'),
            f.col('ethnic').alias('ethnicity_raw_code')
        )
        .distinct()
        .transform(map_column_values, dict_category_description, column='ethnicity_raw_code', new_column='ethnicity_raw_description')
        .withColumn('ethnicity_mapped_code', f.col('ethnicity_raw_code'))
        .withColumn('ethnicity_mapped_description', f.col('ethnicity_raw_description'))
        .transform(map_column_values, dict_broad_group, column='ethnicity_mapped_code', new_column='ethnicity_broad_group')
        .withColumn('data_source', f.lit('gdppr'))
    )

    return ethnicity_gdppr


def gdppr_snomed_ethnicity(gdppr: DataFrame) -> DataFrame:
    """
    Process the ethnicity data from SNOMED codes in the GDPPR table, ensuring distinct records and mapping ethnicity codes to categories.

    Args:
        gdppr (DataFrame): DataFrame containing the GDPPR table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    codelist_ethnicity = (
        read_csv_file(path='ethnicity_categories_snomed_mapping.csv', repo='hds_reference_data')
    )

    dict_category_description = create_dict_from_csv(
        path='ethnicity_categories.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_broad_group = create_dict_from_csv(
        path='ethnicity_categories.csv', key_column='code', value_columns='ethnic_broad_group', repo='hds_reference_data'
    )

    ethnicity_gdppr_snomed = (
        gdppr
        .select(
            'person_id',
            f.col('date').alias('record_date'),
            f.col('code')
        )
        .join(codelist_ethnicity, on = 'code', how = 'inner')
        .distinct()
        .transform(map_column_values, dict_category_description, column='ethnicity_raw_code', new_column='ethnicity_raw_description')
        .withColumn('ethnicity_mapped_code', f.col('ethnicity_raw_code'))
        .withColumn('ethnicity_mapped_description', f.col('ethnicity_raw_description'))
        .transform(map_column_values, dict_broad_group, column='ethnicity_mapped_code', new_column='ethnicity_broad_group')
        .withColumn('data_source', f.lit('gdppr'))
    )

    return ethnicity_gdppr