from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from functools import reduce
from typing import List, Dict
from .table_management import load_table
from .table_management import save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row
from .data_wrangling import map_column_values

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
    sex_multisource = reduce(DataFrame.unionByName, sex_from_sources)

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


def create_sex_individual(
    table_multisource: str = 'sex_multisource',
    table_individual: str = 'sex_individual',
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    data_source: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1},
):
    """
    Wrapper function to create and save a table containing selected sex records for each individual.

    Args:
        table_multisource (str): Name of the multisource sex table.
        table_individual (str): Name of the individual sex table to be created.
        min_record_date (str, optional): Minimum record date to consider. Defaults to '1900-01-01'.
        max_record_date (str, optional): Maximum record date to consider. Defaults to 'current_date()'.
        data_source (List[str], optional): List of allowed data sources to consider when selecting sex records. 
            If specified, only records from the specified data sources will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): Priority index mapping data sources to priority levels.
            Sources not specified in the mapping will have a default priority level of 0.
            Defaults to {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1}.
    """

    # Load multisource sex table
    sex_multisource = load_table(table_multisource)

    # Select individual sex records
    sex_individual = sex_record_selection(
        sex_multisource,
        min_record_date=min_record_date,
        max_record_date=max_record_date,
        data_source=data_source,
        priority_index=priority_index
    )

    # Save individual sex table
    save_table(sex_individual, table_individual)


def sex_record_selection(
    sex_multisource: DataFrame,
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    data_source: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1},
) -> DataFrame:
    """
    Selects a single record for each individual from the multisource sex DataFrame based on specified criteria.

    Args:
        sex_multisource (DataFrame): DataFrame containing sex data from multiple sources.
        min_record_date (str, optional): Minimum record date to consider. Defaults to '1900-01-01'.
        max_record_date (str, optional): Maximum record date to consider. Defaults to 'current_date()'.
        data_source (List[str], optional): List of allowed data sources to consider when selecting sex records. 
            If specified, only records from the specified data sources will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): Priority index mapping data sources to priority levels.
            Sources not specified in the mapping will have a default priority level of 0.
            Defaults to {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1}.

    Returns:
        DataFrame: DataFrame containing the selected sex records for each individual.
    """

    # Validate data_source argument
    if data_source is not None:
        assert isinstance(data_source, list), "data_source must be a list."
        assert data_source is None or data_source, "data_source cannot be an empty list."
        allowed_sources = {'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'}
        invalid_sources = [str(source) for source in data_source if source not in allowed_sources or not isinstance(source, str)]
        assert not invalid_sources, f"Invalid data sources: {invalid_sources}. Allowed sources are: {allowed_sources}."

    # Filter out anomalous records
    sex_multisource = (
        sex_multisource
        .filter(
            (f.expr('sex IS NOT NULL'))
            & (f.expr('record_date IS NOT NULL'))
        )
    )

    # Apply date restrictions
    if min_record_date is not None:
        sex_multisource = (
            sex_multisource
            .withColumn('min_record_date', f.expr(parse_date_instruction(min_record_date)))
            .filter(f.expr('(record_date >= min_record_date)'))
        )
    
    if max_record_date is not None:
        sex_multisource = (
            sex_multisource
            .withColumn('max_record_date', f.expr(parse_date_instruction(max_record_date)))
            .filter(f.expr('(record_date <= max_record_date)'))
        )

    # Apply data source restrictions
    if data_source is not None:
        sex_multisource = (
            sex_multisource
            .filter(f.col('data_source').isin(data_source))
        )

    # Map source priority
    sex_multisource = (
        sex_multisource
        .transform(map_column_values, map_dict = priority_index, column = 'data_source', new_column = 'source_priority')
        .fillna({'source_priority': 0})
    )

    # Select record for each individual based on source priority and recency rules
    sex_individual = (
        sex_multisource
        .transform(
            first_row,
            partition_by = ['person_id'],
            order_by = [f.col('source_priority').desc(), f.col('record_date').desc(), 'data_source', 'sex', 'sex_code']
        )
    )

    # Select columns
    sex_individual = (
        sex_individual
        .select('person_id', 'sex', 'sex_code', 'record_date', 'data_source')
    )

    return sex_individual