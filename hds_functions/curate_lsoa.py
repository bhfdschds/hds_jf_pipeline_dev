from pyspark.sql import functions as f, DataFrame, Window
from functools import reduce
from typing import List, Dict
from .table_management import load_table, save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row, first_dense_rank
from .data_wrangling import map_column_values

def create_lsoa_multisource(table_multisource: str = 'lsoa_multisource', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing LSOA data from multiple sources and save it to a table.

    Args:
        table_multisource (str, optional): The table key of the table to save the consolidated data. Defaults to 'lsoa_multisource'.
        extraction_methods (List[str], optional): List of methods for extracting LSOA data. Defaults to None.

    Returns:
        None
    """
    if extraction_methods is None:
        extraction_methods = ['gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'vaccine_status']

    # Extract LSOA data from multiple sources
    lsoa_from_sources = [extract_lsoa(method) for method in extraction_methods]
    lsoa_multisource = reduce(DataFrame.unionByName, lsoa_from_sources)

    # Save the consolidated data to a table
    save_table(lsoa_multisource, table_multisource)


def extract_lsoa(extract_method: str) -> DataFrame:
    """
    Extract LSOA data based on the specified data source.

    Args:
        extract_method (str): The method to extract LSOA data from.
            Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'vaccine_status'.

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
        'vaccine_status': {
            'extraction_function': vaccine_status_lsoa,
            'data_source': 'vaccine_status',
            'load_method': 'vaccine_status'
        }
    }

    if extract_method not in extraction_methods:
        raise ValueError(
            f"Invalid extract_method: {extract_method}. Allowed values are: "
            "'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'vaccine_status'."
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


def create_lsoa_individual(
    table_multisource: str = 'lsoa_multisource',
    table_individual: str = 'lsoa_individual',
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    filter_data_sources: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3},
) -> None:
    """
    Wrapper function to create and save a table containing selected LSOA records for each individual.

    Args:
        table_multisource (str): Table key of the multisource LSOA table.
        table_individual (str): Table key of the individual LSOA table to be created.
        min_record_date (str, optional): Expression for minimum record date. Defaults to '1900-01-01'.
        max_record_date (str, optional): Expression for maximum record date. Defaults to 'current_date()'.
        filter_data_sources (List[str], optional): List of data sources to include when selecting LSOA records. 
            If specified, only records in the list will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): A dictionary mapping data sources to their priority levels.
            Lower integer values indicate higher priority. Data sources not included in the dictionary 
            are deprioritised and assigned a null priority index. Defaults to the following priority mapping:
            {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3}.
    """

    # Load multisource LSOA table
    lsoa_multisource = load_table(table_multisource)

    # Select individual LSOA records
    lsoa_individual = lsoa_record_selection(
        lsoa_multisource,
        min_record_date=min_record_date,
        max_record_date=max_record_date,
        filter_data_sources=filter_data_sources,
        priority_index=priority_index
    )

    # Save individual LSOA table
    save_table(lsoa_individual, table_individual)


def lsoa_record_selection(
    lsoa_multisource: DataFrame,
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    filter_data_sources: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3},
) -> DataFrame:
    """
    Selects a single record for each individual from the multisource LSOA DataFrame based on specified criteria.

    Args:
        lsoa_multisource (DataFrame): DataFrame containing LSOA data from multiple sources.
        min_record_date (str, optional): Expression for minimum record date. Defaults to '1900-01-01'.
        max_record_date (str, optional): Expression for maximum record date. Defaults to 'current_date()'.
        filter_data_sources (List[str], optional): List of data sources to include when selecting ethnicity records. 
            If specified, only records in the list will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): A dictionary mapping data sources to their priority levels.
            Lower integer values indicate higher priority. Data sources not included in the dictionary 
            are deprioritised and assigned a null priority index. Defaults to the following priority mapping:
            {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3}.

    Returns:
        DataFrame: DataFrame containing the selected LSOA records for each individual.
    """

    # Allowed data sources
    allowed_sources = {'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'vaccine_status'}

    # Validate filter_data_sources argument
    if filter_data_sources is not None:
        # Ensure filter_data_sources is a list
        if not isinstance(filter_data_sources, list):
            raise ValueError("filter_data_sources must be a list.")
        
        # Ensure filter_data_sources is not an empty list
        if not filter_data_sources:
            raise ValueError("filter_data_sources cannot be an empty list.")
        
        # Check for invalid data sources
        invalid_sources = [str(source) for source in filter_data_sources if source not in allowed_sources or not isinstance(source, str)]
        if invalid_sources:
            raise ValueError(f"Invalid data sources: {invalid_sources}. Allowed sources are: {allowed_sources}.")

    # Validate priority_index argument
    if priority_index is not None:
        # Ensure that all values in priority_index are integers
        if not all(isinstance(value, int) for value in priority_index.values()):
            raise ValueError("Not all values in priority_index are integers.")
        
        # Ensure all keys in priority_index are valid data sources
        invalid_keys = [key for key in priority_index.keys() if key not in allowed_sources]
        if invalid_keys:
            raise ValueError(f"Invalid keys in priority_index: {invalid_keys}. Allowed keys are: {allowed_sources}.")

    # Filter out anomalous records
    lsoa_multisource = (
        lsoa_multisource
        .filter('(person_id IS NOT NULL) AND (lsoa IS NOT NULL) AND (record_date IS NOT NULL)')
    )

    # Apply date restrictions
    if min_record_date is not None:
        lsoa_multisource = (
            lsoa_multisource
            .withColumn('min_record_date', f.expr(parse_date_instruction(min_record_date)))
            .filter('(record_date >= min_record_date)')
        )
    
    if max_record_date is not None:
        lsoa_multisource = (
            lsoa_multisource
            .withColumn('max_record_date', f.expr(parse_date_instruction(max_record_date)))
            .filter('(record_date <= max_record_date)')
        )

    # Apply data source restrictions
    if filter_data_sources is not None:
        lsoa_multisource = (
            lsoa_multisource
            .filter(f.col('data_source').isin(filter_data_sources))
        )

    # Map source priority
    if priority_index is None:
        lsoa_multisource = (
            lsoa_multisource
            .withColumn('source_priority', f.lit(None))
        )
    else:
        lsoa_multisource = (
            lsoa_multisource
            .transform(map_column_values, map_dict = priority_index, column = 'data_source', new_column = 'source_priority')
        )

    # Select rows of 1st dense rank for each individual based on source priority and recency rules
    lsoa_ties = (
        lsoa_multisource
        .transform(
            first_dense_rank, n = 1,
            partition_by = ['person_id'],
            order_by = [f.col('record_date').desc(), f.col('source_priority').asc_nulls_last(), ]
        )
    )

    # Specify window function to collect ties
    _win_collect_ties = (
        Window
        .partitionBy('person_id')
        .orderBy('data_source', 'lsoa')
    )

    # Create tie flag and collect ties in arrays
    lsoa_ties = (
        lsoa_ties
        .withColumn(
            'lsoa_distinct_value',
            f.collect_set(f.col('lsoa')).over(_win_collect_ties)
        )
        .withColumn(
            'lsoa_tie_flag',
            f.when(f.size(f.col('lsoa_distinct_value')) > f.lit(1), f.lit(1))
        )
        .withColumn(
            'lsoa_tie_value',
            f.when(f.col('lsoa_tie_flag') == f.lit(1), f.collect_list(f.col('lsoa')).over(_win_collect_ties))
        )
        .withColumn(
            'lsoa_tie_data_source',
            f.when(f.col('lsoa_tie_flag') == f.lit(1), f.collect_list(f.col('data_source')).over(_win_collect_ties))
        )
    )

    # Randomly select record to break tie
    lsoa_individual = (
        lsoa_ties
        .transform(
            first_row, n = 1,
            partition_by = ['person_id'], order_by = [f.rand(seed = 124910)]
        )
    )

    # Select columns
    lsoa_individual = (
        lsoa_individual
        .select(
            'person_id', 'lsoa', 'record_date', 'data_source', 'lsoa_tie_flag', 'lsoa_tie_value', 'lsoa_tie_data_source'
        )
    )

    return lsoa_individual