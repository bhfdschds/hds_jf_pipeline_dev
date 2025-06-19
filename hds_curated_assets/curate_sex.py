from pyspark.sql import functions as f, DataFrame, Window
from functools import reduce
from typing import List, Dict
from .table_management import load_table, save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row, first_dense_rank
from .data_wrangling import map_column_values

def create_sex_multisource(table_multisource: str = 'sex_multisource', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing sex data from multiple sources and save it to a table.

    Args:
        table_multisource (str, optional): The table key of the table to save the consolidated data. Defaults to 'sex_multisource'.
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


def extract_sex(extract_method: str) -> DataFrame:
    """
    Extract sex data based on the specified data source.

    Args:
        extract_method (str): The method to extract sex data.
            Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'.

    Returns:
        DataFrame: DataFrame containing the extracted sex data from the selected source.
            The DataFrame includes columns for person ID, record date, sex, sex code and data source.

    """

    extraction_methods = {
        'gdppr': {
            'extraction_function': gdppr_sex,
            'data_source': 'gdppr_demographics',
            'load_method': None
        },
        'hes_apc': {
            'extraction_function': hes_apc_sex,
            'data_source': 'hes_apc',
            'load_method': 'hes_apc'
        },
        'hes_op': {
            'extraction_function': hes_op_sex,
            'data_source': 'hes_op',
            'load_method': 'hes_op'
        },
        'hes_ae': {
            'extraction_function': hes_ae_sex,
            'data_source': 'hes_ae',
            'load_method': 'hes_ae'
        },
        'ssnap': {
            'extraction_function': ssnap_sex,
            'data_source': 'ssnap',
            'load_method': 'ssnap'
        }
    }

    if extract_method not in extraction_methods:
        raise ValueError(f"Invalid extract_method: {extract_method}. Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'.")

    return extraction_methods[extract_method]['extraction_function'](
        load_table(
            table=extraction_methods[extract_method]['data_source'],
            method=extraction_methods[extract_method]['load_method']
        )
    )

def gdppr_sex(gdppr_demographics: DataFrame) -> DataFrame:
    """
    Process the sex data from the GDPPR table, ensuring distinct records and mapping sex codes to categories.

    Args:
        gdppr_demographics (DataFrame): DataFrame containing GDPPR demographics data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    sex_gdppr = (
        gdppr_demographics
        .select(
            'person_id',
            f.col('reporting_period_end_date').alias('record_date'),
            f.col('sex').alias('sex_code')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (sex_code IS NOT NULL)")
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
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (sex_code IS NOT NULL)")
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                '0': None,
                '1': 'M',
                '2': 'F',
                '3': 'I',
                '9': 'I',
                'X': None
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
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (sex_code IS NOT NULL)")
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                '0': None,
                '1': 'M',
                '2': 'F',
                '9': 'I',
                'X': None
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
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (sex_code IS NOT NULL)")
        .distinct()
        .withColumn('sex', f.col('sex_code'))
        .replace(
            {
                '0': None,
                '1': 'M',
                '2': 'F',
                '3': 'I',
                '9': 'I',
                'X': None
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
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (sex_code IS NOT NULL)")
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
    filter_data_sources: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3},
) -> None:
    """
    Wrapper function to create and save a table containing selected sex records for each individual.

    Args:
        table_multisource (str): Table key of the multisource sex table.
        table_individual (str): Table key of the individual sex table to be created.
        min_record_date (str, optional): Expression for minimum record date. Defaults to '1900-01-01'.
        max_record_date (str, optional): Expression for maximum record date. Defaults to 'current_date()'.
        filter_data_sources (List[str], optional): List of data sources to include when selecting sex records. 
            If specified, only records in the list will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): A dictionary mapping data sources to their priority levels.
            Lower integer values indicate higher priority. Data sources not included in the dictionary 
            are deprioritised and assigned a null priority index. Defaults to the following priority mapping:
            {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3}.

    Note:
        The parameters min_record_date and max_record_date are passed to parse_date_instruction() before being processed
        by Pyspark's expr() function.
    """

    # Load multisource sex table
    sex_multisource = load_table(table_multisource)

    # Select individual sex records
    sex_individual = sex_record_selection(
        sex_multisource,
        min_record_date=min_record_date,
        max_record_date=max_record_date,
        filter_data_sources=filter_data_sources,
        priority_index=priority_index
    )

    # Save individual sex table
    save_table(sex_individual, table_individual)


def sex_record_selection(
    sex_multisource: DataFrame,
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    filter_data_sources: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3},
) -> DataFrame:
    """
    Selects a single record for each individual from the multisource sex DataFrame based on specified criteria.

    Args:
        sex_multisource (DataFrame): DataFrame containing sex data from multiple sources.
        min_record_date (str, optional): Expression for minimum record date. Defaults to '1900-01-01'.
        max_record_date (str, optional): Expression for maximum record date. Defaults to 'current_date()'.
        filter_data_sources (List[str], optional): List of data sources to include when selecting sex records. 
            If specified, only records in the list will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): A dictionary mapping data sources to their priority levels.
            Lower integer values indicate higher priority. Data sources not included in the dictionary 
            are deprioritised and assigned a null priority index. Defaults to the following priority mapping:
            {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3}.

    Returns:
        DataFrame: DataFrame containing the selected sex records for each individual.

    Note:
        The parameters min_record_date and max_record_date are passed to parse_date_instruction() before being processed
        by Pyspark's expr() function.
    """

    # Allowed data sources
    allowed_sources = {'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'}

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
    sex_multisource = (
        sex_multisource
        .filter('(person_id IS NOT NULL) AND (sex IS NOT NULL) AND (record_date IS NOT NULL)')
    )

    # Apply date restrictions
    if min_record_date is not None:
        sex_multisource = (
            sex_multisource
            .withColumn('min_record_date', f.expr(parse_date_instruction(min_record_date)))
            .filter('(record_date >= min_record_date)')
        )
    
    if max_record_date is not None:
        sex_multisource = (
            sex_multisource
            .withColumn('max_record_date', f.expr(parse_date_instruction(max_record_date)))
            .filter('(record_date <= max_record_date)')
        )

    # Apply data source restrictions
    if filter_data_sources is not None:
        sex_multisource = (
            sex_multisource
            .filter(f.col('data_source').isin(filter_data_sources))
        )

    # Map source priority
    sex_multisource = (
        sex_multisource
        .transform(map_column_values, map_dict = priority_index, column = 'data_source', new_column = 'source_priority')
    )

    # Select rows of 1st dense rank for each individual based on source priority and recency rules
    sex_ties = (
        sex_multisource
        .transform(
            first_dense_rank, n = 1,
            partition_by = ['person_id'],
            order_by = [f.col('source_priority').asc_nulls_last(), f.col('record_date').desc()]
        )
    )

    # Specify window function to collect ties
    _win_collect_ties = (
        Window
        .partitionBy('person_id')
        .orderBy('data_source', 'sex_code', 'sex')
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    # Create tie flag and collect ties in arrays
    sex_ties = (
        sex_ties
        .withColumn(
            'sex_distinct_value',
            f.collect_set(f.col('sex')).over(_win_collect_ties)
        )
        .withColumn(
            'sex_tie_flag',
            f.when(f.size(f.col('sex_distinct_value')) > f.lit(1), f.lit(1))
        )
        .withColumn(
            'sex_code_tie_value',
            f.when(f.col('sex_tie_flag') == f.lit(1), f.collect_list(f.col('sex_code')).over(_win_collect_ties))
        )
        .withColumn(
            'sex_tie_value',
            f.when(f.col('sex_tie_flag') == f.lit(1), f.collect_list(f.col('sex')).over(_win_collect_ties))
        )
        .withColumn(
            'sex_tie_data_source',
            f.when(f.col('sex_tie_flag') == f.lit(1), f.collect_list(f.col('data_source')).over(_win_collect_ties))
        )
    )

    # Randomly select record to break tie
    sex_individual = (
        sex_ties
        .transform(
            first_row, n = 1,
            partition_by = ['person_id'], order_by = [f.rand(seed = 124910)]
        )
    )

    # Select columns
    sex_individual = (
        sex_individual
        .select(
            'person_id', 'sex_code', 'sex', 'record_date', 'data_source',
            'sex_tie_flag', 'sex_code_tie_value', 'sex_tie_value', 'sex_tie_data_source'
        )
    )

    return sex_individual