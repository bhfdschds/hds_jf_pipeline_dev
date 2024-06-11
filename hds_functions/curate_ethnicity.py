from pyspark.sql import functions as f, DataFrame, Window
from functools import reduce
from typing import List, Dict
from .table_management import load_table, save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row, first_dense_rank
from .data_wrangling import map_column_values
from .csv_utils import read_csv_file, create_dict_from_csv

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
    ethnicity_multisource = reduce(DataFrame.unionByName, ethnicity_from_sources)

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
            'data_source': 'gdppr_demographics',
            'load_method': None
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


def gdppr_snomed_ethnicity(gdppr: DataFrame) -> DataFrame:
    """
    Process the ethnicity data from SNOMED codes in the GDPPR table, ensuring distinct records and mapping ethnicity codes to categories.

    Args:
        gdppr (DataFrame): DataFrame containing the GDPPR table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    codelist_ethnicity = (
        read_csv_file(path='ethnicity_mapping/ethnicity_categories_snomed_mapping.csv', repo='hds_reference_data')
        .select('code', 'description', 'ethnic_category')
    )

    dict_category_description = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_broad_group = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='ethnic_broad_group', repo='hds_reference_data'
    )

    ethnicity_gdppr_snomed = (
        gdppr
        .select('person_id', 'date', 'record_date', 'code')
        .withColumn(
            'record_date',
            f.when(
                (f.col('record_date').isNull()) | (f.col('record_date') == f.to_date(f.lit('1899-12-30'))),
                f.col('date')
            )
            .otherwise(f.col('record_date'))
        )
        .drop('date')
        .join(codelist_ethnicity, on = 'code', how = 'inner')
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL)")
        .distinct()
        .withColumnRenamed('code', 'ethnicity_raw_code')
        .withColumnRenamed('description', 'ethnicity_raw_description')
        .withColumnRenamed('ethnic_category', 'ethnicity_18_code')
        .transform(map_column_values, dict_category_description, column='ethnicity_18_code', new_column='ethnicity_18_group')
        .transform(map_column_values, dict_broad_group, column='ethnicity_18_code', new_column='ethnicity_5_group')
        .withColumn('data_source', f.lit('gdppr_snomed'))
    )

    return ethnicity_gdppr_snomed


def gdppr_ethnicity(gdppr: DataFrame) -> DataFrame:
    """
    Process the ethnicity data from the GDPPR table, ensuring distinct records and mapping ethnicity codes to categories.

    Args:
        gdppr (DataFrame): DataFrame containing the GDPPR table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    dict_category_description = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_broad_group = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='ethnic_broad_group', repo='hds_reference_data'
    )

    ethnicity_gdppr = (
        gdppr
        .select(
            'person_id',
            f.col('reporting_period_end_date').alias('record_date'),
            f.col('ethnic').alias('ethnicity_raw_code')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (ethnicity_raw_code IS NOT NULL)")
        .distinct()
        .transform(map_column_values, dict_category_description, column='ethnicity_raw_code', new_column='ethnicity_raw_description')
        .withColumn('ethnicity_18_code', f.col('ethnicity_raw_code'))
        .withColumn('ethnicity_18_group', f.col('ethnicity_raw_description'))
        .transform(map_column_values, dict_broad_group, column='ethnicity_18_code', new_column='ethnicity_5_group')
        .withColumn('data_source', f.lit('gdppr'))
    )

    return ethnicity_gdppr


def hes_apc_ethnicity(hes_apc: DataFrame) -> DataFrame:
    """
    Process the ethnicity data from the HES-APC (Admitted Patient Care) table, ensuring distinct records and mapping
    ethnicity codes to categories.

    Args:
        hes_apc (DataFrame): DataFrame containing the HES-APC table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    
    dict_pre_2003_description = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories_pre_2003.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_ethnicity_18_description = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_combined_description = {**dict_pre_2003_description, **dict_ethnicity_18_description}

    dict_pre_2003_to_ethnicity_18_mapping = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories_pre_2003.csv', key_column='code', value_columns='ethnic_category', repo='hds_reference_data'
    )

    dict_broad_group = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='ethnic_broad_group', repo='hds_reference_data'
    )

    ethnicity_hes_apc = (
        hes_apc
        .select(
            'person_id',
            f.col('epistart').alias('record_date'),
            f.col('ethnos').alias('ethnicity_raw_code')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (ethnicity_raw_code IS NOT NULL)")
        .distinct()
        .transform(
            map_column_values,
            dict_combined_description,
            column='ethnicity_raw_code',
            new_column='ethnicity_raw_description'
        )
        .withColumn('ethnicity_18_code', f.col('ethnicity_raw_code'))
        .replace(
            dict_pre_2003_to_ethnicity_18_mapping,
            subset = ['ethnicity_18_code']
        )
        .transform(
            map_column_values,
            dict_ethnicity_18_description,
            column='ethnicity_18_code',
            new_column='ethnicity_18_group'
        )
        .transform(
            map_column_values,
            dict_broad_group,
            column='ethnicity_18_code',
            new_column='ethnicity_5_group'
        )
        .withColumn('data_source', f.lit('hes_apc'))
    )

    return ethnicity_hes_apc


def hes_op_ethnicity(hes_op: DataFrame) -> DataFrame:
    """
    Process the ethnicity data from the HES-OP (Outpatients) table, ensuring distinct records and mapping
    ethnicity codes to categories.

    Args:
        hes_op (DataFrame): DataFrame containing the HES-OP table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    dict_ethnicity_18_description = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_broad_group = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='ethnic_broad_group', repo='hds_reference_data'
    )

    ethnicity_hes_op = (
        hes_op
        .select(
            'person_id',
            f.col('apptdate').alias('record_date'),
            f.col('ethnos').alias('ethnicity_raw_code')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (ethnicity_raw_code IS NOT NULL)")
        .distinct()
        .transform(
            map_column_values,
            dict_ethnicity_18_description,
            column='ethnicity_raw_code',
            new_column='ethnicity_raw_description'
        )
        .withColumn('ethnicity_18_code', f.col('ethnicity_raw_code'))
        .withColumn('ethnicity_18_group', f.col('ethnicity_raw_description'))
        .transform(
            map_column_values,
            dict_broad_group,
            column='ethnicity_18_code',
            new_column='ethnicity_5_group'
        )
        .withColumn('data_source', f.lit('hes_op'))
    )

    return ethnicity_hes_op


def hes_ae_ethnicity(hes_ae: DataFrame) -> DataFrame:
    """
    Process the ethnicity data from the HES-A&E (Accident and Emergency) table, ensuring distinct records and mapping
    ethnicity codes to categories.

    Args:
        hes_ae (DataFrame): DataFrame containing the HES-AE table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    dict_ethnicity_18_description = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='description', repo='hds_reference_data'
    )

    dict_broad_group = create_dict_from_csv(
        path='ethnicity_mapping/ethnicity_categories.csv', key_column='code', value_columns='ethnic_broad_group', repo='hds_reference_data'
    )

    ethnicity_hes_ae = (
        hes_ae
        .select(
            'person_id',
            f.col('arrivaldate').alias('record_date'),
            f.col('ethnos').alias('ethnicity_raw_code')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (ethnicity_raw_code IS NOT NULL)")
        .distinct()
        .transform(
            map_column_values,
            dict_ethnicity_18_description,
            column='ethnicity_raw_code',
            new_column='ethnicity_raw_description'
        )
        .withColumn('ethnicity_18_code', f.col('ethnicity_raw_code'))
        .withColumn('ethnicity_18_group', f.col('ethnicity_raw_description'))
        .transform(
            map_column_values,
            dict_broad_group,
            column='ethnicity_18_code',
            new_column='ethnicity_5_group'
        )
        .withColumn('data_source', f.lit('hes_ae'))
    )

    return ethnicity_hes_ae


def create_ethnicity_individual(
    table_multisource: str = 'ethnicity_multisource',
    table_individual: str = 'ethnicity_individual',
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    data_source: List[str] = None,
    ethnicity_18_null_codes: List[str] = ['', 'X', 'Z', '99'],
    priority_index: Dict[str, int] = {'gdppr_snomed': 1, 'gdppr': 2, 'hes_apc': 3, 'hes_op': 4, 'hes_ae': 4},
):
    """
    Wrapper function to create and save a table containing selected ethnicity records for each individual.

    Args:
        table_multisource (str): Name of the multisource ethnicity table.
        table_individual (str): Name of the individual ethnicity table to be created.
        min_record_date (str, optional): Minimum record date to consider. Defaults to '1900-01-01'.
        max_record_date (str, optional): Maximum record date to consider. Defaults to 'current_date()'.
        data_source (List[str], optional): List of allowed data sources to consider when selecting ethnicity records. 
            If specified, only records from the specified data sources will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        ethnicity_18_null_codes (List[str], optional): List of indeterminate ethnicity_18 codes that will be removed from selection.  
        priority_index (Dict[str, int], optional): Priority mapping for data sources; lower indices are prioritised.
            Defaults to {'gdppr_snomed': 1, 'gdppr': 2, 'hes_apc': 3, 'hes_op': 4, 'hes_ae': 4}.
    """

    # Load multisource ethnicity table
    ethnicity_multisource = load_table(table_multisource)

    # Select individual ethnicity records
    ethnicity_individual = ethnicity_record_selection(
        ethnicity_multisource,
        min_record_date=min_record_date,
        max_record_date=max_record_date,
        data_source=data_source,
        ethnicity_18_null_codes=ethnicity_18_null_codes,
        priority_index=priority_index
    )

    # Save individual ethnicity table
    save_table(ethnicity_individual, table_individual)


def ethnicity_record_selection(
    ethnicity_multisource: DataFrame,
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    data_source: List[str] = None,
    ethnicity_18_null_codes: List[str] = ['', 'X', 'Z', '99'],
    priority_index: Dict[str, int] = {'gdppr_snomed': 1, 'gdppr': 2, 'hes_apc': 3, 'hes_op': 4, 'hes_ae': 4},
) -> DataFrame:
    """
    Selects a single record for each individual from the multisource ethnicity DataFrame based on specified criteria.

    Args:
        ethnicity_multisource (DataFrame): DataFrame containing ethnicity data from multiple sources.
        min_record_date (str, optional): Minimum record date to consider. Defaults to '1900-01-01'.
        max_record_date (str, optional): Maximum record date to consider. Defaults to 'current_date()'.
        data_source (List[str], optional): List of allowed data sources to consider when selecting ethnicity records. 
            If specified, only records from the specified data sources will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        ethnicity_18_null_codes (List[str], optional): List of indeterminate ethnicity_18 codes that will be removed from selection.     
        priority_index (Dict[str, int], optional): Priority mapping for data sources; lower indices are prioritised.
            Defaults to {'gdppr_snomed': 1, 'gdppr': 2, 'hes_apc': 3, 'hes_op': 4, 'hes_ae': 4}.

    Returns:
        DataFrame: DataFrame containing the selected ethnicity records for each individual.
    """

    # Validate data_source argument
    if data_source is not None:
        assert isinstance(data_source, list), "data_source must be a list."
        assert data_source is None or data_source, "data_source cannot be an empty list."
        allowed_sources = {'gdppr_snomed', 'gdppr', 'hes_apc', 'hes_op', 'hes_ae'}
        invalid_sources = [str(source) for source in data_source if source not in allowed_sources or not isinstance(source, str)]
        assert not invalid_sources, f"Invalid data sources: {invalid_sources}. Allowed sources are: {allowed_sources}."

    # Filter out anomalous records
    ethnicity_multisource = (
        ethnicity_multisource
        .filter(
            (f.col('person_id')isNotNull())
            & (f.col('record_date').isNotNull())
            & (f.col('ethnicity_raw_code').isNotNull())
            & (f.col('ethnicity_18_code').isNotNull())
            & (~f.col('ethnicity_18_code').isin(ethnicity_18_null_codes))
        )
    )

    # Apply date restrictions
    if min_record_date is not None:
        ethnicity_multisource = (
            ethnicity_multisource
            .withColumn('min_record_date', f.expr(parse_date_instruction(min_record_date)))
            .filter('(record_date >= min_record_date)')
        )
    
    if max_record_date is not None:
        ethnicity_multisource = (
            ethnicity_multisource
            .withColumn('max_record_date', f.expr(parse_date_instruction(max_record_date)))
            .filter('(record_date <= max_record_date)')
        )

    # Apply data source restrictions
    if data_source is not None:
        ethnicity_multisource = (
            ethnicity_multisource
            .filter(f.col('data_source').isin(data_source))
        )

    # Map source priority
    ethnicity_multisource = (
        ethnicity_multisource
        .transform(map_column_values, map_dict = priority_index, column = 'data_source', new_column = 'source_priority')
    )

    # Select rows of 1st dense rank for each individual based on source priority and recency rules
    ethnicity_ties = (
        ethnicity_multisource
        .transform(
            first_dense_rank, n = 1,
            partition_by = ['person_id'],
            order_by = [f.col('source_priority').asc_nulls_last(), f.col('record_date').desc()]
        )
    )

    # Specify window function to collect ties 
    _win_collect_ties_5 = (
        Window
        .partitionBy('person_id')
        .orderBy('data_source', 'ethnicity_5_group')
    )

    # Create tie flag and collect ties in arrays
    ethnicity_ties = (
        ethnicity_ties
        .withColumn(
            'ethnicity_5_distinct_group',
            f.collect_set(f.col('ethnicity_5_group')).over(_win_collect_ties_5)
        )
        .withColumn(
            'ethnicity_5_tie_flag',
            f.when(f.size(f.col('ethnicity_5_distinct_group')) > f.lit(1), f.lit(1))
        )
        .withColumn(
            'ethnicity_5_tie_group',
            f.when(f.col('ethnicity_5_tie_flag') == f.lit(1), f.collect_list(f.col('ethnicity_5_group')).over(_win_collect_ties_5))
        )
        .withColumn(
            'ethnicity_5_tie_data_source',
            f.when(f.col('ethnicity_5_tie_flag') == f.lit(1), f.collect_list(f.col('data_source')).over(_win_collect_ties_5))
        )
    )

    # Specify window function to collect ties
    _win_collect_ties_18 = (
        Window
        .partitionBy('person_id')
        .orderBy('data_source', 'ethnicity_18_code', 'ethnicity_18_group')
    )

    # Create tie flag and collect ties in arrays
    ethnicity_ties = (
        ethnicity_ties
        .withColumn(
            'ethnicity_18_distinct_group',
            f.collect_set(f.col('ethnicity_18_group')).over(_win_collect_ties_18)
        )
        .withColumn(
            'ethnicity_18_tie_flag',
            f.when(f.size(f.col('ethnicity_18_distinct_group')) > f.lit(1), f.lit(1))
        )
        .withColumn(
            'ethnicity_18_tie_group',
            f.when(f.col('ethnicity_18_tie_flag') == f.lit(1), f.collect_list(f.col('ethnicity_18_group')).over(_win_collect_ties_18))
        )
        .withColumn(
            'ethnicity_18_tie_data_source',
            f.when(f.col('ethnicity_18_tie_flag') == f.lit(1), f.collect_list(f.col('data_source')).over(_win_collect_ties_18))
        )
    )

    # Randomly select record to break tie
    ethnicity_individual = (
        ethnicity_ties
        .transform(
            first_row, n = 1,
            partition_by = ['person_id'], order_by = [f.rand(seed = 124910)]
        )
    )

    # Select columns
    ethnicity_individual = (
        ethnicity_individual
        .select(
            'person_id', 'ethnicity_raw_code', 'ethnicity_raw_description', 'ethnicity_18_code', 'ethnicity_18_group',
            'ethnicity_5_group', 'record_date', 'data_source',
            'ethnicity_5_tie_flag', 'ethnicity_5_tie_group', 'ethnicity_5_tie_data_source',
            'ethnicity_18_tie_flag', 'ethnicity_18_tie_group', 'ethnicity_18_tie_data_source'
        )
    )

    return ethnicity_individual