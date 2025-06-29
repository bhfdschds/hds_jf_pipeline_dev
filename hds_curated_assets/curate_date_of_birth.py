from pyspark.sql import functions as f, DataFrame, Window
from functools import reduce
from typing import List, Dict
from .table_management import load_table, save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row, first_dense_rank
from .data_wrangling import map_column_values

def create_date_of_birth_multisource(table_multisource: str = 'date_of_birth_multisource', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing date of birth data from multiple sources and save it to a table.

    Args:
        table_multisource (str, optional): The table key of the table to save the consolidated data. Defaults to 'date_of_birth_multisource'.
        extraction_methods (List[str], optional): List of methods for extracting date of birth data. Defaults to None.

    Returns:
        None
    """
    if extraction_methods is None:
        extraction_methods = ['gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap', 'vaccine_status']

    # Extract date of birth data from multiple sources
    date_of_birth_from_sources = [extract_date_of_birth(method) for method in extraction_methods]
    date_of_birth_multisource = reduce(DataFrame.unionByName, date_of_birth_from_sources)

    # Save the consolidated data to a table
    save_table(date_of_birth_multisource, table_multisource)


def extract_date_of_birth(extract_method: str) -> DataFrame:
    """
    Extract date of birth data based on the specified data source.

        Args:
        extract_method (str): The method to extract date of birth data.
            Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap', 'vaccine_status'.

    Returns:
        DataFrame: DataFrame containing the extracted date of birth data from the selected source.
            The DataFrame includes columns for archive version, person ID, record date, date of birth and data source.

    """

    extraction_methods = {
        'gdppr': {
            'extraction_function': gdppr_date_of_birth,
            'data_source': 'gdppr_demographics',
            'load_method': None
        },
        'hes_apc': {
            'extraction_function': hes_apc_date_of_birth,
            'data_source': 'hes_apc',
            'load_method': 'hes_apc'
        },
        'hes_op': {
            'extraction_function': hes_op_date_of_birth,
            'data_source': 'hes_op',
            'load_method': 'hes_op'
        },
        'hes_ae': {
            'extraction_function': hes_ae_date_of_birth,
            'data_source': 'hes_ae',
            'load_method': 'hes_ae'
        },
        'ssnap': {
            'extraction_function': ssnap_date_of_birth,
            'data_source': 'ssnap',
            'load_method': 'ssnap'
        },
        'vaccine_status': {
            'extraction_function': vaccine_status_date_of_birth,
            'data_source': 'vaccine_status',
            'load_method': 'vaccine_status'
        }
    }

    if extract_method not in extraction_methods:
        raise ValueError(f"Invalid extract_method: {extract_method}. Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap', 'vaccine_status'.")

    return extraction_methods[extract_method]['extraction_function'](
        load_table(
            table=extraction_methods[extract_method]['data_source'],
            method=extraction_methods[extract_method]['load_method']
        )
    )



def gdppr_date_of_birth(gdppr_demographics: DataFrame) -> DataFrame:
    """
    Process the date of birth data from the GDPPR demographics table, ensuring distinct records.

    Args:
        gdppr_demographics (DataFrame): DataFrame containing GDPPR demographics data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    date_of_birth_gdppr = (
        gdppr_demographics
        .select(
            'person_id',
            f.col('reporting_period_end_date').alias('record_date'),
            f.to_date('year_month_of_birth', 'yyyy-MM').alias('date_of_birth')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (date_of_birth IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('gdppr'))
    )

    return date_of_birth_gdppr


def hes_apc_date_of_birth(hes_apc: DataFrame) -> DataFrame:
    """
    Process the date of birth data from the HES-APC (Admitted Patient Care) table, ensuring distinct records.

    Args:
        hes_apc (DataFrame): DataFrame containing the HES-APC table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    date_of_birth_hes_apc = (
        hes_apc
        .select(
            'person_id',
            f.col('epistart').alias('record_date'),
            f.to_date('mydob', 'MMyyyy').alias('date_of_birth')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (date_of_birth IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('hes_apc'))
    )

    return date_of_birth_hes_apc


def hes_op_date_of_birth(hes_op: DataFrame) -> DataFrame:
    """
    Process the date of birth data from the HES-OP (Outpatients) table, ensuring distinct records.

    The 'apptage_calc' column in HES-OP represents the age of the individual at the time of appointment.
    For those aged 1 year or more, these are in completed years, while those less than 1 year of age have
    decimalised values. When converting age to date of birth, an upward adjustment of 0.5 years is applied
    to those aged 1 year or more to approximate fractional ages.

    Args:
        hes_op (DataFrame): DataFrame containing the HES-OP table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    date_of_birth_hes_op = (
        hes_op
        .select(
            'person_id',
            f.col('apptdate').alias('record_date'),
            f.col('apptage_calc').alias('age_at_appointment')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (age_at_appointment IS NOT NULL)")
        .distinct()
        .withColumn(
            'date_of_birth',
            f.when(
                f.col('age_at_appointment') < f.lit(1),
                f.date_sub(f.col('record_date'), f.round(f.col('age_at_appointment') * 365.25).cast('integer'))
            )
            .when(
                f.col('age_at_appointment') >= f.lit(1),
                f.date_sub(
                    f.col('record_date'),
                    f.round((f.col('age_at_appointment') + 0.5) * 365.25).cast('integer')
                )
            )
        )
        .drop('age_at_appointment')
        .withColumn('data_source', f.lit('hes_op'))
    )

    return date_of_birth_hes_op


def hes_ae_date_of_birth(hes_ae: DataFrame) -> DataFrame:
    """
    Process the date of birth data from the HES-A&E (Accident and Emergency) table, ensuring distinct records.

    The 'arrivalage_calc' column in HES-AE represents the age of the individual at the time of A&E arrival.
    For those aged 1 year or more, these are in completed years, while those less than 1 year of age have
    decimalised values. When converting age to date of birth, an upward adjustment of 0.5 years is applied
    to those aged 1 year or more to approximate fractional ages.

    Args:
        hes_ae (DataFrame): DataFrame containing the HES-AE table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """
    date_of_birth_hes_ae = (
        hes_ae
        .select(
            'person_id',
            f.col('arrivaldate').alias('record_date'),
            f.col('arrivalage_calc').alias('age_at_arrival')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (age_at_arrival IS NOT NULL)")
        .distinct()
        .withColumn(
            'date_of_birth',
            f.when(
                f.col('age_at_arrival') < f.lit(1),
                f.date_sub(f.col('record_date'), f.round(f.col('age_at_arrival') * 365.25).cast('integer'))
            )
            .when(
                f.col('age_at_arrival') >= f.lit(1),
                f.date_sub(
                    f.col('record_date'),
                    f.round((f.col('age_at_arrival') + 0.5) * 365.25).cast('integer')
                )
            )
        )
        .drop('age_at_arrival')
        .withColumn('data_source', f.lit('hes_ae'))
    )

    return date_of_birth_hes_ae


def ssnap_date_of_birth(ssnap: DataFrame) -> DataFrame:
    """
    Process the date of birth data from the SSNAP (Sentinel Stroke National Audit Programme) table, ensuring distinct
    records.

    The 's1ageonarrival' column in SSNAP records the age of the individual at the time of arrival to the hospital.
    While this column is an integer, it represents completed years. Therefore, an upward adjustment of 0.5 years
    is applied to approximate fractional ages.

    Args:
        ssnap (DataFrame): DataFrame containing the SSNAP table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    date_of_birth_ssnap = (
        ssnap
        .select(
            'person_id',
            f.to_date('s1firstarrivaldatetime').alias('record_date'),
            f.col('s1ageonarrival').alias('age_on_arrival')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (age_on_arrival IS NOT NULL)")
        .distinct()
        .withColumn(
            'date_of_birth',
            f.date_sub(f.col('record_date'), f.round((f.col('age_on_arrival') + 0.5)*365.25).cast('integer'))
        )
        .drop('age_on_arrival')
        .withColumn('data_source', f.lit('ssnap'))
    )

    return date_of_birth_ssnap


def vaccine_status_date_of_birth(vaccine_status: DataFrame) -> DataFrame:
    """
    Process the date of birth data from the COVID-19 vaccination status table, ensuring distinct
    records.

    Args:
        vaccine_status (DataFrame): DataFrame containing the COVID-19 vaccination status table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    date_of_birth_vaccine_status = (
        vaccine_status
        .select(
            'person_id',
            f.col('recorded_date').alias('record_date'),
            f.when(
                f.col('mydob').rlike('\d{6}'),
                f.to_date(f.col('mydob'), 'MMyyyy')
            ).alias('date_of_birth')
        )
        .filter("(person_id IS NOT NULL) AND (record_date IS NOT NULL) AND (date_of_birth IS NOT NULL)")
        .distinct()
        .withColumn('data_source', f.lit('vaccine_status'))
    )

    return date_of_birth_vaccine_status


def create_date_of_birth_individual(
    table_multisource: str = 'date_of_birth_multisource',
    table_individual: str = 'date_of_birth_individual',
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    min_date_of_birth: str = '1880-01-01', 
    max_date_of_birth: str = 'current_date()',
    filter_data_sources: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3},
) -> None:
    """
    Wrapper function to create and save a table containing selected date of birth records for each individual.

    Args:
        table_multisource (str): Table key of the multisource date of birth table.
        table_individual (str): Table key of the individual date of birth table to be created.
        min_record_date (str, optional): Expression for minimum record date. Defaults to '1900-01-01'.
        max_record_date (str, optional): Expression for maximum record date. Defaults to 'current_date()'.
        min_date_of_birth (str, optional): Expression for minimum date of birth. Defaults to '1880-01-01'.
        max_date_of_birth (str, optional): Expression for maximum date of birth. Defaults to 'current_date()'.
        filter_data_sources (List[str], optional): List of data sources to include when selecting date of birth records. 
            If specified, only records in the list will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): A dictionary mapping data sources to their priority levels.
            Lower integer values indicate higher priority. Data sources not included in the dictionary 
            are deprioritised and assigned a null priority index. Defaults to the following priority mapping:
            {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3}.

    Note:
        The parameters min_record_date, max_record_date, min_date_of_birth, and max_date_of_birth are passed to
        parse_date_instruction() before being processed by Pyspark's expr() function.
    """

    # Load multisource date of birth table
    date_of_birth_multisource = load_table(table_multisource)

    # Select individual date of birth records
    date_of_birth_individual = date_of_birth_record_selection(
        date_of_birth_multisource,
        min_record_date=min_record_date,
        max_record_date=max_record_date,
        min_date_of_birth=min_date_of_birth,
        max_date_of_birth=max_date_of_birth,
        filter_data_sources=filter_data_sources,
        priority_index=priority_index
    )

    # Save individual date of birth table
    save_table(date_of_birth_individual, table_individual)


def date_of_birth_record_selection(
    date_of_birth_multisource: DataFrame,
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    min_date_of_birth: str = '1880-01-01', 
    max_date_of_birth: str = 'current_date()',
    filter_data_sources: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3},
) -> DataFrame:
    """
    Selects a single record for each individual from the multisource date of birth DataFrame based on specified criteria.
    Record selection is based on data sources with the lowest priority index followed by the latest record date. If ties exist,
    the tie values and data sources are retained in seperate columns while a random record is selected to break the tie.

    Args:
        date_of_birth_multisource (DataFrame): DataFrame containing date of birth data from multiple sources.
        min_record_date (str, optional): Expression for minimum record date. Defaults to '1900-01-01'.
        max_record_date (str, optional): Expression for maximum record date. Defaults to 'current_date()'.
        min_date_of_birth (str, optional): Expression for minimum date of birth. Defaults to '1880-01-01'.
        max_date_of_birth (str, optional): Expression for maximum date of birth. Defaults to 'current_date()'.
        filter_data_sources (List[str], optional): List of data sources to include when selecting date of birth records. 
            If specified, only records in the list will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): A dictionary mapping data sources to their priority levels.
            Lower integer values indicate higher priority. Data sources not included in the dictionary 
            are deprioritised and assigned a null priority index. Defaults to the following priority mapping:
            {'gdppr': 1, 'hes_apc': 2, 'hes_op': 3, 'hes_ae': 3}.

    Returns:
        DataFrame: DataFrame containing the selected date of birth records for each individual.

    Note:
        The parameters min_record_date, max_record_date, min_date_of_birth, and max_date_of_birth are passed to
        parse_date_instruction() before being processed by Pyspark's expr() function.
    """

    # Allowed data sources
    allowed_sources = {'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap', 'vaccine_status'}

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
    date_of_birth_multisource = (
        date_of_birth_multisource
        .filter(
            """
            (person_id IS NOT NULL) AND (date_of_birth IS NOT NULL)
            AND (record_date IS NOT NULL) AND (record_date >= date_of_birth)
            """
        )
    )

    # Apply date restrictions
    if min_record_date is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('min_record_date', f.expr(parse_date_instruction(min_record_date)))
            .filter('(record_date >= min_record_date)')
        )
    
    if max_record_date is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('max_record_date', f.expr(parse_date_instruction(max_record_date)))
            .filter('(record_date <= max_record_date)')
        )

    if min_date_of_birth is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('min_date_of_birth', f.expr(parse_date_instruction(min_date_of_birth)))
            .filter('(date_of_birth >= min_date_of_birth)')
        )

    if max_date_of_birth is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('max_date_of_birth', f.expr(parse_date_instruction(max_date_of_birth)))
            .filter('(date_of_birth <= max_date_of_birth)')
        )

    # Apply data source restrictions
    if filter_data_sources is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .filter(f.col('data_source').isin(filter_data_sources))
        )

    # Map source priority
    date_of_birth_multisource = (
        date_of_birth_multisource
        .transform(map_column_values, map_dict = priority_index, column = 'data_source', new_column = 'source_priority')
    )

    # Select rows of 1st dense rank for each individual based on source priority and recency rules
    date_of_birth_ties = (
        date_of_birth_multisource
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
        .orderBy('data_source', 'date_of_birth')
        .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
    )

    # Create tie flag and collect ties in arrays
    date_of_birth_ties = (
        date_of_birth_ties
        .withColumn(
            'date_of_birth_distinct_value',
            f.collect_set(f.col('date_of_birth')).over(_win_collect_ties)
        )
        .withColumn(
            'date_of_birth_tie_flag',
            f.when(f.size(f.col('date_of_birth_distinct_value')) > f.lit(1), f.lit(1))
        )
        .withColumn(
            'date_of_birth_tie_value',
            f.when(f.col('date_of_birth_tie_flag') == f.lit(1), f.collect_list(f.col('date_of_birth')).over(_win_collect_ties))
        )
        .withColumn(
            'date_of_birth_tie_data_source',
            f.when(f.col('date_of_birth_tie_flag') == f.lit(1), f.collect_list(f.col('data_source')).over(_win_collect_ties))
        )
    )

    # Randomly select record to break tie
    date_of_birth_individual = (
        date_of_birth_ties
        .transform(
            first_row, n = 1,
            partition_by = ['person_id'], order_by = [f.rand(seed = 124910)]
        )
    )

    # Select columns
    date_of_birth_individual = (
        date_of_birth_individual
        .select(
            'person_id', 'date_of_birth', 'record_date', 'data_source',
            'date_of_birth_tie_flag', 'date_of_birth_tie_value', 'date_of_birth_tie_data_source'
        )
    )

    return date_of_birth_individual