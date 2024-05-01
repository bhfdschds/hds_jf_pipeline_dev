from pyspark.sql import functions as f
from pyspark.sql import DataFrame
from functools import reduce
from typing import List, Dict
from .table_management import load_table
from .table_management import save_table
from .date_functions import parse_date_instruction
from .data_aggregation import first_row

def create_date_of_birth_multisource(table_multisource: str = 'date_of_birth_multisource', extraction_methods: List[str] = None) -> None:
    """
    Create a consolidated DataFrame containing date of birth data from multiple sources and save it to a table.

    Args:
        table_multisource (str, optional): The name of the table to save the consolidated data. Defaults to 'date_of_birth_multisource'.
        extraction_methods (List[str], optional): List of methods for extracting date of birth data. Defaults to None.

    Returns:
        None
    """
    if extraction_methods is None:
        extraction_methods = ['gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap']

    # Extract date of birth data from multiple sources
    date_of_birth_from_sources = [extract_date_of_birth(method) for method in extraction_methods]
    date_of_birth_multisource = reduce(DataFrame.unionByName, date_of_birth_from_sources)

    # Save the consolidated data to a table
    save_table(date_of_birth_multisource, table_multisource)


def extract_date_of_birth(data_source: str) -> DataFrame:
    """
    Extract date of birth data based on the specified data source.

    Args:
        data_source (str): The data source to extract date of birth data from.
            Allowed values are: 'gdppr', 'hes_apc', 'hes_op', 'hes_ae', 'ssnap'.

    Returns:
        DataFrame: DataFrame containing the extracted date of birth data from the selected source.
            The DataFrame includes columns for person ID, record date, date of birth and data source.

    """
    if data_source == 'gdppr':
        return gdppr_date_of_birth(load_table('gdppr', method='gdppr'))
    elif data_source == 'hes_apc':
        return hes_apc_date_of_birth(load_table('hes_apc', method='hes_apc'))
    elif data_source == 'hes_op':
        return hes_op_date_of_birth(load_table('hes_op', method='hes_op'))
    elif data_source == 'hes_ae':
        return hes_ae_date_of_birth(load_table('hes_ae', method='hes_ae'))
    elif data_source == 'ssnap':
        return ssnap_date_of_birth(load_table('ssnap', method='ssnap'))
    else:
        raise ValueError(f"Invalid data source: {data_source}")


def gdppr_date_of_birth(gdppr: DataFrame) -> DataFrame:
    """
    Process the date of birth data from the GDPPR table, ensuring distinct records.

    Args:
        gdppr (DataFrame): DataFrame containing the GDPPR table data.

    Returns:
        DataFrame: Processed DataFrame with metadata added.
    """

    date_of_birth_gdppr = (
        gdppr
        .select(
            'person_id',
            f.col('reporting_period_end_date').alias('record_date'),
            f.to_date('year_month_of_birth', 'yyyy-MM').alias('date_of_birth')
        )
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
        .distinct()
        .withColumn(
            'date_of_birth',
            f.date_sub(f.col('record_date'), f.round((f.col('age_on_arrival') + 0.5)*365.25).cast('integer'))
        )
        .drop('age_on_arrival')
        .withColumn('data_source', f.lit('ssnap'))
    )

    return date_of_birth_ssnap


def create_date_of_birth_individual(
    table_multisource: str = 'date_of_birth_multisource',
    table_individual: str = 'date_of_birth_individual',
    min_record_date: str = '1900-01-01',
    max_record_date: str = 'current_date()', 
    min_date_of_birth: str = '1880-01-01', 
    max_date_of_birth: str = 'current_date()',
    data_source: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1},
):
    """
    Wrapper function to create and save a table containing selected date of birth records for each individual.

    Args:
        table_multisource (str): Name of the multisource date of birth table.
        table_individual (str): Name of the individual date of birth table to be created.
        min_record_date (str, optional): Minimum record date to consider. Defaults to '1900-01-01'.
        max_record_date (str, optional): Maximum record date to consider. Defaults to 'current_date()'.
        min_date_of_birth (str, optional): Minimum date of birth to consider. Defaults to '1880-01-01'.
        max_date_of_birth (str, optional): Maximum date of birth to consider. Defaults to 'current_date()'.
        data_source (List[str], optional): List of allowed data sources to consider when selecting date of birth records. 
            If specified, only records from the specified data sources will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): Priority index mapping data sources to priority levels.
            Sources not specified in the mapping will have a default priority level of 0.
            Defaults to {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1}.
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
        data_source=data_source,
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
    data_source: List[str] = None,
    priority_index: Dict[str, int] = {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1},
) -> DataFrame:
    """
    Selects a single record for each individual from the multisource date of birth DataFrame based on specified criteria.

    Args:
        date_of_birth_multisource (DataFrame): DataFrame containing date of birth data from multiple sources.
        min_record_date (str, optional): Minimum record date to consider. Defaults to '1900-01-01'.
        max_record_date (str, optional): Maximum record date to consider. Defaults to 'current_date()'.
        min_date_of_birth (str, optional): Minimum date of birth to consider. Defaults to '1880-01-01'.
        max_date_of_birth (str, optional): Maximum date of birth to consider. Defaults to 'current_date()'.
        data_source (List[str], optional): List of allowed data sources to consider when selecting date of birth records. 
            If specified, only records from the specified data sources will be included in the selection process. 
            If None, records from all available data sources will be considered. 
            Defaults to None.
        priority_index (Dict[str, int], optional): Priority index mapping data sources to priority levels.
            Sources not specified in the mapping will have a default priority level of 0.
            Defaults to {'gdppr': 3, 'hes_apc': 2, 'hes_op': 1, 'hes_ae': 1}.

    Returns:
        DataFrame: DataFrame containing the selected date of birth records for each individual.
    """

    # Filter out anomalous records
    date_of_birth_multisource = (
        date_of_birth_multisource
        .filter(
            (f.expr('date_of_birth IS NOT NULL'))
            & (f.expr('record_date IS NOT NULL'))
            & (f.expr('record_date >= date_of_birth'))
        )
    )

    # Apply date restrictions
    if min_record_date is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('min_record_date', f.expr(parse_date_instruction(min_record_date)))
            .filter(f.expr('(record_date >= min_record_date)'))
        )
    
    if max_record_date is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('max_record_date', f.expr(parse_date_instruction(max_record_date)))
            .filter(f.expr('(record_date <= max_record_date)'))
        )

    if min_date_of_birth is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('min_date_of_birth', f.expr(parse_date_instruction(min_date_of_birth)))
            .filter(f.expr('(date_of_birth >= min_date_of_birth)'))
        )

    if max_date_of_birth is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .withColumn('max_date_of_birth', f.expr(parse_date_instruction(max_date_of_birth)))
            .filter(f.expr('(date_of_birth <= max_date_of_birth)'))
        )

    # Apply data source restrictions
    if data_source is not None:
        date_of_birth_multisource = (
            date_of_birth_multisource
            .filter(f.col('data_source').isin(data_source))
        )

    # Map source priority
    date_of_birth_multisource = (
        date_of_birth_multisource
        .transform(map_column_values, map_dict = priority_index, column = 'data_source', new_column = 'source_priority')
        .fillna({'source_priority': 0})
    )

    # Select record for each individual based on source priority and recency rules
    date_of_birth_individual = (
        date_of_birth_multisource
        .transform(
            first_row,
            partition_by = ['person_id'],
            order_by = [f.col('source_priority').desc(), f.col('record_date').desc(), 'data_source', 'date_of_birth']
        )
    )

    # Select columns
    date_of_birth_individual = (
        date_of_birth_individual
        .select('person_id', 'date_of_birth', 'record_date', 'data_source')
    )

    return date_of_birth_individual