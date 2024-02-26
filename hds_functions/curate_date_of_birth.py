from pyspark.sql import functions as f

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

