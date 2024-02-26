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
