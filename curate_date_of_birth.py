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
