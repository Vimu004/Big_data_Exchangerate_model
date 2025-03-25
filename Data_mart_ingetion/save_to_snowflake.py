from pyspark.sql import DataFrame

def save_dataframe_to_snowflake(
    df: DataFrame,
    table_name: str,
    sf_user: str,
    sf_password: str,
    sf_url: str,
    sf_database: str,
    sf_schema: str,
    sf_warehouse: str,
    sf_role: str = "ACCOUNTADMIN",
    mode: str = "overwrite"  # or "append"
):
    """
    Save a Spark DataFrame to a Snowflake table.

    Args:
        df (DataFrame): The Spark DataFrame to save.
        table_name (str): Name of the Snowflake table.
        sf_user (str): Snowflake username.
        sf_password (str): Snowflake password.
        sf_url (str): Snowflake account URL (e.g., 'dlbgnxt-iu64182.snowflakecomputing.com').
        sf_database (str): Snowflake database name.
        sf_schema (str): Snowflake schema name.
        sf_warehouse (str): Snowflake warehouse name.
        sf_role (str): Snowflake role (default: ACCOUNTADMIN).
        mode (str): Save mode ('overwrite', 'append').
    """

    sf_options = {
        "sfURL": sf_url,
        "sfUser": sf_user,
        "sfPassword": sf_password,
        "sfDatabase": sf_database,
        "sfSchema": sf_schema,
        "sfWarehouse": sf_warehouse,
        "sfRole": sf_role
    }

    df.write \
        .format("snowflake") \
        .options(**sf_options) \
        .option("dbtable", table_name) \
        .mode(mode) \
        .save()

    print(f"✅ Data saved to Snowflake table: {table_name}")
