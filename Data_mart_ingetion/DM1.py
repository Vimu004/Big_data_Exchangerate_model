#creation of the data mart

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from save_to_snowflake import save_dataframe_to_snowflake

sf_options = {
    "sfURL": "https://dlbgnxt-iu64182.snowflakecomputing.com",
    "sfUser": "###",
    "sfPassword": "####",
    "sfDatabase": "EXCHANGE_RATES_SL",
    "sfSchema": "DBO",
    "sfWarehouse": "COMPUTE_WH",
    "sfRole": "ACCOUNTADMIN"
}

spark = SparkSession.builder \
    .appName("FinalPredictionFeatureTable") \
    .config("spark.jars.packages", "net.snowflake:snowflake-jdbc:3.13.30,net.snowflake:spark-snowflake_2.12:2.9.1-spark_3.1") \
    .getOrCreate()

def load_from_snowflake(table_name):
    return spark.read \
        .format("snowflake") \
        .options(**sf_options) \
        .option("dbtable", table_name) \
        .load()

print("📥 Loading data from Snowflake...")
exchange_df = load_from_snowflake("EXCHANGE_RATES_SL")
econ_df = load_from_snowflake("SL_ECONOMIC_DATA")
tourism_df = load_from_snowflake("TOURSIM_SL")

exchange_df = exchange_df.withColumn("DATE", col("DATE").cast("date"))
econ_df = econ_df.withColumn("DATE", col("DATE").cast("date"))
tourism_df = tourism_df.withColumn("DATE", col("DATE").cast("date"))

econ_selected_cols = [
    "DATE",
    "SRI_LANKA_GDP_USD_BILLIONS_USD",
    "ANNUAL_CHANGE_PCR",
    "PER_CAPITA_USD",
    "MERCHANDISE_EXPORTS_MILLIONS_USD",
    "LABOUR_MIGRATION_SL",
    "MERCHANDISE_IMPORTS_MILLIONS_USD",
    "INFLATION_RATE_PCR",
    "GDP_PER_CAPITA_USD",
    "GDP_GROWTH_PCT"
]

exchange_features = exchange_df.select("DATE", "USD", "EUR", "GBP", "INR", "RUB", "CNY", "AUD")
econ_features = econ_df.select(econ_selected_cols)
tourism_features = tourism_df.select("DATE", "TOURISM_SL")

final_df = exchange_features \
    .join(tourism_features, on="DATE", how="left") \
    .join(econ_features, on="DATE", how="left")

final_df = final_df.dropna(thresh=5)

print("Final Feature Table Preview:")
final_df.show(10020, truncate=False)  

save_dataframe_to_snowflake(
    df=final_df,
    table_name="PREDICTION_FEATURES",
    sf_user="###",
    sf_password="###",
    sf_url="dlbgnxt-iu64182.snowflakecomputing.com",
    sf_database="EXCHANGE_RATES_SL",
    sf_schema="DW",
    sf_warehouse="COMPUTE_WH",
    mode="overwrite"
)
