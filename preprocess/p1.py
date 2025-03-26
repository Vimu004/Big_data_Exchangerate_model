#Insight about the data from an AI 

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, unix_timestamp, col, concat_ws, lit
from openai import AzureOpenAI
import pandas as pd

AZURE_OPENAI_API_KEY = "###"
AZURE_OPENAI_DEPLOYMENT_ID = "###"
AZURE_ENDPOINT = "###"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2023-07-01-preview",
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_ID
)

def get_spark_session():
    return SparkSession.builder \
        .appName("FeatureSuggestionGPT") \
        .config("spark.jars.packages", "net.snowflake:snowflake-jdbc:3.13.30,net.snowflake:spark-snowflake_2.12:2.9.1-spark_3.1") \
        .getOrCreate()

def load_table_from_snowflake(spark, table_name, sf_user, sf_password):
    sfOptions = {
        "sfURL": "https://dlbgnxt-iu64182.snowflakecomputing.com",
        "sfUser": sf_user,
        "sfPassword": sf_password,
        "sfDatabase": "EXCHANGE_RATES_SL",
        "sfSchema": "DBO",
        "sfWarehouse": "COMPUTE_WH",
        "sfRole": "ACCOUNTADMIN"
    }

    return spark.read \
        .format("snowflake") \
        .options(**sfOptions) \
        .option("dbtable", table_name) \
        .load()

def ask_gpt_feature_strategy(exchange_df: pd.DataFrame, econ_df: pd.DataFrame, tourism_df: pd.DataFrame):
    prompt = f"""
You are a data scientist helping to predict **exchange rates**.

You have access to 3 separate datasets pulled from Snowflake:

---

📘 **1. EXCHANGE_RATES_SL** (daily):
{exchange_df.head(10).to_markdown()}

---

📗 **2. SL_ECONOMIC_DATA** (yearly or monthly):
{econ_df.head(10).to_markdown()}

---

📙 **3. TOURSIM_SL** (monthly):
{tourism_df.head(10).to_markdown()}

---

**Your tasks:**
1. For each table, identify the most important features that may help in predicting exchange rates.
2. How can we align and merge these tables to create a **daily-level super dataset**?
3. Suggest a robust preprocessing and feature engineering pipeline.
4. Mention the best modeling approaches (e.g., time-series, ML) suitable for this goal. we were asked to use apache spark or is their anything else to do it

Please respond in detailed bullet points.
"""

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT_ID,
        messages=[
            {"role": "system", "content": "You are a senior data scientist."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    spark = get_spark_session()

    sf_user = "VIMUTHU04"
    sf_password = "Vimuthu20042007"

    print("🔄 Loading tables from Snowflake...")
    exchange_rates_df = load_table_from_snowflake(spark, "EXCHANGE_RATES_SL", sf_user, sf_password)
    economic_df = load_table_from_snowflake(spark, "SL_ECONOMIC_DATA", sf_user, sf_password)
    tourism_df = load_table_from_snowflake(spark, "TOURSIM_SL", sf_user, sf_password)

    print("✅ Tables Loaded.")

    exchange_sample = exchange_rates_df.limit(10).toPandas()
    econ_sample = economic_df.limit(10).toPandas()
    tourism_sample = tourism_df.limit(10).toPandas()

    print("🤖 Asking GPT for feature strategy...")
    gpt_response = ask_gpt_feature_strategy(exchange_sample, econ_sample, tourism_sample)

    with open("gpt_feature_strategy.txt", "w", encoding="utf-8") as f:
        f.write(gpt_response)

    print("\n🧠 GPT's Response saved to 'gpt_feature_strategy.txt'.")

    print("\n🧠 GPT's Response:\n")
    print(gpt_response)