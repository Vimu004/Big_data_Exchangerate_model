from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ExchangeRatePrediction") \
    .config("spark.jars", "/path/to/snowflake-jdbc.jar") \
    .config("spark.jars.packages", "net.snowflake:snowflake-jdbc:REPLACE_VERSION") \
    .getOrCreate()

sfOptions = {
    "sfURL" : "https://dlbgnxt-iu64182.snowflakecomputing.com",
    "sfUser" : "VIMUTHU04",
    "sfPassword" : "Vimuthu20042007",
    "sfDatabase" : "EXCHANGE_RATES_SL",
    "sfSchema" : "DW",
    "sfWarehouse" : "COMPUTE_WH",
    "sfRole" : "ACCOUNTADMIN",  
}

df = spark.read \
    .format("snowflake") \
    .options(**sfOptions) \
    .option("dbtable", "PREDICTION_FEATURES") \
    .load()

features = [
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
targets = ["USD", "EUR", "GBP", "INR", "RUB", "CNY", "AUD"]

from pyspark.sql.functions import year
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

df = df.withColumn("YEAR", year("DATE"))
features.append("YEAR")

assembler = VectorAssembler(inputCols=features, outputCol="features")

from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

models = {}
predictions = {}

for target in targets:
    print(f"Training model for {target}...")

    # Assemble training data
    assembled_data = assembler.transform(df).select("features", target)
    
    # Split train-test
    train_data, test_data = assembled_data.randomSplit([0.8, 0.2], seed=42)

    # Train model
    gbt = GBTRegressor(featuresCol="features", labelCol=target, maxDepth=5, maxIter=100)
    model = gbt.fit(train_data)
    models[target] = model

    # Predict & evaluate
    pred = model.transform(test_data)
    predictions[target] = pred
    evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(pred)
    print(f"{target} RMSE: {rmse:.4f}")

    from pyspark.sql import Row

# Example input — replace these with actual 2025 projections
input_2025 = Row(
    SRI_LANKA_GDP_USD_BILLIONS_USD=100,
    ANNUAL_CHANGE_PCR=5.5,
    PER_CAPITA_USD=4000,
    MERCHANDISE_EXPORTS_MILLIONS_USD=11000,
    LABOUR_MIGRATION_SL=250000,
    MERCHANDISE_IMPORTS_MILLIONS_USD=19000,
    INFLATION_RATE_PCR=6.2,
    GDP_PER_CAPITA_USD=4100,
    GDP_GROWTH_PCT=3.8,
    YEAR=2025
)

input_df = spark.createDataFrame([input_2025])
input_vector = assembler.transform(input_df)

# Predict each currency
print("📈 Predicted Exchange Rates for 2025:")
for target in targets:
    prediction = models[target].transform(input_vector)
    predicted_value = prediction.select("prediction").collect()[0][0]
    print(f"{target}: {predicted_value:.4f}")




