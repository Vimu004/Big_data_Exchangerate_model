import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import snowflake.connector
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def load_prediction_data():
    conn = snowflake.connector.connect(
        user="###",
        password="###",
        account="###",
        warehouse="COMPUTE_WH",
        database="EXCHANGE_RATES_SL",
        schema="DW"
    )
    query = "SELECT * FROM PREDICTION_FEATURES"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def preprocess_data(df, target_col):
    df = df.dropna(subset=[target_col])
    df = df.dropna()  
    df["DATE"] = pd.to_datetime(df["DATE"])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    dates = X["DATE"]
    X = X.drop(columns=["DATE"])  

    return train_test_split(X, y, dates, test_size=0.2, random_state=42)

def train_and_evaluate_model(X_train, X_test, y_train, y_test, dates_test, target_col):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\n📈 {target_col} Prediction Metrics:")
    print(f"R² Score : {r2_score(y_test, preds):.4f}")
    print(f"MAE      : {mean_absolute_error(y_test, preds):.4f}")
    print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, preds)):.4f}")

    df_plot = pd.DataFrame({
        "DATE": dates_test,
        "Actual": y_test.values,
        "Predicted": preds
    }).sort_values("DATE")

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df_plot, x="DATE", y="Actual", label="Actual", color="blue")
    sns.lineplot(data=df_plot, x="DATE", y="Predicted", label="Predicted", color="orange")
    plt.title(f"{target_col} Exchange Rate Prediction")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    target_col = "USD"
    df = load_prediction_data()
    X_train, X_test, y_train, y_test, dates_train, dates_test = preprocess_data(df, target_col)
    model = train_and_evaluate_model(X_train, X_test, y_train, y_test, dates_test, target_col)
