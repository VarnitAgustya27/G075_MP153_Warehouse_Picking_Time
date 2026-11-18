import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

def train_baseline():
    # 1. Load cleaned data
    df = pd.read_csv('data/cleaned_train.csv')
    
    # 2. Preprocess
    # Drop identifiers, dates, and 'pick_time_from_scanner' (data leakage)
    cols_to_drop = ['record_id', 'date', 'pick_time_from_scanner']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # One-hot encode categoricals 
    df = pd.get_dummies(df, columns=['zone_type', 'shift'], drop_first=True)
    
    # Separate features and target
    X = df.drop(columns=['pick_seconds'])
    y = df['pick_seconds']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow Local SQLite Registry
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Warehouse_Picking_Time_G075")

    with mlflow.start_run() as run:
        # 4. Train Model
        params = {"n_estimators": 100, "max_depth": 7, "random_state": 42}
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        # 5. Evaluate
        preds = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"Model trained! RMSE: {rmse:.2f}, R2: {r2:.2f}")

        # 6. Log parameters, metrics, and register model
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # Keeping artifact_path as it acts as the folder name inside MLflow runs
            registered_model_name="PickingTime_RF_Model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print(f"Model successfully registered in MLflow. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_baseline()