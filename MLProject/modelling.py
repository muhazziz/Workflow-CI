import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def main():
    mlflow.set_experiment("Bank_Customer_Churn_CI")
    mlflow.sklearn.autolog()

    print("Memuat dataset...")
    train_df = pd.read_csv("churn_preprocessing/train.csv")
    test_df = pd.read_csv("churn_preprocessing/test.csv")

    X_train = train_df.drop('Exited', axis=1)
    y_train = train_df['Exited']

    with mlflow.start_run() as run:
        print("Melatih model Random Forest...")
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(rf, "model")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()