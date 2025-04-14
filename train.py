import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import optuna
import mlflow
from mlflow.client import MlflowClient
import pandas as pd
from data_preparation import create_dataset_for_train_val_test
from sklearn.ensemble import RandomForestClassifier
from mlflow.models import infer_signature
from eval import evaluate


def objective(trial, train_df, val_df,experiment_id):
    # Định nghĩa siêu tham số
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
    }

    model = RandomForestClassifier(**params, random_state=42)
    x_train = train_df.drop(columns=["Target"])
    y_train = train_df["Target"]
    x_val = val_df.drop(columns=["Target"])
    y_val = val_df["Target"]

    signature = infer_signature(x_train, y_train)
    

    with mlflow.start_run(run_name=f'rf-validation-{trial.number}',experiment_id=experiment_id, nested=True) as child_run:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        metrics = evaluate(y_true=y_val, y_pred=y_pred, prefix="validation")

        mlflow.log_params(params)  # Log chỉ các tham số cần thiết
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, f"model-trial-{trial.number}", signature=signature)

    return metrics["validation_accuracy"]

if __name__ == '__main__':
    # Tải dataset
    train_df, val_df, test_df = create_dataset_for_train_val_test()
    total_df = pd.concat([train_df, val_df, test_df], axis=0)

    x_train = train_df.drop(columns=["Target"])
    y_train = train_df["Target"]
    x_test = test_df.drop(columns=["Target"])
    y_test = test_df["Target"]
    signature = infer_signature(x_train, y_train)

    # Tạo hoặc lấy experiment
    experiment_name = "Pipeline_Training_With_Hyperparameter_Tuning"
    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location='artifacts',
            tags={"mlflow.note.content": "Lab01"}
        )
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Khởi tạo parent run
    with mlflow.start_run(run_name="train_model_and_tuning", experiment_id=experiment_id) as parent_run:
        # Log thông tin dataset
        dataset_info = {
            'Data_Source': 'https://www.binance.com/',
            'Data_Version': '1',
            'Data_Description': 'Dữ liệu lịch sử giá Bitcoin đã được biến đổi, với cột Target có giá trị 0 (nên bán) hoặc 1 (nên mua). Cột Target được tính dựa trên hiệu số giá đóng cửa của phiên tiếp theo trừ phiên trước đó',
            'Data_Columns': list(total_df.columns),
            'Data_Size': len(total_df),
            'Data_Collection_Date': f'{total_df.index.min()} to {total_df.index.max()}',
            'Data_Shape_with_Target': total_df.shape,
        }
        mlflow.log_params(dataset_info)
        mlflow.log_artifact('Data/CrawlBitCoin.csv')

        # Kiểm tra parent run
        if not mlflow.active_run():
            raise RuntimeError("Parent run not active!")

        # Tối ưu hóa với Optuna
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_df, val_df,experiment_id), n_trials=5)  # Dùng val_df

        # Log kết quả tốt nhất
        best_params = study.best_params
        mlflow.log_params(best_params)
        mlflow.log_metric("best_validation_accuracy", study.best_value)
        mlflow.log_param("best_trial_number", study.best_trial.number)

        # Huấn luyện và đánh giá mô hình tốt nhất trên test set
        best_model = RandomForestClassifier(**best_params, random_state=42)
        best_model.fit(x_train, y_train)
        y_pred = best_model.predict(x_test)
        metrics = evaluate(y_true=y_test, y_pred=y_pred, prefix="test")

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            best_model,
            artifact_path="checkpoints/best_model",
            signature=signature,
            registered_model_name="best_rf_model"
        )
        client = MlflowClient()
        model_version = client.get_latest_versions("best_rf_model", stages=["None"])[0]
        mlflow.log_param("model_version", model_version.version)
        print(f"Registered model 'best_rf_model' with version {model_version.version}")
