import argparse
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import boto3
import tarfile

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--bucket', type=str, default='predicciontips')
    parser.add_argument('--alpha', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Cargar los datos
    try:
        train_data = pd.read_csv(os.path.join(args.train, "train.csv"))
        test_data = pd.read_csv(os.path.join(args.test, "test.csv"))
        if 'tip' not in train_data.columns or 'tip' not in test_data.columns:
            raise KeyError("'tip' column is missing in the data")
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Preparar el preprocesador y entrenar el modelo
    try:
        numeric_features = ['total_bill', 'size']
        numeric_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ],
            remainder='passthrough'
        )

        X_train = train_data.drop('tip', axis=1)
        y_train = train_data['tip']
        X_test = test_data.drop('tip', axis=1)
        y_test = test_data['tip']

        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Error training model: {e}")
        raise
    
    # Guardar el modelo
    try:
        model_path = os.path.join(args.model_dir, "model.joblib")
        joblib.dump(pipeline, model_path)
        
        # Empaquetar el modelo en un archivo tar.gz
        with tarfile.open('model.tar.gz', 'w:gz') as tar:
            tar.add(model_path, arcname='model.joblib')
        
        # Subir el archivo tar.gz a S3
        s3_client = boto3.client('s3')
        bucket = args.bucket
        model_tar_path = 'model/model.tar.gz'
        s3_client.upload_file('model.tar.gz', bucket, model_tar_path)
        
        print(f'Model tar.gz uploaded to s3://{bucket}/{model_tar_path}')
    except Exception as e:
        print(f"Error saving or uploading model: {e}")
        raise
    
    # Evaluar el modelo y guardar métricas
    try:
        y_train_pred = pipeline.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)
        mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    
        y_test_pred = pipeline.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mape_test = mean_absolute_percentage_error(y_test, y_test_pred)
    
        os.makedirs(args.output_data_dir, exist_ok=True)
        metrics_path = os.path.join(args.output_data_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write("Training Metrics:\n")
            f.write(f"Mean Squared Error: {mse_train}\n")
            f.write(f"R^2: {r2_train}\n")
            f.write(f"Mean Absolute Percentage Error: {mape_train}\n")
            f.write("\nTest Metrics:\n")
            f.write(f"Mean Squared Error: {mse_test}\n")
            f.write(f"R^2: {r2_test}\n")
            f.write(f"Mean Absolute Percentage Error: {mape_test}\n")
        
        train_predictions = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred})
        test_predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
        train_predictions_path = os.path.join(args.output_data_dir, "train_predictions.csv")
        test_predictions_path = os.path.join(args.output_data_dir, "test_predictions.csv")
        train_predictions.to_csv(train_predictions_path, index=False)
        test_predictions.to_csv(test_predictions_path, index=False)

        output_prefix = 'output'
        s3_client.upload_file(metrics_path, args.bucket, f'{output_prefix}/metrics.txt')
        s3_client.upload_file(train_predictions_path, args.bucket, f'{output_prefix}/train_predictions.csv')
        s3_client.upload_file(test_predictions_path, args.bucket, f'{output_prefix}/test_predictions.csv')
    except Exception as e:
        print(f"Error during evaluation and saving metrics: {e}")
        raise
