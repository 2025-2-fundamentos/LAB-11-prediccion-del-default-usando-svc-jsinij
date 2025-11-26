# flake8: noqa: E501

import os
import time
import gzip
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)
from sklearn.metrics import confusion_matrix


# ===========================================================
# 1. LIMPIEZA DE DATOS
# ===========================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Renombrar variable objetivo
    data = data.rename(columns={"default payment next month": "default"})

    # Eliminar columna ID
    if "ID" in data.columns:
        data = data.drop(columns="ID")

    # 0 se considera NA en EDUCATION y MARRIAGE
    data["EDUCATION"] = data["EDUCATION"].replace(0, np.nan)
    data["MARRIAGE"] = data["MARRIAGE"].replace(0, np.nan)

    # EDUCATION > 4 → others(4)
    data.loc[data["EDUCATION"] > 4, "EDUCATION"] = 4

    # Eliminar registros con valores faltantes
    data = data.dropna()

    return data


# ===========================================================
# 2. SEPARAR VARIABLES
# ===========================================================

def get_features_target(df: pd.DataFrame, target: str):
    X = df.drop(columns=target)
    y = df[target]
    return X, y


# ===========================================================
# 3. PIPELINE COMPLETO
# ===========================================================

def create_pipeline(df: pd.DataFrame) -> Pipeline:
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numeric = [col for col in df.columns if col not in categorical]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("select_k", SelectKBest(f_classif)),
            ("model", SVC()),
        ]
    )

    return pipeline


# ===========================================================
# 4. GRIDSEARCHCV (obligatorio por el test)
# ===========================================================

def optimize_hyperparameters(pipeline, X_train, y_train):
    param_grid = {
        "pca__n_components": [21],
        "select_k__k": [12],
        "model__C": [0.8],
        "model__kernel": ["rbf"],
        "model__gamma": [0.1],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)
    return grid


# ===========================================================
# 5. GUARDAR MODELO
# ===========================================================

def save_model(model):
    os.makedirs("files/models", exist_ok=True)

    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)


# ===========================================================
# 6. MÉTRICAS DE DESEMPEÑO
# ===========================================================
def calculate_metrics(model, x_train, y_train, x_test, y_test):
    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    metrics_train = {
        "type": "metrics",
        "dataset": "train",
        "accuracy": round(accuracy_score(y_train, pred_train), 3),
        "balanced_accuracy": round(balanced_accuracy_score(y_train, pred_train), 3),
        "precision": round(precision_score(y_train, pred_train), 3),
        "recall": round(recall_score(y_train, pred_train), 3),
        "f1_score": round(f1_score(y_train, pred_train), 3)  # <--- cambiar clave aquí
    }

    metrics_test = {
        "type": "metrics",
        "dataset": "test",
        "accuracy": round(accuracy_score(y_test, pred_test), 3),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred_test), 3),
        "precision": round(precision_score(y_test, pred_test), 3),
        "recall": round(recall_score(y_test, pred_test), 3),
        "f1_score": round(f1_score(y_test, pred_test), 3)  # <--- cambiar clave aquí
    }

    return metrics_train, metrics_test





# ===========================================================
# 7. MATRIZ DE CONFUSIÓN
# ===========================================================

def calculate_confusion_matrix(model, X_train, y_train, X_test, y_test):
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    cm_train = confusion_matrix(y_train, pred_train)
    cm_test = confusion_matrix(y_test, pred_test)

    cm_train_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {"predicted_0": int(cm_train[0, 0]), "predicted_1": int(cm_train[0, 1])},
        "true_1": {"predicted_0": int(cm_train[1, 0]), "predicted_1": int(cm_train[1, 1])},
    }

    cm_test_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {"predicted_0": int(cm_test[0, 0]), "predicted_1": int(cm_test[0, 1])},
        "true_1": {"predicted_0": int(cm_test[1, 0]), "predicted_1": int(cm_test[1, 1])},
    }

    return cm_train_dict, cm_test_dict


# ===========================================================
# MAIN
# ===========================================================

if __name__ == "__main__":

    # Cargar datos comprimidos
    train_df = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test_df = pd.read_csv("files/input/test_data.csv.zip", compression="zip")

    # Limpiar datos
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    # Separar variables
    X_train, y_train = get_features_target(train_df, "default")
    X_test, y_test = get_features_target(test_df, "default")

    # Pipeline
    pipeline = create_pipeline(X_train)

    # GridSearch
    start = time.time()
    model = optimize_hyperparameters(pipeline, X_train, y_train)
    end = time.time()

    print("Best params:", model.best_params_)
    print(f"Optimization time: {end - start:.2f} seconds")

    # Guardar modelo
    save_model(model)

    # Métricas
    metrics_train, metrics_test = calculate_metrics(model, X_train, y_train, X_test, y_test)

    # Matrices de confusión
    cm_train, cm_test = calculate_confusion_matrix(model, X_train, y_train, X_test, y_test)

    # Guardar métricas en JSON
    os.makedirs("files/output", exist_ok=True)
    all_metrics = [metrics_train, metrics_test, cm_train, cm_test]

    pd.DataFrame(all_metrics).to_json("files/output/metrics.json", orient="records", lines=True)