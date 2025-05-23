import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import gradio as gr
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ======== CONFIGURAÇÃO ======== #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "modelos"
DATA_PATH = DATA_DIR / "dataset_transformado444.csv"
TOP_FEATURES_PATH = ARTIFACTS_DIR / "top_features.json"
TARGET = "TEMP_MIN_ANT_C"
FEATURES_A_REMOVER = ["UMIDADE_MAX_ANT_PERCENT", "UMIDADE_MIN_ANT_PERCENT","HORA_UTC","ANO","RADIACAO_GLOBAL_KJ_M2","hora_sin","hora_cos"]

DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ======== FUNÇÕES ======== #

def carregar_dados():
    df = pd.read_csv(DATA_PATH).dropna()
    for col in FEATURES_A_REMOVER:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def treinar_modelo_completo(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"📊 Modelo completo - R²: {r2_score(y_test, y_pred):.4f} | MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    joblib.dump(pipeline, MODELS_DIR / "modelo_completo.pkl")
    joblib.dump((X_test, y_test), MODELS_DIR / "test_data.pkl")

    return pipeline, X_train, X_test, y_train, y_test

def identificar_top8_features(modelo_completo, X_train):
    xgb_model = modelo_completo.named_steps['model']
    importancias = xgb_model.feature_importances_
    indices = np.argsort(importancias)[::-1][:8]
    top8 = X_train.columns[indices].tolist()

    with open(TOP_FEATURES_PATH, "w") as f:
        json.dump(top8, f, indent=4)

    print(f"🏆 Top 8 features: {top8}")

    plt.figure(figsize=(10, 6))
    plt.title("Importância das Top 8 Features")
    plt.barh(range(8), importancias[indices][::-1], align='center')
    plt.yticks(range(8), [X_train.columns[i] for i in indices[::-1]])
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "feature_importance.png")
    plt.close()

    return top8

def treinar_modelo_top8(df, top8):
    df = df[top8 + [TARGET]]
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"📊 Modelo Top 8 - R²: {r2_score(y_test, y_pred):.4f} | MAE: {mean_absolute_error(y_test, y_pred):.4f}")

    joblib.dump(pipeline, MODELS_DIR / "modelo_top8.pkl")
    joblib.dump((X_test, y_test), MODELS_DIR / "test_data_top8.pkl")

    return pipeline, X_test, y_test, y_pred

def criar_scatter_plot(y_test, y_pred):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
    plt.xlabel("Valores Reais")
    plt.ylabel("Previsões")
    plt.title("Previsões vs Valores Reais")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "scatter_plot.png")
    plt.close()

def obter_exemplos_teste(X_test, n=5):
    return X_test.sample(n=n, random_state=42).values.tolist()

def criar_interface():
    modelo_top8 = joblib.load(MODELS_DIR / "modelo_top8.pkl")
    with open(TOP_FEATURES_PATH) as f:
        top8 = json.load(f)
    df = carregar_dados()
    X_test, y_test = joblib.load(MODELS_DIR / "test_data_top8.pkl")

    y_pred = modelo_top8.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    exemplos = obter_exemplos_teste(X_test, n=5)

    def prever_dia_seguinte(*valores):
        input_data = pd.DataFrame([valores], columns=top8)
        predicao = modelo_top8.predict(input_data)[0]

        diffs = (X_test - valores).abs().sum(axis=1)
        idx_proximo = diffs.idxmin()
        real_proximo = y_test.loc[idx_proximo]

        resultado = f"🌤️ Previsão para o dia seguinte: {predicao:.2f}°C\n"
        resultado += f"📏 Valor real no conjunto de teste: {real_proximo:.2f}°C\n"
        resultado += f"🔍 Diferença: {abs(predicao - real_proximo):.2f}°C\n"
        resultado += f"📈 R² do modelo: {r2:.4f}"

        return resultado, ARTIFACTS_DIR / "feature_importance.png", ARTIFACTS_DIR / "scatter_plot.png"

    with gr.Blocks() as demo:
        gr.Markdown("## 🌡️ Previsão de Temperatura Mínima com Top 8 Features")

        with gr.Row():
            with gr.Column():
                inputs = [gr.Number(label=feat) for feat in top8]
                btn = gr.Button("Prever")

                gr.Markdown("### Exemplos do Conjunto de Teste:")
                for i, exemplo in enumerate(exemplos):
                    btn_exemplo = gr.Button(f"Exemplo {i+1}")

                    def preencher_exemplo(valores=exemplo):
                        return [gr.update(value=v) for v in valores]

                    btn_exemplo.click(fn=preencher_exemplo, inputs=[], outputs=inputs)

            with gr.Column():
                saida = gr.Textbox(label="Resultado", lines=6)
                importancia_img = gr.Image(label="Importância das Features")
                scatter_img = gr.Image(label="Dispersão Previsões x Reais")

        btn.click(fn=prever_dia_seguinte, inputs=inputs, outputs=[saida, importancia_img, scatter_img])

    return demo

# ======== EXECUÇÃO ======== #
if __name__ == "__main__":
    df = carregar_dados()

    if not (MODELS_DIR / "modelo_completo.pkl").exists():
        modelo_completo, X_train, X_test, y_train, y_test = treinar_modelo_completo(df)
        top8 = identificar_top8_features(modelo_completo, X_train)
        modelo_top8, X_test_top8, y_test_top8, y_pred_top8 = treinar_modelo_top8(df, top8)
        criar_scatter_plot(y_test_top8, y_pred_top8)
    elif not (MODELS_DIR / "modelo_top8.pkl").exists():
        modelo_completo = joblib.load(MODELS_DIR / "modelo_completo.pkl")
        df = carregar_dados()
        top8 = identificar_top8_features(modelo_completo, df.drop(columns=[TARGET]))
        modelo_top8, X_test_top8, y_test_top8, y_pred_top8 = treinar_modelo_top8(df, top8)
        criar_scatter_plot(y_test_top8, y_pred_top8)

    demo = criar_interface()
    demo.launch()