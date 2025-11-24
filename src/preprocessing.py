"""
Funções utilitárias para limpeza, exploração e pré-processamento
do dataset.
Estas funções são usadas no Notebook 01 (EDA e Pré-processamento)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler


# --------------------- LOADING & CLEANING -------------------

def load_raw_dataset(path: str) -> pd.DataFrame:
    """Carrega o dataset bruto."""
    return pd.read_csv(path)


def remove_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove colunas desnecessárias."""
    return df.drop(columns=cols, errors="ignore")


def split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa x (dados preditivos) e y (dados alvo).
    """

    y = df[target]
    x = df.drop(columns=[target])

    return x, y


# -------------------------- EDA ------------------------------

def plot_histograms(df: pd.DataFrame, bins: int = 30):
    """Plota histogramas de todas as colunas numéricas."""
    df.hist(bins=bins, figsize=(14, 10))
    plt.suptitle("Histogramas das Variáveis Numéricas")
    plt.tight_layout()
    plt.show()


def plot_target_distribution(y: pd.Series):
    """Plota distribuição da variável alvo categórica."""
    sns.countplot(x=y)
    plt.title("Distribuição da Variável Alvo (Classes)")
    plt.show()


def plot_boxplots(df: pd.DataFrame):
    """Plota boxplots das features numéricas."""
    num_cols = df.select_dtypes(include=[np.number]).columns

    fig, axes = plt.subplots(
        nrows=int(np.ceil(len(num_cols) / 3)),
        ncols=3,
        figsize=(16, 12)
    )
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame):
    """Plota matriz de correlação."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Matriz de Correlação")
    plt.show()


# ------------------ OUTLIER DETECTION ------------------------

def find_outliers_iqr(series: pd.Series) -> tuple[pd.Series, float, float]:
    """Identifica outliers usando IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]
    return outliers, lower, upper


# --------------------- NORMALIZATION -------------------------
def fit_and_save_scaler(X: pd.DataFrame, save_path: str) -> StandardScaler:
    """
    Ajusta StandardScaler nos dados e salva para reuso.
    NÃO retorna dados normalizados — isso deve ser feito
    dentro do cross-validation para evitar data leakage.
    """
    scaler = StandardScaler()
    scaler.fit(X)

    joblib.dump(scaler, save_path)
    return scaler


# -------------------- SAVE DATASETS --------------------------
def save_processed_dataset(df: pd.DataFrame, path: str):
    """Salva dataset processado em CSV."""
    df.to_csv(path, index=False)
