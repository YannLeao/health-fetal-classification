import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import os

def run_classification_experiment(model, param_grid, X, y, model_name):
    """
    Executa o pipeline de treinamento com GridSearch e validação cruzada estratificada.
    Salva os resultados em CSV.
    """

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        'accuracy' : 'accuracy',
        'precision' : make_scorer(precision_score, average='weighted'),
        'recall' : make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }

    print(f"--- Iniciando o GridSearch para {model_name} ---")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit='f1',
        return_train_score=False,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X, y)

    print(f"Melhor F1 para {model_name}: {grid.best_score_:.4f}")
    save_metrics_to_csv(grid, model_name)
    
    return grid

def save_metrics_to_csv(grid_object, model_name):
    """
    Extrai médias e desvios padrão do GridSearch e salva em CSV.
    Requisito: Média e desvio padrão para cada algoritmo[cite: 12].
    """
    results = pd.DataFrame(grid_object.cv_results_)
    
    cols_to_keep = ['params']
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        cols_to_keep.append(f'mean_test_{metric}') # Média
        cols_to_keep.append(f'std_test_{metric}')  # Desvio Padrão
    
    final_df = results[cols_to_keep].copy()
    
    rename_map = {}
    for col in final_df.columns:
        if 'mean_test_' in col:
            rename_map[col] = f'Mean {col.replace("mean_test_", "").capitalize()}'
        elif 'std_test_' in col:
            rename_map[col] = f'Std {col.replace("std_test_", "").capitalize()}'
            
    final_df.rename(columns=rename_map, inplace=True)
    
    
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, 'results', 'metrics')
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar
    filepath = os.path.join(output_dir, f'{model_name}_results.csv')
    final_df.to_csv(filepath, index=False)
    print(f"Resultados salvos em: {filepath}")