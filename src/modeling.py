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
    Extrai métricas, separa os hiperparâmetros em colunas e salva em CSV.
    Garante que salva na pasta 'results/metrics' na RAIZ do projeto.
    """
    results = pd.DataFrame(grid_object.cv_results_)
 
    params_df = pd.json_normalize(results['params'])
    
    metric_cols = []
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        metric_cols.append(f'mean_test_{metric}')
        metric_cols.append(f'std_test_{metric}')
    
    final_df = pd.concat([params_df, results[metric_cols]], axis=1)
    

    rename_map = {}
    for col in final_df.columns:
        if 'mean_test_' in col:
            rename_map[col] = f'Mean {col.replace("mean_test_", "").capitalize()}'
        elif 'std_test_' in col:
            rename_map[col] = f'Std {col.replace("std_test_", "").capitalize()}'
        else:
            rename_map[col] = col.capitalize()
            
    final_df.rename(columns=rename_map, inplace=True)
    

    if 'Mean F1' in final_df.columns:
        final_df = final_df.sort_values(by='Mean F1', ascending=False)
    
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(src_dir)
    output_dir = os.path.join(project_root, 'results', 'metrics')
    
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f'{model_name}_results.csv')
    final_df.to_csv(filepath, index=False)
    print(f"Resultados salvos e tratados em: {filepath}")