# Fetal Health Classification — Descrição do Dataset

**Disponível em**: [Fetal Health](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification)

Este dataset contém dados provenientes de exames de cardiotocografia (CTG), que monitoram a frequência cardíaca fetal (FHR — *Fetal Heart Rate*) e as contrações uterinas (UC — *Uterine Contractions*).  
Cada linha representa um exame, e cada coluna representa uma característica extraída automaticamente desses sinais.

O objetivo é classificar o estado de saúde fetal em três categorias:
- **Normal**
- **Suspect**
- **Pathological**

Não há valores ausentes no dataset, e as features são diretamente derivadas dos sinais coletados.

## **Descrição das Colunas**

| Coluna | Descrição |
|--------|-----------|
| **baseline_value** | Valor basal do batimento cardíaco fetal (FHR) em bpm, calculado em períodos sem acelerações ou desacelerações. |
| **accelerations** | Número de acelerações por segundo — picos curtos de aumento na FHR. |
| **fetal_movement** | Quantidade de movimentos fetais detectados por segundo. |
| **uterine_contractions** | Quantidade de contrações uterinas por segundo. |
| **light_decelerations** | Número de desacelerações leves por segundo — quedas pequenas na FHR. |
| **severe_decelerations** | Número de desacelerações severas por segundo — quedas bruscas e mais longas. |
| **prolongued_decelerations** | Número de desacelerações prolongadas por segundo — quedas de duração elevada, indicativas de possível sofrimento fetal. |
| **abnormal_short_term_variability** | Percentual do tempo com variabilidade de curto prazo anormal na FHR. |
| **mean_value_of_short_term_variability** | Valor médio da variabilidade de curto prazo (STV) da FHR. |
| **percentage_of_time_with_abnormal_long_term_variability** | Percentual do tempo com variabilidade de longo prazo (LTV) anormal. |
| **mean_value_of_long_term_variability** | Valor médio da variabilidade de longo prazo da FHR. |
| **histogram_width** | Largura do histograma da FHR — medida da dispersão dos valores. |
| **histogram_min** | Mínimo valor observado no histograma da FHR. |
| **histogram_max** | Máximo valor observado no histograma da FHR. |
| **histogram_number_of_peaks** | Número de picos identificados no histograma da FHR. |
| **histogram_number_of_zeroes** | Número de pontos do histograma com frequência igual a zero. |
| **histogram_mode** | Modo do histograma — valor mais frequente de FHR. |
| **histogram_mean** | Média dos valores de FHR durante o exame. |
| **histogram_median** | Mediana dos valores de FHR. |
| **histogram_variance** | Variância dos valores de FHR, indicando dispersão. |
| **histogram_tendency** | Tendência do histograma — medida relacionada à assimetria da distribuição. |
| **fetal_health** | Classe alvo: 1 = Normal, 2 = Suspect, 3 = Pathological. |

## **Contexto Clínico Resumido**

- A cardiotocografia (CTG) monitora simultaneamente a FHR e as contrações uterinas.
- Valores basais normais da FHR geralmente ficam entre **110–160 bpm**.
- A presença, duração e frequência de desacelerações são indicadores críticos de risco fetal.
- A variabilidade (STV e LTV) é usada para avaliar a responsividade do sistema nervoso fetal.
- As estatísticas do histograma representam a distribuição geral da FHR ao longo do exame.

## **Notas Importantes para Exploração e Modelagem**

- As features possuem escalas muito diferentes → recomenda-se **normalização ou padronização**.
- O dataset é **desbalanceado**, com predomínio da classe *Normal*.
- Features relacionadas a acelerações, desacelerações e variabilidade costumam ter forte impacto em modelos preditivos.

## **Referências**

- Cardiotocography Dataset — UCI Machine Learning Repository:  
  https://archive.ics.uci.edu/dataset/193/cardiotocography
- Fetal Health Classification — Kaggle:  
  https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
- Article: *Early Diagnosis and Classification of Fetal Health Status from a Fetal Cardiotocography Dataset Using Ensemble Learning*  
  https://www.mdpi.com/2075-4418/13/15/2471
- Article: *Fetal Well-Being Diagnostic Method Based on Cardiotocographic Morphological Pattern Utilizing Autoencoder and Recursive Feature Elimination*  
  https://pubmed.ncbi.nlm.nih.gov/37296783/
- Overview of CTG and interpretation guidelines — Wikipedia:  
  https://en.wikipedia.org/wiki/Cardiotocography