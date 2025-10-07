# Análise de Churn — Resumo Executivo

O Problema:

Uma empresa de telecomunicações estava perdendo clientes, mas não sabia como identificar proativamente os clientes em risco.

Minha Abordagem:

Desenvolvi um modelo de Machine Learning em Python para prever a probabilidade de churn. O processo incluiu limpeza de dados, análise exploratória e a comparação de múltiplos algoritmos. O desafio principal foi o baixo Recall do modelo inicial (apenas 57%), que foi resolvido tratando o desbalanceamento dos dados. Essa abordagem estratégica elevou o Recall para 80%...

Resultados e Impacto:

O modelo final (Regressão Logística) alcançou um Recall de 80%, o que significa que ele é capaz de identificar corretamente 8 de cada 10 clientes que iriam cancelar o serviço. Também realizei uma análise de trade-off para permitir que a empresa ajuste a sensibilidade do modelo conforme a sua estratégia de retenção.

Para a análise técnica completa, explore o notebook `analise_churn.ipynb`.
 
 ## Como rodar o projeto (scaffolded)
 Abaixo está um passo-a-passo mínimo para treinar o modelo localmente e rodar a API de demonstração.
 1. Instale dependências:
	 pip install -r requirements.txt
 2. Treine o modelo:
	 python -m src.train --config config.yaml
 3. Teste predições no CSV de entrada:
	 python -m src.predict output/LogisticRegression_class_weight.joblib WA_Fn-UseC_-Telco-Customer-Churn.csv
 4. Rodar a API localmente (após treinar):
	 uvicorn app.main:app --reload --port 8080
 
 Arquivos adicionados:
 - `src/train.py` — script de treino reprodutível (gera artifacts em `output/`).
 - `src/predict.py` — utilitário de inferência em lote.
 - `src/preprocess.py` — funções de limpeza e split.
 - `app/main.py` — FastAPI app minimal para demo.
 - `config.yaml` — configurações (caminhos de dataset e output).
 - `Dockerfile` — containerize o serviço demo.
 
 Sugestões de próximos passos:
 - Adicionar SHAP explainability e um notebook com interpretações.
 - Incluir pequenos testes unitários e um workflow de CI.
 - Melhorar HPO com Optuna e tracking de experimentos (MLflow).
 
Experiment tracking:
- Este projeto tenta registrar runs no MLflow se a biblioteca estiver instalada. Caso contrário, os metadados de cada execução são anexados em `output/runs.csv` (parâmetros, métricas e caminhos dos artifacts).


<!-- images removed: LinkedIn media cleaned up -->
