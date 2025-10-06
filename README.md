# Análise de Churn — Resumo Executivo

O Problema:

Uma empresa de telecomunicações estava perdendo clientes, mas não sabia como identificar proativamente os clientes em risco.

Minha Abordagem:

Desenvolvi um modelo de Machine Learning em Python para prever a probabilidade de churn. O processo incluiu limpeza de dados, análise exploratória e a comparação de múltiplos algoritmos. O desafio principal foi o baixo Recall do modelo inicial (apenas 57%), que foi resolvido tratando o desbalanceamento dos dados. Essa abordagem estratégica elevou o Recall para 80%...

Resultados e Impacto:

O modelo final (Regressão Logística) alcançou um Recall de 80%, o que significa que ele é capaz de identificar corretamente 8 de cada 10 clientes que iriam cancelar o serviço. Também realizei uma análise de trade-off para permitir que a empresa ajuste a sensibilidade do modelo conforme a sua estratégia de retenção.

Para a análise técnica completa, explore o notebook `analise_churn.ipynb`.
