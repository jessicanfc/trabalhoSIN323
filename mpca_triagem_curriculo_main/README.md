# Classificação Automática de Currículos de Profissionais de TI

Estudo realizado como parte dos requisitos para a obtenção do grau de Mestre em Computação Aplicada pelo Instituto Federal do Espírito Santo e apresentado no 20º Encontro Nacional de Inteligência Artificial e Computacional (ENIAC). O objetivo desta pesquisa é fazer uma comparação entre diferentes abordagens para a classificação automatizada de currículos de profissionais da área de Tecnologia da Informação e Comunicação (TIC). Essas abordagens variam desde algoritmos tradicionais até modelos avançados baseados em redes neurais profundas, incluindo também o uso de modelos pré-treinados de linguagem. A avaliação dessas abordagens foi realizada em uma base de dados que contém 27.405 currículos, os quais foram categorizados em oito grupos distintos relacionados às áreas de atuação dos profissionais de TIC.

## [Base de dados](https://github.com/florex/resume_corpus)

Fonte: Jiechieu, K.F.F., Tsopze, N. Skills prediction based on multi-label resume classification using CNN with model predictions explanation. Neural Comput & Applic (2020).  [https://doi.org/10.1007/s00521-020-05302-x](https://doi.org/10.1007/s00521-020-05302-x)

## Fluxo de dados
```mermaid
graph LR
A(Texto do currículo) --> B(Pré-processamento)
B --> C(Representações numéricas)
C --> D(Aplicação dos algoritmos)
D --> E(Avaliação)
```

## Algoritmos avaliados

**Algoritmos Tradicionais:**
- Árvore de Decisão
- CatBoost
- Extra Trees
- Floresta Aleatória
- KNN
- LightGBM
- MLP
- Regressão Logística
- SVM
- XGBoost

**Algoritmos baseados em redes neurais profundas:**
- CNN
- CNN+BiLSTM

**Modelos pré-treinados:**
- ALBERT
- BERT
- DistilBERT
- RoBERTa


## Métricas de Avaliação

- Acurácia
- Precisão
- Cobertura
- Medida-F
