
# 1) Inserindo bibliotecas:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2) Inserindo tabela de dados do artigo:
# Obs: Para usar o arquivo é preciso fazer upload dele na lateral esquerda
df = pd.read_excel("C:\\Users\\elisa\\Projetos Inteligencia Artificial\\Projeto K-Nearest Neighbor\\Dados do Teste AQ-10.xlsx")

# 3) Retirando dados faltantes
df = df[(df != '?').all(axis=1)]

# 4) Converte as colunas 'Age Numeric' e 'Result Numeric' para tipo numérico, substituindo valores não convertíveis por NaN
df['Age Numeric'] = pd.to_numeric(df['Age Numeric'], errors='coerce')
df['Result Numeric'] = pd.to_numeric(df['Result Numeric'], errors='coerce')

# Remove linhas com NaN em 'Age Numeric' ou 'Result Numeric'
df = df.dropna(subset=['Age Numeric', 'Result Numeric'])

# 5) Retirando outliers nos valores da idade

# Calcular o primeiro e o terceiro quartil (Q1 e Q3)
Q1 = df['Age Numeric'].quantile(0.25)
Q3 = df['Age Numeric'].quantile(0.75)

# Calcular o intervalo interquartil (IQR)
IQR = Q3 - Q1

# Definir os limites superior e inferior para os outliers
limite_inferior = Q1 - 2 * IQR
limite_superior = Q3 + 2 * IQR

# Filtrar o dataframe para remover os outliers
df_sem_outliers = df[(df['Age Numeric'] >= limite_inferior) & (df['Age Numeric'] <= limite_superior)]

# 6) Transformando colunas em booleanas
df['Gender'] = df['Gender'].map({'f': True, 'm': False})
df['Jundice'] = df['Jundice'].map({'yes': True, 'no': False})
df['Autism'] = df['Autism'].map({'yes': True, 'no': False})
df['Used App Before'] = df['Used App Before'].map({'yes': True, 'no': False})
df['Class/ASD'] = df['Class/ASD'].map({'YES': True, 'NO': False})  # Verifique se os valores estão em caixa alta

# Lista de colunas de pontuação
score_columns = ['A1 - Score', 'A2 - Score', 'A3 - Score', 'A4 - Score', 'A5 - Score',
                 'A6 - Score', 'A7 - Score', 'A8 - Score', 'A9 - Score', 'A10 - Score']  # Corrigido "A6 = Score"

# Convertendo colunas de score para booleano
for col in score_columns:
    df[col] = df[col].apply(lambda x: x == 1)

# Definir variáveis independentes e dependentes
X = df.drop(columns=['Class/ASD'])
y = df['Class/ASD']

# 7) Define as colunas categóricas e numéricas

# Variáveis categóricas
categorical_features = ['Ethnicity', 'Country of Residence', 'Age Description', 'Relation']

# Variáveis booleanas
boolean_features = ['Gender', 'Jundice', 'Autism', 'Used App Before', 'A1 - Score', 'A2 - Score', 'A3 - Score',
                   'A4 - Score', 'A5 - Score', 'A6 - Score', 'A7 - Score', 'A8 - Score', 'A9 - Score', 'A10 - Score']

# Variáveis numéricas
numeric_features = ['Age Numeric','Result Numeric']

# 8) Padronizando variáveis categóricas usando o OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),      # Codificação one-hot para categorias
        ('bool', 'passthrough', boolean_features),           # Booleanos (True/False)
        ('num', StandardScaler(), numeric_features)          # Normalização de dados numéricos
])

# 9) Cria um pipeline para aplicar as transformações e o classificador KNN
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# 10) Realizando validação cruzada como sugerido no artigo:
scores = cross_val_score(pipeline, X, y, cv=10)

# Exibindo os resultados
print("Scores de validação cruzada:", scores)
print("Acurácia média:", scores.mean())

"""**3) Treniamento do Modelo**"""

# 11) Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 12) Ajustando o pipeline aos dados de treino
pipeline.fit(X_train, y_train)

# Testando Casos -----------------------------------------------------------------------------------------------

# 13) Realizando previsões no conjunto de teste
y_pred = pipeline.predict(X_test)

# 14) Métricas de Avaliação

# a) Acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia no conjunto de teste: {accuracy:.4f}')

# b) Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(conf_matrix)

# c) Relatório de classificação (inclui precisão, recall, f1-score, etc.)
class_report = classification_report(y_test, y_pred)
print("Relatório de Classificação:")
print(class_report)

# 15) Refazendo processo completo para avaliar somente as questões A1 a A10

# ---------------------------------------------- 10 questões ----------------------------------------------------------
# Definindo novas colunas de entrada:
X_score_10 = df.iloc[:, :10] # Selecionando as 10 primeiras colunas da base de dados = Scores

# Adequando o pré processador do pipeline para somente valores booleanos:
pre_processador_score = ColumnTransformer(
    transformers=[
        ('bool', 'passthrough', X_score_10.columns),           # Booleanos (True/False)
])

# Criando um pipeline para aplicar transformações:
pipeline_score = Pipeline([
    ('preprocessor', pre_processador_score),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# Realizando a validação cruzada:
scores = cross_val_score(pipeline_score, X_score_10, y, cv=10)

# Treinando o modelo e realizando previsões no conjunto de teste:
X_train_score, X_test_score, y_train, y_test = train_test_split(X_score_10, y, test_size=0.2, random_state=42, stratify=y)
pipeline_score.fit(X_train_score, y_train)
y_pred_score = pipeline_score.predict(X_test_score)

# Métricas de avaliação:
# a) Acurácia
accuracy = accuracy_score(y_test, y_pred_score)
print(f'Acurácia no conjunto de teste: {accuracy:.4f}')

# b) Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred_score)
print("Matriz de Confusão:")
print(conf_matrix)

# c) Relatório de classificação (inclui precisão, recall, f1-score, etc.)
class_report = classification_report(y_test, y_pred_score)
print("Relatório de Classificação:")
print(class_report)

# -------------------------------------------- Apenas Q9,Q6,Q5----------------------------------------------------------
# Definindo novas colunas de entrada:
X_score_3 = df.iloc[:, [5, 6, 9]] # Selecionando as 3 colunas selecionadas no artigo da base de dados = Scores

# Adequando o pré processador do pipeline para somente valores booleanos:
pre_processador_score = ColumnTransformer(
    transformers=[
        ('bool', 'passthrough', X_score_3.columns),           # Booleanos (True/False)
])

# Criando um pipeline para aplicar transformações:
pipeline_score = Pipeline([
    ('preprocessor', pre_processador_score),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# Realizando a validação cruzada:
scores = cross_val_score(pipeline_score, X_score_3, y, cv=10)

# Treinando o modelo e realizando previsões no conjunto de teste:
X_train_score, X_test_score, y_train, y_test = train_test_split(X_score_3, y, test_size=0.2, random_state=42, stratify=y)
pipeline_score.fit(X_train_score, y_train)
y_pred_score = pipeline_score.predict(X_test_score)

# Métricas de avaliação:
# a) Acurácia
accuracy = accuracy_score(y_test, y_pred_score)
print(f'Acurácia no conjunto de teste: {accuracy:.4f}')

# b) Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred_score)
print("Matriz de Confusão:")
print(conf_matrix)

# c) Relatório de classificação (inclui precisão, recall, f1-score, etc.)
class_report = classification_report(y_test, y_pred_score)
print("Relatório de Classificação:")
print(class_report)


# -------------------------------------------- Apenas Q9,Q6,Q5,Q4,Q3,Q1 ----------------------------------------------------------
# Definindo novas colunas de entrada:
X_score_6 = df.iloc[:, [1, 3, 4, 5, 6, 9]] # Selecionando as 6 colunas selecionadas no artigo da base de dados = Scores

# Adequando o pré processador do pipeline para somente valores booleanos:
pre_processador_score = ColumnTransformer(
    transformers=[
        ('bool', 'passthrough', X_score_6.columns),           # Booleanos (True/False)
])

# Criando um pipeline para aplicar transformações:
pipeline_score = Pipeline([
    ('preprocessor', pre_processador_score),
    ('classifier', KNeighborsClassifier(n_neighbors=3))
])

# Realizando a validação cruzada:
scores = cross_val_score(pipeline_score, X_score_6, y, cv=10)

# Treinando o modelo e realizando previsões no conjunto de teste:
X_train_score, X_test_score, y_train, y_test = train_test_split(X_score_6, y, test_size=0.2, random_state=42, stratify=y)
pipeline_score.fit(X_train_score, y_train)
y_pred_score = pipeline_score.predict(X_test_score)

# Métricas de avaliação:
# a) Acurácia
accuracy = accuracy_score(y_test, y_pred_score)
print(f'Acurácia no conjunto de teste: {accuracy:.4f}')

# b) Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred_score)
print("Matriz de Confusão:")
print(conf_matrix)

# c) Relatório de classificação (inclui precisão, recall, f1-score, etc.)
class_report = classification_report(y_test, y_pred_score)
print("Relatório de Classificação:")
print(class_report)