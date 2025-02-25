

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score


#----------------------------------------------------- 1° Etapa: Inserir Base de Dados Original -------------------------------------------------------

df = pd.read_excel("C:\\Users\\elisa\\Projetos Inteligencia Artificial\\Projeto Regressão Linear\\Dados dos Alunos.xlsx")
df= df.drop(['carimbo'], axis=1)

#----------------------------------------------------------- 2° Etapa:Tratamento de Dados ------------------------------------------------------------

# 1) Excluir casos "Outros"
df = df[~df.apply(lambda row: 'Outros' in row.values, axis=1)]

# 2) Funções de conversão (Converter horas em minutos e converter quilômetros e metros)
def horas_para_minutos (horas):
      horas, minutos = map(int, horas.split(':'))
      return horas * 60 + minutos

def km_para_m (distancia):
        km, m = map(int, distancia.split('.'))
        return (km * 1000) + m

# 3) *Convertendo hora de chegada em minutos
df['horario_chegada_min'] = df['horario_chegada'].apply(horas_para_minutos)

# *Convertendo hora de chegada em minutos
df['distancia_metros'] = df['distancia'].astype(str).apply(km_para_m)

# 4) Preenchendo horários de expediente de não trabalhadores
df.fillna({'horario_expediente': 'Nulo'}, inplace=True)

# 5) Função que calcula as horas trabalhadadas baseada no expediente
def calcular_horas(expediente):
    if expediente == '09hrs às 18hrs' or expediente == '08hrs às 17hrs' :
        return 9
    elif expediente == '08hrs às 14hrs' or expediente == '12hrs às 18hrs':
        return 6
    else:
        return 0

# 6) Criando a nova coluna de horas trabalhadas
df['horas_trabalhadas'] = df['horario_expediente'].apply(calcular_horas)

# 7) Separar variáveis de entrada (x) e a saída (y)
X = df[['trabalho', 'horario_expediente', 'meio_transporte', 'horas_trabalhadas', 'distancia_metros']]
y = df['horario_chegada_min']

# 8) Padronizando variáveis categóricas usando o OneHotEncoder
column_transformer = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['trabalho', 'horario_expediente', 'meio_transporte'])],
    remainder='passthrough' #Ignora colunas não selecionadads
)

# 9) Variável que armazena o resultado da transformação
X_encoded = column_transformer.fit_transform(X)

#--------------------------------------------------------- 3° Etapa: Treinamento do Modelo -----------------------------------------------------------

# 1) Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 2) Treinar o modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# 3) Fazer previsões utilizando o modelo
y_pred = model.predict(X_test)

# 4) Calculando erro médio quadrático
mse = mean_squared_error(y_test, y_pred)
print(f"Erro médio quadrático: {mse}")

# 5) Calculando porcentagem de erro
valor_medio_reais = y.mean()
porcentagem_erro = (mse / valor_medio_reais) * 100
print(f"Porcentagem de erro: {porcentagem_erro:.2f}%")

# 6) Calculando coeficiente de determinação
r2 = r2_score(y_test, y_pred)
print(f"Coeficiente de Determinação (R²): {r2}")

#------------------------------------------------------------- 4° Etapa: Plotar o Gráfico ----------------------------------------------------------------

# 1) Plotar gráficos da regressão
plt.figure(figsize=(14, 6))

# 2) Gráfico 1: Valores reais vs. Previsões
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.xlabel('Valores reais')
plt.ylabel('Previsões')
plt.title('Valores reais vs. Previsões')

# 3) Gráfico 2: Resíduos (Erro)
residuos = y_test - y_pred
plt.subplot(1, 2, 2)
plt.scatter(y_test, residuos)
plt.axhline(0, color='red', lw=2)
plt.xlabel('Valores reais')
plt.ylabel('Resíduos')
plt.title('Resíduos (Valores reais - Previsões)')

# 4) Imprimir Gráfico
plt.tight_layout()
plt.show()

#------------------------------------------------------------ 5° Etapa: Testar um Cenário --------------------------------------------------------------

# 1) Inserindo dados do usuário:

print("Trabalho:")
trabalho = input("[1] Não trabalho / [2] Estagiário / [3] CLT/PJ: ").strip().lower()
if trabalho == '1': trabalho = 'Não trabalho'
elif trabalho == '2': trabalho = 'Estagiário'
elif trabalho == '3': trabalho = 'CLT/PJ'
print("\n")

print("Meio de Transporte:")
meio_transporte = input("[1] Transporte Público / [2] Transporte Privado (Carro e Moto) / [3] Serviço de Transporte Pago: ").strip().lower()
if meio_transporte == '1': meio_transporte = 'Transporte Público'
elif meio_transporte == '2': meio_transporte = 'Privado (Carro ou moto)'
elif meio_transporte == '3': meio_transporte = 'Serviço de transporte pago'
print("\n")

print("Horário do Expediente:")
horario_expediente = input("[1] 08hrs às 14hrs / [2] 08hrs às 17hrs / [3] 09hrs às 18hrs / [4] 12hrs às 18hrs / [5] NÃO : ").strip().lower()
if horario_expediente == '1': horario_expediente = '08hrs às 14hrs'
elif horario_expediente == '2': horario_expediente = '08hrs às 17hrs'
elif horario_expediente == '3': horario_expediente = '09hrs às 18hrs'
elif horario_expediente == '4': horario_expediente = '12hrs às 18hrs'
elif horario_expediente == '5': horario_expediente = 'Nulo'
print("\n")

distancia = input("Qual a distância da faculdade do seu ponto de partida (km,m)? ");
print("\n")

# 2) Criar um novo DataFrame com as características específicas do usuário
novo_df = pd.DataFrame({
    'trabalho': [trabalho],
    'horario_expediente': [horario_expediente],
    'horas_trabalhadas' : [calcular_horas(horario_expediente)],
    'meio_transporte': [meio_transporte],
    'distancia_metros': [km_para_m(distancia)]
})

# 3) Codificar o novo dado da mesma forma que fizemos com os dados de treino
novo_df_encoded = column_transformer.transform(novo_df)

# 4) Fazer a previsão com o modelo treinado
horario_previsto_minutos =  model.predict(novo_df_encoded)[0]

# 5) Converter o horário previsto em horas e minutos
horas_previstas = int(horario_previsto_minutos // 60)
minutos_previstos = int(horario_previsto_minutos % 60)

# 6) Expondo resultado da pesquisa
print(f"Horário previsto de chegada: {horas_previstas:02d}:{minutos_previstos:02d}")