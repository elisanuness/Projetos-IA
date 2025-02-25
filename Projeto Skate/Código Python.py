

#Importação de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# 3° Etapa) Importação de Dados
# Observação: É preciso inserir o caminho do arquivo da base de dados em seu próprio drive.
# Inserindo path do arquivo de dados para o tratamento de dados.
dataset_path = 'C:\\Users\\elisa\\Projetos Inteligencia Artificial\\Projeto Skate\\baseskate.xlsx'

# Carregar a base de dados
df = pd.read_excel(dataset_path, engine='openpyxl')

# 4° Etapa) Tratamento de Dados

# Tratamento de Valores Faltantes
df.dropna(inplace=True)  # Remover linhas com dados faltantes

# Conversão de Tipos de Dados
df['Manobra'] = df['Manobra'].astype('category')
df['Nome do Skatista'] = df['Nome do Skatista'].astype('category')
df['Dificuldade'] = df['Dificuldade'].astype('category')
df['Dificuldade'] = df['Dificuldade'].cat.set_categories(['Baixa', 'Média', 'Alta'], ordered=True)
df['Tempo de Execução (s)'] = df['Tempo de Execução (s)'].astype(float)
df['Classificação Final'] = df['Classificação Final'].astype('int')

# Criação de Variáveis Derivadas
df['Nota Média'] = df[['Nota Juiz 1', 'Nota Juiz 2', 'Nota Juiz 3']].mean(axis=1)

# Definir variáveis independentes e dependentes
X = df.drop(columns=['Classificação Final', 'Nota Juiz 1', 'Nota Juiz 2', 'Nota Juiz 3'])
y = df['Classificação Final']

# 5° Etapa) Função de Interpretação de Dificuldade Dominante de um Skatista

# Parâmetros: DataFrame, nome do skatista, manobra realizada
def dificuldade_predominante(df, skatista, manobra):

    # Verificando a existência da manobra
    existe_manobra = df['Manobra'].isin([manobra]).any()
    if not existe_manobra:
      dificuldade_final = None
      print(f"Manobra '{manobra}' não encontrada no DataFrame.")
      return dificuldade_final

# --------------------------------------- 1° Opção --------------------------------------------------------------------------
    # Caso o DataFrame possua um ou mais registros do skatista realizando a manobra, escolha a dificuldade predominante entre elas, caso exista.
    dificuldades_skatista = df[(df['Nome do Skatista'] == skatista) & (df['Manobra'] == manobra)]['Dificuldade']

    if not dificuldades_skatista.empty:
        moda = dificuldades_skatista.mode()
        if len(moda) == 1:
          dificuldade_predominante_skatista = moda[0]
          dificuldade_final = dificuldade_predominante_skatista
          return dificuldade_final

# --------------------------------------- 2° Opção --------------------------------------------------------------------------
    # Caso o DataFrame não possua um ou mais registros do skatista realizando a manobra, escolha a dificuldade predominante entre as dificuldades de execução da manobra de todos os skatistas, caso exista.
    else:
        dificuldades_todos = df[df['Manobra'] == manobra]['Dificuldade']

        if not dificuldades_todos.empty:
            moda = dificuldades_todos.mode()
            if len(moda) == 1:
              dificuldade_predominante_todos = moda[0]  # A moda é única, retorna o valor
              dificuldade_final = dificuldade_predominante_todos
              return dificuldade_final

# --------------------------------------- 3° Opção --------------------------------------------------------------------------
    # Caso não haja predominância de dificuldade para a manobra escolhida, nem para o skatista escolhido nem para todos:

    # 1° Opção: Selecionar dificuldade da manobra escolhida que teve maior classificação, considerando que o skatista executou a manobra.
    classificacao_skatista = df[(df['Nome do Skatista'] == skatista) & (df['Manobra'] == manobra)][['Classificação Final', 'Dificuldade']]
    if not classificacao_skatista.empty:
      maior_classificacao = classificacao_skatista['Classificação Final'].min()
      dificuldade_maior_classificacao = classificacao_skatista[classificacao_skatista['Classificação Final'] == maior_classificacao]['Dificuldade']
      dificuldade_final = dificuldade_maior_classificacao.max()
      return dificuldade_final

    # 2° Opção: Selecionar dificuldade da manobra escolhida que teve maior classificação, considerando que algum dos skatistas (que não é o escolhido) executou a manobra
    classificacao_todos = df[(df['Manobra'] == manobra)][['Classificação Final', 'Dificuldade']]
    if not classificacao_todos.empty:
      maior_classificacao = classificacao_todos['Classificação Final'].min()
      dificuldade_maior_classificacao = classificacao_todos[classificacao_todos['Classificação Final'] == maior_classificacao]['Dificuldade']
      dificuldade_final = dificuldade_maior_classificacao.max().iloc[0]
      return dificuldade_final

dificuldade_predominante(df, 'Lucas', 'Frontside 180')

# 6° Etapa) Transformação de Colunas

preprocessor = ColumnTransformer(
    transformers=[
        # Ajustando valores de tempo e nota média para terem média zero e desvio padrão igual a um
        ('num', StandardScaler(), ['Tempo de Execução (s)', 'Nota Média']),
        # Convertendo categorias em colunas de valores binários
        ('cat', OneHotEncoder(drop='first', sparse_output=False), ['Nome do Skatista', 'Manobra', 'Dificuldade'])
    ]
)

#7° Etapa) Criando pipelines que incluem a transformação dos dados e os modelos (Árvore de Decisão e Naive Bayes)

# Árvore de Decisão
pipeline_arvoreDeDecisao = Pipeline(steps=[
    ('preprocessador', preprocessor),
    ('classificador', DecisionTreeClassifier(max_depth=5))
])

# Naive Bayes
pipeline_naiveBayes = Pipeline(steps=[
    ('preprocessador', preprocessor),
    ('classificador', GaussianNB())
])

#8° Etapa) Validação Cruzada

# Calcular o número de folds baseado na menor classe
n_classes = y.value_counts().min()
n_folds = min(10, n_classes)

validacao_cruzada = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Validação cruzada para o modelo de Árvore de Decisão
scores_arvoreDeDecisao = cross_val_score(pipeline_arvoreDeDecisao, X, y, cv=validacao_cruzada, scoring='accuracy')
#print(f"Acurácia média da Árvore de Decisão (validação cruzada): {scores_dt.mean():.4f}") # Descomentar para visualizar a Acurácia média da Árvore de Decisões

# Validação cruzada para o modelo de Naive Bayes
scores_naiveBayes = cross_val_score(pipeline_naiveBayes, X, y, cv=validacao_cruzada, scoring='accuracy')
#print(f"Acurácia média do Naive Bayes (validação cruzada): {scores_nb.mean():.4f}") # Descomentar para visualizar a Acurácia média do Naive Bayes

#9° Etapa) Treinamento de Modelos

# Divisão em Conjuntos de Treino e Teste
x_treino, x_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE no conjunto de treino apenas nas colunas numéricas
num_colunas = ['Tempo de Execução (s)', 'Nota Média']
x_treino_num = x_treino[num_colunas]

# Aplicar SMOTE no conjunto de treino
smote = SMOTE(k_neighbors=2,random_state=42)
x_treino_res_num, y_treino_res = smote.fit_resample(x_treino_num, y_treino)

# Juntar as colunas categóricas de volta
x_treino_res = pd.DataFrame(x_treino_res_num, columns=num_colunas)
x_treino_res = pd.concat([x_treino_res, x_treino.loc[x_treino.index].drop(columns=num_colunas)], axis=1)

# Treinar o modelo de Árvore de Decisão
pipeline_arvoreDeDecisao.fit(x_treino_res, y_treino_res)
y_pred_arvoreDeDecisao = pipeline_arvoreDeDecisao.predict(x_teste)

# Treinar modelo de Naive Bayes
pipeline_naiveBayes.fit(x_treino_res, y_treino_res)
y_pred_naiveBayes = pipeline_naiveBayes.predict(x_teste)

#10° Etapa) Avaliação de Modelos

# Definindo acurácia, precisão, recall e f1-score
def avaliar_modelo(y_teste, y_pred):
    acuracia = accuracy_score(y_teste, y_pred)
    precisao = precision_score(y_teste, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_teste, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_teste, y_pred, average='weighted', zero_division=0)

    return acuracia, precisao, recall, f1

# Obter métricas dos modelos
# Árvore de Decisão
metricas_dt = avaliar_modelo(y_teste, y_pred_arvoreDeDecisao)

# Naive Bayes
metricas_nb = avaliar_modelo(y_teste, y_pred_naiveBayes)

# Comparar resultados
resultados = pd.DataFrame({
    'Modelo': ['Árvore de Decisão', ' Naive Bayes'],
    'Acurácia': [f'{metricas_dt[0]* 100:.2f}%', f'{metricas_nb[0]* 100:.2f}%'],
    'Precisão': [f'{metricas_dt[1]* 100:.2f}%', f'{metricas_nb[1]* 100:.2f}%'],
    'Recall': [f'{metricas_dt[2]* 100:.2f}%', f'{metricas_nb[2]* 100:.2f}%'],
    'F1-Score': [f'{metricas_dt[3]* 100:.2f}%', f'{metricas_nb[3]* 100:.2f}%']
})

print("\n\033[1mComparação de Resultados:\033[0m")
print(resultados)

#11° Etapa) Exibir Árvore de Decisão

# Visualizar a Árvore de Decisão
plt.figure(figsize=(60,13))  # Ajustar o tamanho da figura
plot_tree(pipeline_arvoreDeDecisao.named_steps['classificador'],
          feature_names=pipeline_arvoreDeDecisao.named_steps['preprocessador'].get_feature_names_out(),
          class_names=[str(c) for c in y.unique()],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Representação Árvore de Decisão", fontsize=24)
plt.show()

#12° Etapa) Matriz de Confusão

# Calcular e exibir matriz de confusão para a Árvore de Decisão
cm_dt = confusion_matrix(y_teste, y_pred_arvoreDeDecisao)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=pipeline_arvoreDeDecisao.classes_)
disp_dt.plot(cmap='Purples')
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.show()

# Calcular e exibir matriz de confusão para o Naive Bayes
cm_nb = confusion_matrix(y_teste, y_pred_naiveBayes)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=pipeline_naiveBayes.classes_)
disp_nb.plot(cmap='Blues')
plt.title("Matriz de Confusão - Naive Bayes")
plt.show()

#13° Etapa) Sugestões de melhoria para Lucas

# Filtrar os dados apenas para o Lucas no DataFrame original
df_lucas_original = df[df['Nome do Skatista'] == 'Lucas'].copy()

# Calcular a média de tempo para cada manobra executada por Lucas
media_tempos_lucas = df_lucas_original.groupby('Manobra', observed=False)['Tempo de Execução (s)'].mean().to_dict()
media_notas_lucas = df_lucas_original.groupby('Manobra', observed=False)['Nota Média'].mean().to_dict()

# Juntar as sugestões de melhorias e treino
sugestoes_melhoria = {}
sugestoes_treino = {}

# Sugestões para as manobras executadas pelo Lucas (melhorar)
piores_manobras_dt = df_lucas_original[['Manobra', 'Classificação Final', 'Tempo de Execução (s)']]

for index, row in piores_manobras_dt.iterrows():
    manobra = row['Manobra']
    tempo = row['Tempo de Execução (s)']

    df_melhores_tempos = df[df['Manobra']== manobra][['Classificação Final','Tempo de Execução (s)']]
    maior_classificacao = df_melhores_tempos['Classificação Final'].min()
    tempo_maior_classificacao = df_melhores_tempos[df_melhores_tempos['Classificação Final'] == maior_classificacao]['Tempo de Execução (s)']

    tempo_ideal = tempo_maior_classificacao.min()


    tempo_medio = media_tempos_lucas.get(manobra, tempo)

    # Criar um DataFrame com a manobra para prever a classificação
    entrada_prev = pd.DataFrame({
        'Tempo de Execução (s)': [tempo_ideal],  # Usar o tempo atual
        'Nota Média': [media_notas_lucas.get(manobra, 0)],  # Média do tempo para a manobra
        'Nome do Skatista': ['Lucas'],
        'Manobra': [manobra],
        'Dificuldade': [dificuldade_predominante(df, 'Lucas', manobra)]  # Obter dificuldade
    })

    class_pred = pipeline_arvoreDeDecisao.predict(entrada_prev)

    # Análise dos erros
    erros = []
    if tempo > tempo_medio:
        erros.append("Melhorar o tempo da manobra")
    if class_pred[0] < row['Classificação Final']:
        erros.append("Melhorar a dificuldade da manobra")
    if media_tempos_lucas.get(manobra, 0) < 7:  # Exemplo de critério para notas baixas
        erros.append("Melhorar técnica e consistência")

    # Formatar a mensagem de sugestão
    if len(erros) > 1:
        sugestao_erro = " e ".join(erros)
    elif erros:
        sugestao_erro = erros[0]
    else:
        sugestao_erro = "Não há sugestões específicas."

    sugestoes_melhoria[manobra] = {
        'Sugestao': f"Melhorar a manobra '{manobra}': {sugestao_erro}.",
        'Tempo Para Treino': tempo_ideal,
        'Tempo Médio Anterior': tempo_medio,
        'Classificação Prevista': class_pred[0],
        'Dificuldade': dificuldade_predominante(df, 'Lucas', manobra)
    }

# Sugestões para as manobras que Lucas não executou (treinar)
melhores_manobras = df.groupby(['Manobra'], observed=False).agg({
    'Classificação Final': 'max',
    'Tempo de Execução (s)': 'max'
}).reset_index()

# Filtrar manobras que ninguém obteve nota máxima
melhores_manobras = melhores_manobras[melhores_manobras['Classificação Final'] < 10]

# Verificar as manobras que o Lucas não executou
for index, row in melhores_manobras.iterrows():
    manobra = row['Manobra']
    if manobra not in df_lucas_original['Manobra'].values:  # Verificar se Lucas não executou
        tempo_maximo = row['Tempo de Execução (s)']
        sugestoes_treino[manobra] = {
            'Sugestao': f"Treinar a manobra '{manobra}'",
            'Tempo': tempo_maximo
        }

# Exibir as sugestões de melhoria
print("\n\033[1mSugestões de Melhoria para o Lucas:\033[0m")
for manobra, info in sorted(sugestoes_melhoria.items()):
    print("_____________________________________________________________________________________________________")
    print(f"{info['Sugestao']}")
    print(f"Tempo médio anterior: {info['Tempo Médio Anterior']:.2f} segundos.")
    print(f"Meta de tempo: {info['Tempo Para Treino']:.2f} segundos.")
    print(f"Classificação prevista: {info['Classificação Prevista']}.")
print("\n-----------------------------------------------------------------------------------------------------")

# Exibir as sugestões de Treino
print("\n\033[1mSugestões de Treino:\033[0m")
for manobra, info in sorted(sugestoes_treino.items()):
    print("_____________________________________________________________________________________________________")
    print(f"{info['Sugestao']}")
    print(f"Tempo máximo: {info['Tempo']:.2f} segundos.")
print("_____________________________________________________________________________________________________")