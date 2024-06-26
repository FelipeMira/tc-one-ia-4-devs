import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Alterando as opções de exibição para evitar a truncagem  
pd.set_option('display.max_columns', None)

# Alterando a opção de exibição para a largura máxima das colunas  
pd.set_option('display.width', None)

df_seguros = pd.read_csv("dados_ficticios.csv", low_memory=False)

print('\nDataFrame original')
print(df_seguros.head(10))

# Analisando informações básicas sobre nosso dataframe como tipos de variáveis, tamanho, amostra estatísticas básicas e etc...
print("\nInformações básicas sobre o DataFrame original")
print(df_seguros.describe())

# O método info() nos mostra informações sobre o DataFrame como quantidade de linhas, colunas, tipos de variáveis e quantidade de valores não nulos
print("\nInformações sobre o DataFrame df_seguros:")
print(df_seguros.info())

# O shape nos mostra a quantidade de linhas e colunas do DataFrame
print("\nTamanho do DataFrame original")
print(df_seguros.shape)

# -------------- TRANSFORMANDO OS DADOS --------------
# Transformando as variáveis categóricas em numéricas
print("\nAntes da transformação")
print(df_seguros.head())

print("\nApós a transformação")

# Importando as bibliotecas necessárias
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.compose import make_column_transformer


# Função para aplicar o LabelEncoder
def create_cat_pipeline(df, columns):
    # Função para aplicar o LabelEncoder
    def apply_label_encoder(df):
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':  # Se a coluna for categórica
                df[col] = le.fit_transform(df[col])
            else:  # Se a coluna já for numérica
                df[col] = df[col]
        return df

    # Criando um transformador de função para a função apply_label_encoder
    label_encoder_transformer = FunctionTransformer(apply_label_encoder)

    # Criando a pipeline
    pipeline = make_column_transformer(
        (label_encoder_transformer, columns)
    )

    return pipeline


# Definindo as colunas que você deseja processar
columns_to_process = df_seguros.columns

# Criando a pipeline
cat_pipeline = create_cat_pipeline(df_seguros, columns_to_process)

# Aplicando a pipeline ao DataFrame
df_seguros_transformed = cat_pipeline.fit_transform(df_seguros)

df_seguros_transformed = pd.DataFrame(df_seguros_transformed, columns=df_seguros.columns)

print(df_seguros_transformed.head(10))

# -------------- ANÁLISE DE CORRELAÇÃO --------------

# Criando dataframe com nossas variáveis numericas
df_seguros_numerico = df_seguros_transformed.select_dtypes([np.number])
# Calcula a matriz de correlação
correlation_matrix = df_seguros_numerico.corr()
print("\nMatriz de correlação")
print(correlation_matrix)

# Adicionando um grafico com o mapa de calor da matriz de correlação
import seaborn as sns

# Visualização da matriz de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação')
plt.show()

# -------------- TRATANDO DADOS NULOS --------------

from sklearn.impute import SimpleImputer

# Preenche os valores NaN com a média das colunas
# strategy='mean' preenche os valores faltantes com a média
imputer = SimpleImputer(strategy='mean')
seguros_num = pd.DataFrame(imputer.fit_transform(df_seguros_numerico), columns=df_seguros_numerico.columns)

# Imprimindo as primeiras linhas do DataFrame 'seguros_num' para visualização
print("\nPrimeiras linhas do DataFrame seguros_num após a imputação:")
print(seguros_num.head(10))

# -------------- SEPARANDO OS DADOS --------------
from sklearn.model_selection import train_test_split

X = seguros_num.drop(columns=['Encargos'])  # Variáveis características
y = seguros_num['Encargos']  # O que eu quero prever. (Target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimindo o número de instâncias nos conjuntos de treinamento e teste
print("\nNúmero de instâncias nos conjuntos de treinamento e teste:")
print(len(X_train), "treinamento +", len(X_test), "teste")

# -------------- NORMALIZANDO OS DADOS --------------

# Importando a classe Pipeline do módulo sklearn.pipeline
# Pipeline é uma classe que permite encadear várias etapas de transformação e pré-processamento em uma sequência ordenada
from sklearn.pipeline import Pipeline

# Importando a classe StandardScaler do módulo sklearn.preprocessing
# StandardScaler é uma classe que pode ser usada para padronizar os recursos removendo a média e escalando para a variância da unidade
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Criando uma instância da classe Pipeline
# O argumento é uma lista de tuplas, onde cada tupla representa uma etapa no pipeline
# A primeira tupla representa a primeira etapa, onde a classe SimpleImputer é usada para substituir os valores ausentes pela mediana
# A segunda tupla representa a segunda etapa, onde a classe StandardScaler é usada para padronizar os recursos
num_pipeline = Pipeline([
    ('min_max_scaler', MinMaxScaler()),  # padronizando as escalas dos dados
    ('std_scaler', StandardScaler()),  # padronizando as escalas dos dados
])

# A função fit_transform() do objeto 'num_pipeline' é usada para ajustar o pipeline aos dados e, em seguida, transformar os dados.
# O argumento 'housing_num' é o DataFrame que queremos transformar.
# A função fit_transform() retorna um array numpy onde os valores ausentes foram substituídos pela mediana e os recursos foram padronizados.
# O resultado é armazenado na variável 'housing_num_tr'.
seguros_num_tr = num_pipeline.fit_transform(seguros_num)

# Imprimindo as primeiras linhas do DataFrame 'housing_num_tr' para visualização
print("\nPrimeiras linhas do DataFrame housing_num_tr:")
print(seguros_num_tr[:10])

# -------------- APLICANDO O PCA --------------

from sklearn.decomposition import PCA

pca = PCA()
# Aplicando o PCA ao dataframe sem a coluna 'Encargos'
pca.fit(seguros_num_tr)
variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_)

# Visualização da variância explicada acumulada
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(variancia_cumulativa) + 1), variancia_cumulativa, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Acumulada Explicada')
plt.title('Variância Acumulada Explicada pelo PCA')
plt.show()


def calcular_num_componentes(variancia_desejada, dados):
    pca = PCA()
    pca.fit(dados)
    variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_)
    num_de_pca = np.argmax(variancia_cumulativa >= variancia_desejada) + 1
    return num_de_pca


# Vamos definir um limiar de 80%, ou seja, queremos obter uma porcentagem de explicancia sobre
# nossos dados de igual a 80%
limiar_de_variancia = 0.80

# Calculando o número de componentes para 80% da variância
num_de_pca = calcular_num_componentes(limiar_de_variancia, seguros_num_tr)

print(f"Número de Componentes para {limiar_de_variancia * 100}% da Variância: {num_de_pca}")

#Por fim vamos então utilizar nosso número de PCA desejado e reduzir nossas 7 columns para 6
# Inicializa o objeto PCA
# O parâmetro n_components define o número de componentes principais que desejamos obter
pca = PCA(n_components=num_de_pca)
# Aplica o PCA aos dados padronizados
# O método fit_transform aplica o PCA aos dados e retorna os componentes principais
principal_components = pca.fit_transform(seguros_num_tr)

# Exibe a proporção de variância explicada
# A proporção de variância explicada é a razão entre a variância explicada de um componente e a variância total
explained_variance_ratio = pca.explained_variance_ratio_
print("\nProporção de Variância Explicada")
print(explained_variance_ratio)

# Pegando o número de componentes principais gerados
num_components = principal_components.shape[1]
# Gerando uma lista para cada PCA
column_names = [f'PC{i}' for i in range(1, num_components + 1)]
# Criando um novo dataframe para visualizarmos como ficou nossos dados reduzidos com o PCA
pca_df = pd.DataFrame(data=principal_components, columns=column_names)

print("\nDataFrame reduzido com PCA")
print(pca_df.head(10))

# Criar histogramas para cada coluna
plt.figure(figsize=(15, 8))
for i, col in enumerate(pca_df.columns[:10]):
    plt.subplot(2, 5, i + 1)  # Aqui, ajustei para 2 linhas e 5 colunas
    sns.histplot(pca_df[col], bins=20, kde=True)
    plt.title(f'Histograma {col}')
plt.tight_layout()
plt.show()

# -------------- VERIFICANDO A NORMALIDADE --------------

from scipy.stats import shapiro

# Vamos olhar para cada coluna a normalidade após a redução de dimensionalidade
for column in pca_df.columns:
    stat, p_value = shapiro(pca_df[column])
    print(f'\nVariável: {column}, Estatística de teste: {stat}, Valor p: {p_value}')

    # Você pode então interpretar o valor p para determinar se a variável segue uma distribuição normal
    if p_value > 0.05:
        print(f'A variável {column} parece seguir uma distribuição normal.\n')
    else:
        print(f'A variável {column} não parece seguir uma distribuição normal.\n')

# -------------- APLICANDO UMA PIPELINE --------------

# Obtendo os atributos numéricos do conjunto de dados, excluindo 'Encargos'.
num_attribs = list(df_seguros.select_dtypes(include=[np.number]).columns)
num_attribs.remove('Encargos')  # Removendo 'Encargos' da lista de atributos numéricos.

# Criando uma pipeline para as variáveis numéricas.
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),  #substituindo valores nulos pela mediana
    ('std_scaler', StandardScaler()),  # padronizando as escalas dos dados
])


def apply_pipelines(df):
    # Removendo a coluna 'Encargos' se ela existir
    if 'Encargos' in df.columns:
        df = df.drop(columns=['Encargos'])

    # Aplicando a cat_pipeline com as colunas atuais do DataFrame
    df_cat_transformed = create_cat_pipeline(df, df.columns).fit_transform(df)
    df_cat_transformed = pd.DataFrame(df_cat_transformed, columns=df.columns)

    # Aplicando a num_pipeline
    df_num_transformed = num_pipeline.fit_transform(df_cat_transformed)

    return df_num_transformed


# Criando a pipeline completa
full_pipeline = Pipeline([
    ('apply_pipelines', FunctionTransformer(apply_pipelines)),
])

# Aplicando a pipeline completa ao conjunto de dados para obter os dados preparados para o treinamento do modelo.
seguros_preparado = full_pipeline.fit_transform(df_seguros)

# -------------- TREINANDO O MODELO --------------

from sklearn.tree import DecisionTreeRegressor

# Criando o modelo de DecisionTreeRegressor
# max_depth=10 é o parâmetro que define a profundidade máxima da árvore de decisão
model_dtr = DecisionTreeRegressor(max_depth=10)

# Treinando o modelo com os dados preparados e os rótulos
model_dtr.fit(seguros_preparado, y)

print("\nTreinando o modelo de árvore de decisão...")
print(model_dtr)

# Vamos tentar o pipeline de pré-processamento completo em algumas instâncias de treinamento
# Selecionando as primeiras 5 linhas do DataFrame 'df_seguros' e 'y'
some_data = df_seguros.iloc[:5]
some_labels = y.iloc[:5]

# A função transform() do objeto 'full_pipeline' é usada para transformar os dados.
some_data_prepared = full_pipeline.transform(some_data)

# A função predict() do objeto 'model_dtr' é usada para fazer previsões com o modelo.
predictions = model_dtr.predict(some_data_prepared)

# Imprimindo as previsões para as primeiras 5 instâncias de dados
print("\nPredictions:", model_dtr.predict(some_data_prepared))
print("\nLabels:", list(some_labels))

# -------------- AVALIANDO O MODELO --------------

# Importando a função mean_squared_error do módulo sklearn.metrics
# mean_squared_error é uma função que calcula o erro quadrático médio entre as previsões e os rótulos verdadeiros
from sklearn.metrics import mean_squared_error

# A função mean_squared_error() é usada para calcular o erro quadrático médio entre as previsões e os rótulos verdadeiros.
# O primeiro argumento, 'housing_labels', são os rótulos verdadeiros.
# O segundo argumento, 'housing_predictions', são as previsões feitas pelo modelo.
# A função mean_squared_error() retorna o erro quadrático médio, que é armazenado na variável 'lin_mse'.
lin_mse = mean_squared_error(some_labels, predictions)

# A função np.sqrt() da biblioteca NumPy é usada para calcular a raiz quadrada do erro quadrático médio.
# Isso resulta no erro quadrático médio da raiz (RMSE), que é uma medida comumente usada de quão bem um modelo de regressão se ajusta aos dados.
# O RMSE é armazenado na variável 'lin_rmse'.
lin_rmse = np.sqrt(lin_mse)  # raiz quadrada aqui

# Imprime o valor do RMSE
print("\nErro Quadrático Médio da Raiz (RMSE):")
print(lin_rmse)

from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(some_labels, predictions)

print("\nErro Absoluto Médio (MAE):")
print(lin_mae)

from sklearn.metrics import r2_score

r2 = r2_score(some_labels, predictions)
print("\nCoeficiente de Determinação (R²):")
print('r²', r2)

# -------------- VALIDAÇÃO CRUZADA --------------

import matplotlib.pyplot as plt

# Obtenha as previsões para todo o conjunto de dados
seguros_predictions = model_dtr.predict(seguros_preparado)

# Crie um histograma dos valores reais
plt.hist(y, bins=30, alpha=0.5, label='Valores Reais')

# Crie um histograma dos valores previstos
plt.hist(seguros_predictions, bins=30, alpha=0.5, label='Previsões')

plt.xlabel('Valores')
plt.ylabel('Frequência')
plt.legend(loc='upper right')

plt.show()

# -------------- TESTANDO COM UM DADO NOVO --------------

n = 10  # Número de registros que você deseja gerar
idade_values = np.linspace(18, 65, n)  # Gera n valores de idade entre 18 e 65
imc_values = np.linspace(20, 30, n)  # Gera n valores de IMC entre 20 e 30
fumante_values = ['sim', 'não'] * (n // 2)  # Possíveis valores para fumante
genero_values = ['masculino', 'feminino'] * (n // 2)  # Possíveis valores para fumante

registros = []

for idade, imc, fumante, genero in zip(idade_values, imc_values, fumante_values, genero_values):
    registro = {
        'Idade': int(idade),
        'Gênero': genero,
        'IMC': imc,
        'Filhos': 2,
        'Fumante': fumante,
        'Região': 'sudeste'
    }
    registros.append(registro)

# Convertendo a lista de registros em um DataFrame
novo_registro = pd.DataFrame(registros)

# Passando o novo registro através da pipeline de pré-processamento
novo_registro_preparado = full_pipeline.transform(novo_registro)

# Usando o modelo treinado para fazer uma previsão
nova_previsao = model_dtr.predict(novo_registro_preparado)

# Adicionando os encargos previstos ao DataFrame
novo_registro['Encargos_Previstos'] = nova_previsao

# Imprimindo o DataFrame com os encargos previstos
print(novo_registro)

# Imprimindo a previsão
print("\nPrevisão para os novos registros:", nova_previsao)
