# Análise de Dados de Seguros

Este script Python realiza várias etapas de pré-processamento e modelagem de dados para um conjunto de dados de seguros. As etapas incluem:

1. **Carregamento e visualização inicial dos dados**: O script começa carregando os dados de um arquivo CSV e exibindo as primeiras linhas do DataFrame.

2. **Transformação de variáveis categóricas em numéricas**: As variáveis categóricas 'Gênero', 'Fumante' e 'Região' são transformadas em numéricas usando `LabelEncoder`.

3. **Análise de correlação entre as variáveis**: Uma matriz de correlação é calculada e visualizada como um mapa de calor.

4. **Tratamento de dados nulos**: Os valores nulos são preenchidos com a média das colunas usando `SimpleImputer`.

5. **Separação dos dados em conjuntos de treinamento e teste**: Os dados são divididos em um conjunto de treinamento e um conjunto de teste.

6. **Normalização dos dados**: Os dados são normalizados usando `StandardScaler` e `MinMaxScaler`.

7. **Aplicação do PCA para redução de dimensionalidade**: O PCA é aplicado para reduzir a dimensionalidade dos dados.

8. **Verificação da normalidade dos dados**: A normalidade dos dados é verificada usando o teste de Shapiro-Wilk.

9. **Aplicação de um pipeline para automatizar as etapas de pré-processamento**: Um pipeline é criado para automatizar as etapas de pré-processamento.

10. **Treinamento de um modelo de regressão de árvore de decisão**: Um modelo de árvore de decisão é treinado com os dados preparados.

11. **Avaliação do modelo**: O modelo é avaliado usando RMSE, MAE e R².

12. **Teste do modelo com novos dados**: O modelo é testado com novos dados.

Cada etapa é realizada em uma seção separada do código, e cada seção é comentada para explicar o que está acontecendo. Além disso, o código usa várias bibliotecas Python para ciência de dados e aprendizado de máquina, incluindo pandas, numpy, matplotlib, seaborn, sklearn e scipy.