# Importando as bibliotecas necessárias
from faker import Faker
from decimal import Decimal, getcontext
import random
import pandas as pd
import numpy as np

# Crie uma instância do Faker para dados em português do Brasil
fake = Faker('pt_BR')

# Definindo o número de registros fictícios a serem gerados
num_registros = 50000

# Crie listas vazias para cada campo
idades, generos, imcs, filhos, fumantes, regioes, encargos = [], [], [], [], [], [], []

# Defina a precisão desejada
getcontext().prec = 17

def calcular_encargos(idade, imc, fumante):
    # Definindo os coeficientes para a idade e o IMC
    coef_idade = 200
    coef_imc = 500

    # Calculando o encargo baseado na idade e no IMC
    encargo = idade * coef_idade + imc * coef_imc

    # Adicionando um valor fixo ao encargo se a pessoa for fumante
    if fumante == 'sim':
        encargo += 10000

    return Decimal(str(encargo)).quantize(Decimal('0.00'))

# Gerando os dados fictícios
for _ in range(num_registros):
    # Gera uma idade aleatória entre 18 e 99 com mais peso para idades entre 18 e 65
    probabilidades = [0.012 if i < 66 else 0.004 for i in range(18, 99)]
    total = sum(probabilidades)
    probabilidades_normalizadas = [p/total for p in probabilidades]
    idade = np.random.choice(range(18, 99), p=probabilidades_normalizadas)
    idades.append(idade)

    # Gera um gênero aleatório (masculino ou feminino) e adiciona à lista de gêneros
    generos.append(fake.random_element(['masculino', 'feminino']))

    # Gera um IMC (Índice de Massa Corporal) aleatório com precisão de 17 dígitos decimais e adiciona à lista de IMCs
    imc = Decimal(str(random.randint(20, 40))) + Decimal(str(random.random())).quantize(Decimal('0.000000000000000'))
    imcs.append(imc)

    # Gera um número aleatório de filhos (entre 0 e 5) e adiciona à lista de filhos
    filhos.append(fake.random_int(min=0, max=5))

    # Gera um status de fumante aleatório (sim ou não) e adiciona à lista de fumantes
    fumante = fake.random_element(['sim', 'não'])
    fumantes.append(fumante)

    # Gera uma região aleatória (norte, nordeste, sudeste, sul, centro-oeste) e adiciona à lista de regiões
    regioes.append(fake.random_element(['norte', 'nordeste', 'sudeste', 'sul', 'centro-oeste']))

    # Calcula o encargo e adiciona à lista de encargos
    encargo = calcular_encargos(idade, imc, fumante)
    encargos.append(encargo)

# Crie um DataFrame com os dados gerados
df = pd.DataFrame({
    'Idade': idades,
    'Gênero': generos,
    'IMC': imcs,
    'Filhos': filhos,
    'Fumante': fumantes,
    'Região': regioes,
    'Encargos': encargos
})

# Crie uma lista com os novos dados
novos_dados = [
    [56, 'feminino', 29.774373714007336, 2, 'sim', 'sudoeste', 31109.889763423336],
    [46, 'masculino', 25.857394655216346, 1, 'não', 'nordeste', 26650.702646642694],
    [32, 'masculino', 23.014839993647488, 0, 'não', 'sudoeste', 21459.03799039332]
]

# Crie um DataFrame com os novos dados
df_novos_dados = pd.DataFrame(novos_dados, columns=df.columns)

# Concatene os novos dados com o DataFrame existente
df = pd.concat([df_novos_dados, df]).reset_index(drop=True)

# Defina a proporção de valores vazios que você deseja introduzir
proporcao_vazios = 0.02  # 5% dos valores serão vazios

# Para cada coluna no DataFrame, selecione uma amostra aleatória de índices e defina o valor para vazio
for coluna in df.columns:
    num_vazios = int(proporcao_vazios * num_registros)
    indices_vazios = df[coluna].sample(num_vazios).index
    df.loc[indices_vazios, coluna] = np.nan

# Salve o DataFrame em um arquivo CSV com codificação UTF-8-SIG
# A codificação 'utf-8-sig' adiciona uma assinatura BOM (Byte Order Mark) ao início do arquivo,
# que pode ajudar alguns programas a interpretar corretamente a codificação UTF-8.
df.to_csv('dados_ficticios.csv', index=False, encoding='utf-8-sig')

print("Dados fictícios gerados e salvos no arquivo 'dados_ficticios.csv'")