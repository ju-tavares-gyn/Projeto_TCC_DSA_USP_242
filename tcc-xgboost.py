# -*- coding: utf-8 -*-

"""
MBA Data Science e Analytics USP/ESALQ
@author: Juliano Tavares
github: https://github.com/ju-tavares-gyn/Projeto_TCC_DSA_USP_242.git
"""

#%% Instalar os pacotes necessários

#!pip install pandas
#!pip install numpy
#!pip install matplotlib
#!pip install scikit-learn
#!pip install xgboost
#!pip install category_encoders

#!pip install scipy # Lib serve para padronização de variáveis métricas Z SCORE
#!pip install python-dateutil
#!pip install scikit-optimize

#%% Importando os pacotes

import pandas as pd
import numpy as np

from xgboost import XGBClassifier
# import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

import category_encoders as ce

from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from funcoes_ajuda import avaliaPredicao #, descritiva, encode_strategy,

#from scipy import stats # Lib serve para padronização de variáveis métricas Z SCORE

#%% Carregando os banco de dados antes e pós RedeSim
variaveis = ['CADASTRO_VIA_REDESIM', 'DATA_SOLICITACAO_REGISTRO', 'DATA_HOMOLOGACAO_REGISTRO',  
             'DATA_INICIO_ATIVIDADE', 'DATA_ENCERRAMENTO_ATIVIDADE', 'SITUACAO_CADASTRAL', 
             'MUNICIPIO', 'NATUREZA_JURIDICA', 'ENQUADRAMENTO_EMPRESA', 'TIPO_CONTRIBUINTE',
             'ATIVIDADE_ECONOMICA_DIVISAO', 'QTDE_SOCIOS', 'CAPITAL_SOCIAL']

nomeArquivoAntes = 'dados_antes_redesim.csv'
nomeArquivoApos = 'dados_pos_redesim.csv'

# Mesclar os dados em uma única base
dados_final = pd.concat([pd.read_csv(nomeArquivoAntes, sep='¬', encoding = 'latin-1', usecols=variaveis), 
                        pd.read_csv(nomeArquivoApos, sep='¬', encoding = 'latin-1', usecols=variaveis)], ignore_index=True)

# verificar informações detalhadas das variáveis
dados_final.info()

#%% Tratamento de valores faltantes
print(dados_final.isnull().sum())

# formato de datas
formatoDataTime = "%d/%m/%Y %H:%M:%S"
formatoData = "%d/%m/%Y"

dados_final['DATA_SOLICITACAO_REGISTRO'] = pd.to_datetime(dados_final['DATA_SOLICITACAO_REGISTRO'], format=formatoDataTime)
dados_final['DATA_HOMOLOGACAO_REGISTRO'] = pd.to_datetime(dados_final['DATA_HOMOLOGACAO_REGISTRO'], format=formatoDataTime)
dados_final['DataHomologacaoAno'] = dados_final['DATA_HOMOLOGACAO_REGISTRO'].dt.year

# tratar a data de inicio de atividade que estiver nula será preenchida com a data de homologação do registo de abertura (79 observações).
dados_final['DATA_INICIO_ATIVIDADE'] = dados_final['DATA_INICIO_ATIVIDADE'].fillna(dados_final['DATA_HOMOLOGACAO_REGISTRO'].dt.date)
dados_final['DATA_INICIO_ATIVIDADE'] = pd.to_datetime(dados_final['DATA_INICIO_ATIVIDADE'], format=formatoData)

# tratar o capital social que estiver sem valor será atribuibo zero, porque a maioria pertence a natureza juridica não ligada a atividade de interesse da receita estadual
# dados_capital_nulo = dados_final[dados_final['CAPITAL_SOCIAL'].isnull()].copy()
# dados_capital_nulo['NATUREZA_JURIDICA'].value_counts()

dados_final['CAPITAL_SOCIAL'] = dados_final['CAPITAL_SOCIAL'].fillna(0)
dados_final['CAPITAL_SOCIAL'] = pd.to_numeric(dados_final['CAPITAL_SOCIAL'], errors='coerce')

# tratar a data de encerramento de atividade que estiver sem valor será atribuibo valor pd.NaT (informar valor ausente ao pandas ).
dados_final['DATA_ENCERRAMENTO_ATIVIDADE'] = dados_final['DATA_ENCERRAMENTO_ATIVIDADE'].fillna(pd.NaT)
dados_final['DATA_ENCERRAMENTO_ATIVIDADE'] = pd.to_datetime(dados_final['DATA_ENCERRAMENTO_ATIVIDADE'], format=formatoDataTime)

# Calcular a Duração de Atividade Empresarial em dias para ser usada na predição dos modelos
dataAtual = datetime.now()

dados_final['TempoAtividadeEmpresarial'] = np.where(dados_final['DATA_ENCERRAMENTO_ATIVIDADE'].notna(), (dados_final['DATA_ENCERRAMENTO_ATIVIDADE'] - dados_final['DATA_HOMOLOGACAO_REGISTRO']).dt.days,
                                           np.where(dados_final['DATA_ENCERRAMENTO_ATIVIDADE'].isna(), (dataAtual - dados_final['DATA_HOMOLOGACAO_REGISTRO']).dt.days, np.nan))

## nova verificação de dados faltantes
print(dados_final.isnull().sum())

#%% Calcular indicadores de desempenho da integração com a RedeSim Agrupar por indicador CADASTRO_VIA_REDESIM (S=SIM ou N=Não)

# calcular o Tempo Médio de Abertura de Empresa/Contribuintes, antes e depois da RedeSim.
dados_final['tempo_abertura'] = dados_final['DATA_HOMOLOGACAO_REGISTRO'] - dados_final['DATA_SOLICITACAO_REGISTRO']

media_abertura = dados_final.groupby('CADASTRO_VIA_REDESIM')['tempo_abertura'].mean().reset_index(name='tempo')
tempo1 = media_abertura.loc[media_abertura['CADASTRO_VIA_REDESIM'] == 'N', 'tempo']
tempo2 = media_abertura.loc[media_abertura['CADASTRO_VIA_REDESIM'] == 'S', 'tempo']

tempo1Formatado = (
    tempo1.dt.components['days'].astype(str) + 'd ' +
    tempo1.dt.components['hours'].astype(str) + 'h ' +
    tempo1.dt.components['minutes'].astype(str) + 'm'
)

tempo2Formatado = (
    tempo2.dt.components['days'].astype(str) + 'd ' +
    tempo2.dt.components['hours'].astype(str) + 'h ' +
    tempo2.dt.components['minutes'].astype(str) + 'm'
)

print(f'Tempo médio de abertura de empresa/contribuinte antes da RedeSim = {tempo1Formatado}')
print(f'Tempo médio de abertura de empresa/contribuinte após a RedeSim   = {tempo2Formatado}')


## Calcular o Tempo Médio de Sobrevivência da Empresa/Contribuintes, antes e depois da RedeSim.
dados_final['tempo_sobrevivencia_meses'] = np.where(dados_final['DATA_ENCERRAMENTO_ATIVIDADE'].notna(), 
                                          (dados_final['DATA_ENCERRAMENTO_ATIVIDADE'] - dados_final['DATA_INICIO_ATIVIDADE']).dt.days / 30, # Cálculo
                                           np.nan)           # Valor se for nulo

media_sobrevivencia = dados_final.groupby('CADASTRO_VIA_REDESIM')['tempo_sobrevivencia_meses'].mean().reset_index(name='tempo')
tempo1 = int(media_sobrevivencia.loc[media_abertura['CADASTRO_VIA_REDESIM'] == 'N', 'tempo'])
tempo2 = int(media_sobrevivencia.loc[media_abertura['CADASTRO_VIA_REDESIM'] == 'S', 'tempo'])
print(f"Tempo médio de sobrevivência até o encerramento da atividade empresarial (antes RedeSim) = {tempo1} meses")
print(f"Tempo médio de sobrevivência até o encerramento da atividade empresarial (após RedeSim)  = {tempo2}  meses")


# Calcular a quantidade de abertura de empresa: antes e após a integração com a RedeSim
# 1.Agrupar por ano e indicador de cadastro via RedeSim
dfAgrupamento = dados_final.groupby(['DataHomologacaoAno', 'CADASTRO_VIA_REDESIM']).size() #.unstack().fillna(0)

# 2. Transformar os valores (categotrias) da variável 'CADASTRO_VIA_REDESIM' em colunas
# Matplotlib precisa que as categorias estejam em colunas separadas para criar barras agrupadas ou múltiplas linhas.
dfAgrupamento = dfAgrupamento.unstack().fillna(0)
print(dfAgrupamento)

# 3. Gerar o gráfico de barras (atribuindo a um objeto 'ax')
ax = dfAgrupamento.plot(kind='bar', figsize=(10, 5), width=0.8)

# 4. Adicionar os valores em cima de cada barra
# O Matplotlib armazena as barras em 'ax.containers'
for container in ax.containers:
    ax.bar_label(container, padding=2)

# 4. Ajustes finais
plt.title('Quantitativo de abertura de empresas') # N = Antes RedeSim, S = Após RedeSim
plt.xlabel('Ano')
plt.ylabel('Quantidade')
plt.legend(title='Categorias', bbox_to_anchor=(1, 1))

# Definir manualmente os nomes na legenda
ax.legend(['Antes RedeSim', 'Após RedeSim'], title='Categorias')

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Calcular a Distribuição de Empresas por Município e Ano
# agrupar os dados: Contar empresas por ano e município
# Usamos reset_index() para transformar o agrupamento em colunas utilizáveis
dfAgrupamento = dados_final.groupby(['DataHomologacaoAno', 'MUNICIPIO']).size().reset_index(name='quantidade')


# Ordenar por ano (crescente) e quantidade (decrescente)
df_ordenado = dfAgrupamento.sort_values(by=['DataHomologacaoAno', 'quantidade'], ascending=[True, False])

# Agrupar por ano e obter os Top 5 municipios que abriram mais empresas em cada ano
top_municipio_qtde = 5
df_top5_por_ano = df_ordenado.groupby('DataHomologacaoAno').head(top_municipio_qtde)

# Visualização com Seaborn
plt.figure(figsize=(15, 7))
ax = sns.lineplot(data=df_top5_por_ano, x='DataHomologacaoAno', y='quantidade', hue='MUNICIPIO', marker='o', linewidth=1.0)  # Esquema de cores elegante
plt.title('Quantitativo de Abertura de Empresas por Município e Ano', fontsize=12, fontweight= 'bold')
plt.xlabel('', fontsize=12)
plt.ylabel('', fontsize=12)
plt.grid(True, linestyle='--')
# Captura os handles (linhas) e labels (nomes) gerados
# ax = sns.barplot(data=df_top5_por_ano, x='DataHomologacaoAno', y='quantidade', hue='MUNICIPIO')
# handles, labels = ax.get_legend_handles_labels()
# plt.legend(handles[0:top_municipio_qtde], labels[0:top_municipio_qtde], title='Município', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Município', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# remover do dataframe colunas temporarias utilizadas para calcular os indicadores
dados_final.drop(columns=['DataHomologacaoAno','tempo_abertura', 'tempo_sobrevivencia_meses'], inplace=True)


#%% Definir a variável Targert para o encerramento de atividade empresarial
coluna_target = 'ENCERROU_ATIVIDADE'

# Target: Encerrou_Atividade:
#   1 = encerrou atividade
#   0 = não encerrou atividade
dados_final[coluna_target] = np.where(dados_final['SITUACAO_CADASTRAL'] == 'Baixado', 1, np.where((dados_final['SITUACAO_CADASTRAL']== 'Cassado') | (dados_final['SITUACAO_CADASTRAL']== 'Anulado'), 1, 0))

# remover colunas que não serão utilizadas nos modelos preditivos
dados_final.drop(columns=['DATA_SOLICITACAO_REGISTRO', 'DATA_HOMOLOGACAO_REGISTRO'], inplace=True)
dados_final.drop(columns=['DATA_INICIO_ATIVIDADE', 'DATA_ENCERRAMENTO_ATIVIDADE'], inplace=True)
# dados_final.drop(columns=['SITUACAO_CADASTRAL'], inplace=True)

#%% Realizar a transformação de variáveis categóricas

# variáveis categóricas nominais
colunas_categoricas = ['CADASTRO_VIA_REDESIM', 'SITUACAO_CADASTRAL', 'ENQUADRAMENTO_EMPRESA', 'TIPO_CONTRIBUINTE', 'MUNICIPIO', 'NATUREZA_JURIDICA', 'ATIVIDADE_ECONOMICA_DIVISAO']

# Converter as variáveis categóricas para tipo category
for col in colunas_categoricas:
    dados_final[col] = dados_final[col].astype('category')
    
# tratar variáveis tipo data
# dados_final['DATA_INICIO_ATIVIDADE'] = dados_final['DATA_INICIO_ATIVIDADE'].dt.strftime('%Y%m%d').astype(int)
# dados_final['DATA_ENCERRAMENTO_ATIVIDADE'] = dados_final['DATA_ENCERRAMENTO_ATIVIDADE'].dt.strftime('%Y%m%d').astype(float)

dados_final.info()

#%% Separar as variáveis features e target 

# semente que serve para reproduzir o modelo no futuro e obter os mesmos resultados(2360873  = famoso número de telefone do Bozo) 
randomState=2360873

# Obter 10% (0.1) das linhas aleatoriamente
df_fracao = dados_final.sample(frac=1)

X_features = df_fracao.drop(['ENCERROU_ATIVIDADE'], axis=1) 
y_target = df_fracao['ENCERROU_ATIVIDADE']

# Dividir a base em 80% para treino e 20% para teste / Separa 20% para o teste final (que o modelo nunca verá no treino nem na validação)
X_treino, X_teste, y_treino, y_teste = train_test_split(X_features, y_target, test_size=0.2, random_state=randomState)

# sempre importante conferir a cada passo
print(X_treino.shape)
print(y_treino.shape)
print(X_teste.shape)
print(y_teste.shape)

# Aplicar estratégia mista de transformação de variáveis categóricas

# from sklearn.compose import make_column_transformer
# from sklearn.preprocessing import OneHotEncoder

X_treino_encoded = X_treino.copy()
X_teste_encoded = X_teste.copy()

colunas_OnHot = ['CADASTRO_VIA_REDESIM', 'SITUACAO_CADASTRAL', 'ENQUADRAMENTO_EMPRESA', 'TIPO_CONTRIBUINTE']
colunas_TargetEncoder = ['MUNICIPIO', 'NATUREZA_JURIDICA', 'ATIVIDADE_ECONOMICA_DIVISAO']
 
# One-Hot Encoding para poucas categorias
X_treino_encoded = pd.get_dummies(X_treino, columns=colunas_OnHot, drop_first=True) # dtype=int
X_teste_encoded = pd.get_dummies(X_teste, columns=colunas_OnHot, drop_first=True) 

# Implementação com Suavização (Smoothing) para evitar overfitting
encoder = ce.TargetEncoder(cols=colunas_TargetEncoder, smoothing=1.0)
X_treino_encoded[colunas_TargetEncoder] = encoder.fit_transform(X_treino[colunas_TargetEncoder], y_treino)
X_teste_encoded[colunas_TargetEncoder] = encoder.transform(X_teste[colunas_TargetEncoder])
        
    # else:
    #     # Frequency Encoding para alta cardinalidade
    #     freq = X_treino[col].value_counts(normalize=True)
    #     X_treino_encoded[f'{col}_freq'] = X_treino[col].map(freq)
    #     X_treino_encoded.drop(col, axis=1, inplace=True)

#%% Definir os parâmetros do GridSearchCV
param_grid = {
    # n_estimators = Número de árvores (geralmente entre 100-1000).
    'n_estimators': [50, 100, 200],   
    
    # max_depth = Profundidade da árvore (valores baixos reduzem overfitting - comum: 3-10). Com poucas variáveis, não precisa ser muito profundo.
    'max_depth': [2, 3, 4, 5],   
    
    # learning_rate = Taxa de aprendizado: menor é melhor, mas exige mais estimadores
    'learning_rate': [0.01, 0.1], # [0.01, 0.1, 0.2]
    
    # colsample_bytree = Amostragem de colunas: Em umn modelo com 10 variáveis, 0.7 significa usar 7 variáveis por árvore,
    'colsample_bytree': [0.6, 0.8],    
    
    # subsample = Amostragem de observações/linhas: Pode ajudar a reduzir overfitting (0.5 a 1.0).
    'subsample': [0.6, 0.8], #[0.7, 0.8, 1]
    
    # gamma é um parâmetro de regularização que controla a complexidade da árvore ao exigir uma redução mínima da perda para criar novas divisões.
    #       Configurar de 0 a 2 para testar desde nenhuma restrição até uma restrição moderada.
    'gamma': [0, 1],    
    # min_child_weight = parâmetro que controla a divisão. Valores mais altos evitam divisões em nós com poucas amostras.
    'min_child_weight': [1]
}

#%% Treinar o modelo com o grid search
import time
from datetime import datetime
# iniciar o cronômetro do tempo de treinamento do modelo
data_inicio = datetime.now()

# instanciar a implementação do XGBoosting Classifier
modelo = XGBClassifier(objective='binary:logistic', random_state=randomState)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)

grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, scoring='roc_auc', cv=cv, verbose=0, n_jobs=-1) #error_score='raise'

# treinar o modelo XGBoost com o grid search
# grid_search.fit(X_treino, y_treino)
grid_search.fit(X_treino_encoded, y_treino)

# finalizar o cronômetro do tempo de treinamento do modelo
data_fim = datetime.now()

#%% calculando o tempo de treinamento do modelo
tempoTreino = data_fim - data_inicio
dias = tempoTreino.days
horas, resto = divmod(tempoTreino.seconds, 3600)
minutos, segundos = divmod(resto, 60)

print(f"Tempo de execução do modelo XGBoost: {dias * 24 + horas:02}h :{minutos:02}m :{segundos:02}s")

#%% Verificando os melhores parâmetros do modelo

print("Melhores parâmetros do grid_search:")
print(grid_search.best_params_)

#%% Avaliar o modelo XGBoosting com GridSearch
avaliaPredicao(grid_search.best_estimator_, X_treino_encoded, y_treino, X_teste_encoded, y_teste)

#%% Treinar o modelo XGBoost com o Otimização Baysiana
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Divisão dos dados (Treino + Validação para o Early Stopping)
# Dos 80% restantes, separa uma fatia para VALIDAÇÃO (ex: 20% do que sobrou)
X_train, X_val, y_train, y_val = train_test_split(X_treino_encoded, y_treino, test_size=0.2, random_state=randomState)

# iniciar o cronômetro do tempo de treinamento do modelo
data_inicio = datetime.now()

# 1. Definição do Espaço de Busca
search_spaces = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 10),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'gamma': Real(1e-6, 1.0, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0)
}

# 2. Instância do Modelo
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=20) # Definido na instância para versões recentes


# 3. Configuração da Busca Bayesiana
opt = BayesSearchCV(estimator=xgb_model,
                    search_spaces=search_spaces,
                    n_iter=32,           # Número de combinações a testar
                    cv=5,                # Cross-validation
                    n_jobs=-1,           # Paralelização
                    random_state=randomState
)

# 4. Execução com Early Stopping
# Passamos o eval_set dentro do fit_params para o Scikit-Optimize
opt.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

print(f"Melhores parâmetros: {opt.best_params_}")
print(f"Melhor score: {opt.best_score_}")

# finalizar o cronômetro do tempo de treinamento do modelo
data_fim = datetime.now()

tempoTreino = data_fim - data_inicio
dias = tempoTreino.days
horas, resto = divmod(tempoTreino.seconds, 3600)
minutos, segundos = divmod(resto, 60)

print(f"Tempo de execução do modelo XGBoost com Otimização Bayesiana: {dias * 24 + horas:02}h :{minutos:02}m :{segundos:02}s")

print(f"Melhores parâmetros: {opt.best_params_}")
print(f"Melhor score: {opt.best_score_}")

#%% 
def treinarModelo(parametros, X_treino, y_treino, X_teste, y_teste):
    
    # learning_rate = 
    learning_rate = parametros[0] # Taxa de aprendizado: menor é melhor, mas exige mais estimadores  [0.01, 0.1, 0.2]
    num_leaves = parametros[1] # número máximo de folhas de cada árvore no último nó    
    min_child_samples =  parametros[2] # quantidade de observações em cada nó    
    subsample = parametros[3] # Amostragem de observações/linhas: Pode ajudar a reduzir overfitting (0.5 a 1.0).
    colsample_bytree = parametros[4] # Amostragem de colunas: Em um modelo com 10 variáveis, 0.7 significa usar 7 variáveis por árvore
    
    #'n_estimators': [50, 100] # n_estimators = Número de árvores (geralmente entre 100-1000).
        
    
    modelo = XGBClassifier()    
    modelo.fit(X_treino, y_treino)
    p_teste = modelo.predict_proba(X_teste)[:,1]
    return -roc_auc_score(y_teste, p_teste)
    