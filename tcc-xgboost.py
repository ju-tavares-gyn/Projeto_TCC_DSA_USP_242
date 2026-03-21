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

#!pip install statsmodels
#!pip install scikit-learn
#!pip install --upgrade statstests

#%% Importando os pacotes
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

from skopt import BayesSearchCV
from skopt.space import Real, Integer

import category_encoders as ce

# from scipy.stats import zscore
import matplotlib.pyplot as plt

from datetime import datetime
from funcoes_ajuda import gerarIndicadores
from funcoes_ajuda import gerarMetricasModelo
from funcoes_ajuda import espec_sens

import statsmodels.api as sm # estimação de modelos
# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
from statstests.process import stepwise # procedimento Stepwise
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos


#%% Carregando os banco de dados antes e pós RedeSim
variaveis = ['CADASTRO_VIA_REDESIM', 'DATA_SOLICITACAO_REGISTRO', 'DATA_HOMOLOGACAO_REGISTRO',  
             'DATA_INICIO_ATIVIDADE', 'DATA_ENCERRAMENTO_ATIVIDADE', 'SITUACAO_CADASTRAL', 
             'MUNICIPIO', 'NATUREZA_JURIDICA', 'ENQUADRAMENTO_EMPRESA', 'TIPO_CONTRIBUINTE',
             'ATIVIDADE_ECONOMICA_DIVISAO', 'QTDE_SOCIOS', 'CAPITAL_SOCIAL']

nomeArquivoAntes = 'dados_antes_redesim.csv' # descompactar o arquivo dados_antes_redesim.7z antes de executar o script
nomeArquivoApos = 'dados_pos_redesim.csv'    # descompactar o arquivo dados_pos_redesim.7z   antes de executar o script

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

#%% Gerar os indicadores de desempenho da integração com a RedeSim
gerarIndicadores(dados_final)

#%% Definir a variável Targert para o encerramento de atividade empresarial
coluna_target = 'ENCERROU_ATIVIDADE'

# Target: Encerrou_Atividade:
#   1 = encerrou atividade
#   0 = não encerrou atividade
dados_final[coluna_target] = np.where(dados_final['SITUACAO_CADASTRAL'] == 'Baixado', 1, np.where((dados_final['SITUACAO_CADASTRAL']== 'Cassado') | (dados_final['SITUACAO_CADASTRAL']== 'Anulado'), 1, 0))

# remover colunas que não serão utilizadas nos modelos preditivos
dados_final.drop(columns=['DATA_SOLICITACAO_REGISTRO', 'DATA_HOMOLOGACAO_REGISTRO'], inplace=True)
dados_final.drop(columns=['DATA_INICIO_ATIVIDADE', 'DATA_ENCERRAMENTO_ATIVIDADE'], inplace=True)

#%% Realizar a transformação de variáveis categóricas

# variáveis categóricas nominais
colunas_categoricas = ['CADASTRO_VIA_REDESIM', 'SITUACAO_CADASTRAL', 'ENQUADRAMENTO_EMPRESA', 'TIPO_CONTRIBUINTE', 'MUNICIPIO', 'NATUREZA_JURIDICA', 'ATIVIDADE_ECONOMICA_DIVISAO']

# Converter as variáveis categóricas para tipo category
for col in colunas_categoricas:
    dados_final[col] = dados_final[col].astype('category')
    
# verificar tipo das variáveis
dados_final.info()

#%% Separar as variáveis features e target 

# semente que serve para reproduzir o modelo no futuro e obter os mesmos resultados(242 = turma USP/Esalq) 
randomState=242

# Obter 10% (0.1) das linhas aleatoriamente
#df_fracao = dados_final.sample(frac=1)

X_features = dados_final.drop(['ENCERROU_ATIVIDADE'], axis=1) 
y_target = dados_final['ENCERROU_ATIVIDADE']

# Aplicar estratégia mista de transformação de variáveis categóricas
colunas_OnHot = ['CADASTRO_VIA_REDESIM', 'SITUACAO_CADASTRAL', 'ENQUADRAMENTO_EMPRESA', 'TIPO_CONTRIBUINTE']
colunas_TargetEncoder = ['MUNICIPIO', 'NATUREZA_JURIDICA', 'ATIVIDADE_ECONOMICA_DIVISAO']

X_features_trasnformada = X_features.copy()

# One-Hot Encoding para features com poucas categorias
X_features_trasnformada = pd.get_dummies(X_features, columns=colunas_OnHot, dtype=int, drop_first=True)

# Dividir a base em 80% para treino e 20% para teste / Separa 20% para o teste final (que o modelo nunca verá no treino nem na validação)
X_treino, X_teste, y_treino, y_teste = train_test_split(X_features_trasnformada, y_target, test_size=0.2, random_state=randomState)

# TargetEncoder para features com muitas categorias - implementação com suavização (smoothing) para evitar overfitting
encoder = ce.TargetEncoder(cols=colunas_TargetEncoder, smoothing=1.0)
X_treino[colunas_TargetEncoder] = encoder.fit_transform(X_treino[colunas_TargetEncoder], y_treino)
X_teste[colunas_TargetEncoder] = encoder.transform(X_teste[colunas_TargetEncoder])


# sempre importante conferir a cada passo
print(X_treino.shape)
print(y_treino.shape)
print(X_teste.shape)
print(y_teste.shape)

# verificar tipo das variáveis
X_features_trasnformada.info()

#%% Definir os parâmetros do GridSearchCV
param_grid = {
    # n_estimators = Número de árvores (geralmente entre 100-1000).
    'n_estimators': [100], #[50, 100, 150, 300],   
    
    # max_depth = Profundidade da árvore (valores baixos reduzem overfitting - comum: 3-10). Com poucas variáveis, não precisa ser muito profundo.
    'max_depth': [3],   #3,4,5,6
    
    # learning_rate = Taxa de aprendizado: menor é melhor, mas exige mais estimadores
    'learning_rate': [0.1], # [0.01, 0.1, 0.2]
    
    # colsample_bytree = Amostragem de colunas: Em umn modelo com 10 variáveis, 0.7 significa usar 7 variáveis por árvore,
    'colsample_bytree': [0.6], # 0.6, 0.7, 0.8
    
    # subsample = Amostragem de observações/linhas: Pode ajudar a reduzir overfitting (0.5 a 1.0).
    'subsample': [0.6], #[0.6, 0.7, 0.8]
    
    # gamma é um parâmetro de regularização que controla a complexidade da árvore ao exigir uma redução mínima da perda para criar novas divisões.
    #       Configurar de 0 a 2 para testar desde nenhuma restrição até uma restrição moderada.
    'gamma': [0], # 0, 1, 2
    # min_child_weight = parâmetro que controla a divisão. Valores mais altos evitam divisões em nós com poucas amostras.
    'min_child_weight': [1]
}

#%% Treinar o modelo com o grid search
from datetime import datetime
# iniciar o cronômetro do tempo de treinamento do modelo
data_inicio = datetime.now()

# instanciar a implementação do XGBoosting Classifier
modelo_xgb = xgb.XGBClassifier(objective='binary:logistic', random_state=randomState)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=randomState)

grid_search = GridSearchCV(estimator=modelo_xgb, param_grid=param_grid, scoring='roc_auc', cv=cv, verbose=0, n_jobs=-1) #error_score='raise'

# treinar o modelo XGBoost com o grid search
# grid_search.fit(X_treino, y_treino)
grid_search.fit(X_treino, y_treino)

# finalizar o cronômetro do tempo de treinamento do modelo
data_fim = datetime.now()

#%% calculando o tempo de treinamento do modelo
tempoTreino = data_fim - data_inicio
dias = tempoTreino.days
horas, resto = divmod(tempoTreino.seconds, 3600)
minutos, segundos = divmod(resto, 60)

print(f"Tempo de execução do modelo XGBoost com GridSearch: {dias * 24 + horas:02}h :{minutos:02}m :{segundos:02}s")

#%% Verificando os melhores parâmetros do modelo
print("Melhores parâmetros com GridSearch:")
print(grid_search.best_params_)

#%% Avaliar o modelo XGBoosting com GridSearch
X_treino['phat']  = grid_search.best_estimator_.predict(X_treino)
X_teste['phat']  = grid_search.best_estimator_.predict(X_teste)

# Imprimir matriz de confusão e curva roc, além dos 
# indicadores com sensitividade, especificidade, acurácia, auc_roc, precision, recall, f1-score
gerarMetricasModelo(observado=y_treino, predicts=X_treino['phat'], base='Treino')
gerarMetricasModelo(observado=y_teste, predicts=X_teste['phat'], base='Teste')

X_treino = X_treino.drop('phat', axis=1)
X_teste = X_teste.drop('phat', axis=1)

#%% Imprimir a importância das variáveis após o treinamento do modelo
# model.feature_importances_ retorna um array com as importâncias
feat_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_treino.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

#%% Treinar o modelo XGBoost com o Otimização Bayesiana

# Divisão dos dados (Treino + Validação para o Early Stopping)
# Dos 80% restantes, separa uma fatia para VALIDAÇÃO (ex: 20% do que sobrou)
X_train, X_val, y_train, y_val = train_test_split(X_treino, y_treino, test_size=0.2, random_state=randomState)

# iniciar o cronômetro do tempo de treinamento do modelo
data_inicio = datetime.now()

# 1. Definição do Espaço de Busca. Fixado os melhores parâmetros estimados entre os intervalos testados
search_spaces = {
    'n_estimators': Integer(135, 136), # 100, 1000
    'max_depth': Integer(7, 8), # 3, 10
    'learning_rate': Real(0.093, 0.094, prior='log-uniform'), # 0.01, 0.3
    'colsample_bytree': Real(0.79, 0.795), # 0.5, 1.0
    'subsample': Real(0.8, 0.83), # 0.5, 1.0
    'gamma': Real(1e-6, 1.7e-05, prior='log-uniform') # 1e-6, 1.0
}

#Melhores parâmetros com otimização Bayesiana: OrderedDict({'colsample_bytree': 0.7950160418010597, 
#                                                           'gamma': 1.723247214111904e-05, 
#                                                           'learning_rate': 0.09429320873621094, 
#                                                           'max_depth': 8, 
#                                                           'n_estimators': 136, 
#                                                           'subsample': 0.8350202858637643})

# 2. Instância do Modelo
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', early_stopping_rounds=20) # Definido na instância para versões recentes


# 3. Configuração da Busca Bayesiana
opt = BayesSearchCV(estimator=xgb_model,
                    search_spaces=search_spaces,
                    n_iter=10,           # Número de combinações a testar 32
                    cv=5,                # Cross-validation
                    n_jobs=-1,           # Paralelização
                    random_state=randomState
)

# 4. Execução com Early Stopping
# Passamos o eval_set dentro do fit_params para o Scikit-Optimize
opt.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)


# finalizar o cronômetro do tempo de treinamento do modelo
data_fim = datetime.now()

tempoTreino = data_fim - data_inicio
dias = tempoTreino.days
horas, resto = divmod(tempoTreino.seconds, 3600)
minutos, segundos = divmod(resto, 60)

print(f"Melhores parâmetros com otimização Bayesiana: {opt.best_params_}")
print(f"Melhor score: {opt.best_score_}")
print(f"Tempo de execução do modelo XGBoost com Otimização Bayesiana: {dias * 24 + horas:02}h :{minutos:02}m :{segundos:02}s")

#%% Avaliar o modelo XGBoosting com Otimização Bayesiana
X_treino['phat']  = opt.best_estimator_.predict(X_treino)
X_teste['phat']  = opt.best_estimator_.predict(X_teste)

# Imprimir matriz de confusão e curva roc, além dos 
# indicadores com sensitividade, especificidade, acurácia, auc_roc, precision, recall, f1-score
gerarMetricasModelo(observado=y_treino, predicts=X_treino['phat'], base='Treino')
gerarMetricasModelo(observado=y_teste, predicts=X_teste['phat'], base='Teste')

X_treino = X_treino.drop('phat', axis=1)
X_teste = X_teste.drop('phat', axis=1)

#%% Imprimir a importância das variáveis após o treinamento do modelo XGBoost com Otimização Bayesiana

# model.feature_importances_ retorna um array com as importâncias
feat_importances = pd.Series(opt.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.nlargest(15).plot(kind='barh')
# feat_importances.plot(kind='barh')
plt.show()

#%% Estimação com modelo logístico binário pela função 'sm.Logit.from_formula'

# Aplicar One-Hot Encoding para features com poucas categorias
X_features_trasnformada = pd.get_dummies(X_features, columns=colunas_OnHot, dtype=int, drop_first=True)

# Aplicar Frequency Encoding para alta cardinalidade
# for col in colunas_TargetEncoder:
#     freq = X_features_trasnformada[col].value_counts(normalize=True)
#     X_features_trasnformada[col] = X_features_trasnformada[col].map(freq)
#     X_features_trasnformada.drop(col, axis=1, inplace=True)
    
# Aplicar TargetEncoder para features com muitas categorias
encoder_log_bin = ce.TargetEncoder(cols=colunas_TargetEncoder, smoothing=1.0)
# X_treino[colunas_TargetEncoder] = encoder_log_bin.fit_transform(X_treino[colunas_TargetEncoder], y_treino)
df_treino_TargetEncoder = encoder_log_bin.fit_transform(X_features[colunas_TargetEncoder], y_target)
X_features_trasnformada[colunas_TargetEncoder] = encoder_log_bin.transform(X_features_trasnformada[colunas_TargetEncoder])

# Ajustar nome das variáveis - remover espaços e substituir '_'. Substituir barra por '_'
X_features_trasnformada.columns = X_features_trasnformada.columns.str.strip().str.replace(' ', '_')
X_features_trasnformada.columns = X_features_trasnformada.columns.str.replace('/', '_')

X_features_trasnformada = pd.concat([X_features_trasnformada, y_target], axis=1)

# Tabela de frequências absolutas da variável dependente 'ENCERROU_ATIVIDADE'
X_features_trasnformada['ENCERROU_ATIVIDADE'].value_counts().sort_index()

# Definição da fórmula utilizada no modelo
# lista_colunas_lb = list(X_features_trasnformada.drop(columns='ENCERROU_ATIVIDADE').columns)
# formula_modelo_lb = ' + '.join(lista_colunas_lb)
# formula_modelo_lb = "ENCERROU_ATIVIDADE ~ " + formula_modelo_lb

# ENCERROU_ATIVIDADE ~ QTDE_SOCIOS + CAPITAL_SOCIAL + TempoAtividadeEmpresarial + CADASTRO_VIA_REDESIM_S + SITUACAO_CADASTRAL_Ativo + SITUACAO_CADASTRAL_Baixado + SITUACAO_CADASTRAL_Cassado + SITUACAO_CADASTRAL_Paralisado + SITUACAO_CADASTRAL_Suspenso + ENQUADRAMENTO_EMPRESA_Microempresa + ENQUADRAMENTO_EMPRESA_Normal + ENQUADRAMENTO_EMPRESA_Simples_Nacional_SIMEI + TIPO_CONTRIBUINTE_COMERCIANTE_ATACADISTA + TIPO_CONTRIBUINTE_COMERCIANTE_VAREJISTA + TIPO_CONTRIBUINTE_EXTRATOR_MINERAL_OU_FÓSSIL + TIPO_CONTRIBUINTE_INDUSTRIAL + TIPO_CONTRIBUINTE_OUTRO_PRESTADOR_DE_SERVIÇO + TIPO_CONTRIBUINTE_PRESTADOR_DE_SERVIÇO_DE_COMUNICAÇÃO + TIPO_CONTRIBUINTE_PRODUTOR_RURAL + TIPO_CONTRIBUINTE_PRODUTOR_URBANO + TIPO_CONTRIBUINTE_TRANSPORTADOR + MUNICIPIO + NATUREZA_JURIDICA + ATIVIDADE_ECONOMICA_DIVISAO
formula = 'ENCERROU_ATIVIDADE ~ SITUACAO_CADASTRAL_Baixado + SITUACAO_CADASTRAL_Cassado + SITUACAO_CADASTRAL_Ativo + SITUACAO_CADASTRAL_Suspenso + ENQUADRAMENTO_EMPRESA_Microempresa + ENQUADRAMENTO_EMPRESA_Normal + ENQUADRAMENTO_EMPRESA_Simples_Nacional_SIMEI + TIPO_CONTRIBUINTE_COMERCIANTE_ATACADISTA + TIPO_CONTRIBUINTE_COMERCIANTE_VAREJISTA + TIPO_CONTRIBUINTE_EXTRATOR_MINERAL_OU_FÓSSIL + TIPO_CONTRIBUINTE_INDUSTRIAL + TIPO_CONTRIBUINTE_OUTRO_PRESTADOR_DE_SERVIÇO + TIPO_CONTRIBUINTE_PRESTADOR_DE_SERVIÇO_DE_COMUNICAÇÃO + TIPO_CONTRIBUINTE_PRODUTOR_RURAL + TIPO_CONTRIBUINTE_PRODUTOR_URBANO + TIPO_CONTRIBUINTE_TRANSPORTADOR + MUNICIPIO + NATUREZA_JURIDICA + ATIVIDADE_ECONOMICA_DIVISAO + TempoAtividadeEmpresarial + QTDE_SOCIOS' # CADASTRO_VIA_REDESIM_S + SITUACAO_CADASTRAL_Paralisado + CAPITAL_SOCIAL 
formula = 'ENCERROU_ATIVIDADE ~ SITUACAO_CADASTRAL_Ativo + SITUACAO_CADASTRAL_Baixado + SITUACAO_CADASTRAL_Suspenso + ENQUADRAMENTO_EMPRESA_Normal + MUNICIPIO + NATUREZA_JURIDICA + TempoAtividadeEmpresarial + QTDE_SOCIOS'

modelo_logistico_bin = sm.Logit.from_formula(formula, X_features_trasnformada).fit()

# Parâmetros do modelo
modelo_logistico_bin.summary()

summary_col([modelo_logistico_bin],
            model_names=["MODELO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        })

#%% Estimação do modelo por meio do procedimento Stepwise
step_modelo_logistico_bin = stepwise(modelo_logistico_bin, pvalue_limit=0.05)
step_modelo_logistico_bin.summary()

summary_col([step_modelo_logistico_bin],
            model_names=["MODELO FINAL"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        })

#%% Adicionando os valores previstos de probabilidade na base de dados
X_features_trasnformada['phat'] = step_modelo_logistico_bin.predict()

# Matriz de confusão para cutoff = 0.5
# Imprimir matriz de confusão e curva roc, além dos 
# indicadores com sensitividade, especificidade, acurácia, auc_roc, precision, recall, f1-score
gerarMetricasModelo(observado=X_features_trasnformada['ENCERROU_ATIVIDADE'],
                    predicts=X_features_trasnformada['phat'], 
                    cutoff=0.5,
                    base='Treino')

# Matriz de confusão para cutoff = 0.3
gerarMetricasModelo(observado=X_features_trasnformada['ENCERROU_ATIVIDADE'],
                    predicts=X_features_trasnformada['phat'], 
                    cutoff=0.3,
                    base='Treino')

# Matriz de confusão para cutoff = 0.7
gerarMetricasModelo(observado=X_features_trasnformada['ENCERROU_ATIVIDADE'],
                    predicts=X_features_trasnformada['phat'], 
                    cutoff=0.7,
                    base='Treino')

X_features_trasnformada = X_features_trasnformada.drop('phat', axis=1)

#%% Plotagem de um gráfico que mostra a variação da sensitividade e da especificidade em função do cutoff
dados_plotagem = espec_sens(observado = X_features_trasnformada['ENCERROU_ATIVIDADE'],
                            predicts = X_features_trasnformada['phat'])

plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.sensitividade, marker='o',
         color='indigo', markersize=8)
    plt.plot(dados_plotagem.cutoffs,dados_plotagem.especificidade, marker='o',
         color='darkorange', markersize=8)
plt.xlabel('Cuttoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.show()
