# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 21:03:50 2024
@author: João Mello

Modificado em 12/02/2026
@author: Juliano Tavares
"""

#%%  funções de ajuda

import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve
    
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from dateutil.relativedelta import relativedelta

## Faz a transformação das features categóricas usando a Estratégia Baseada na Cardinalidade:
## - One-Hot Encoding para cardinalidade até 10 categorias
## - Target Encoding para cardinalidade média entre 10 e 50 categorias
##    Boas Práticas Target Encoding:
##    Use apenas no Treino: Calcule as médias apenas no conjunto de treinamento e aplique no de teste (use fit_transform no treino e transform no teste).
##    Cross-Validation: Utilize técnicas de validação cruzada para garantir que o encodificação seja robusta.
## - Frequency Encoding para alta cardinalidade maiores que 50 categorias
def encode_strategy(df, categorical_cols, dftarget, is_treino):
    """
    Estratégia mista baseada na cardinalidade
    """
    df_encoded = df.copy()
    
    for col in categorical_cols:
        n_categories = df[col].nunique()
        
        if n_categories <= 10:
            # One-Hot Encoding para poucas categorias
            df_encoded = pd.get_dummies(
                df_encoded, 
                columns=[col], 
                prefix=col
            )
        
        elif 10 < n_categories <= 50:
            # Target Encoding para cardinalidade média
            # Implementação com Suavização (Smoothing) para evitar overfitting
            # te = ce.TargetEncoder(cols=[col])
            te = ce.TargetEncoder(cols=[col], smoothing=1.0)
            # df_encoded = te.fit_transform(df_encoded[[col]], df[target_col])
            if is_treino:
                df_encoded = te.fit_transform(df_encoded[[col]], dftarget)
                print('TargetEncoder com metodo fit_transform')                
            else:
                df_encoded = te.transform(df_encoded[[col]], dftarget) # df[target_col]
                print('TargetEncoder com metodo transform')                
        else:
            # Frequency Encoding para alta cardinalidade
            freq = df[col].value_counts(normalize=True)
            df_encoded[f'{col}_freq'] = df[col].map(freq)
            df_encoded.drop(col, axis=1, inplace=True)
    
    return df_encoded

# Avaliar o modelo XGBoosting
def avaliaPredicao(modelo, X_train, y_train, X_test, y_test):
    p_train = modelo.predict_proba(X_train)[:, 1]
    # p_train = modelo.predict(X_train)
        
    p_test = modelo.predict_proba(X_test)[:, 1]
    # p_test = modelo.predict(X_test)

    auc_train = roc_auc_score(y_train, p_train)
    auc_test = roc_auc_score(y_test, p_test)
    
    print(f'Avaliação do modelo na base de treino: AUC = {auc_train:.2f}')
    print(f'Avaliação do modelo na base de  teste: AUC = {auc_test:.2f}')
    
    fpr_train, tpr_train, _ = roc_curve(y_train, p_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, p_test)
    
    plt.figure(figsize=(10, 5))
    plt.plot(fpr_train, tpr_train, color='red', label=f'Treino AUC = {auc_train:.2f}', linewidth=2)
    plt.plot(fpr_test, tpr_test, color='blue', label=f'Teste AUC = {auc_test:.2f}', linewidth=1)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positivo')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

def avalia_classificacao(modelo, X, y, rótulos_y=['Em Atividade', 'Encerrou Atividade'], base = 'treino'):
    
    # Calcular as classificações preditas
    pred = modelo.predict(X)
    
    # Calcular a probabilidade de evento
    y_prob = modelo.predict_proba(X)[:, -1]
    
    # Calculando acurácia e matriz de confusão
    cm = confusion_matrix(y, pred)
    ac = accuracy_score(y, pred)
    bac = balanced_accuracy_score(y, pred)

    print(f'\nBase de {base}:')
    print(f'A acurácia é: {ac:.1%}')
    print(f'A acurácia balanceada é: {bac:.1%}')
    
    # Calculando AUC
    auc_score = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc_score:.2%}")
    # print(f"GINI: {(2*auc_score-1):.2%}")
    
    # Visualização gráfica
    sns.heatmap(cm, 
                annot=True, fmt='d', cmap='viridis', 
                xticklabels=rótulos_y, 
                yticklabels=rótulos_y)
    
    # Relatório de classificação do Scikit
    print('\n', classification_report(y, pred))
    
    # Gerar a Curva ROC
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    
    # Plotar a Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linha de referência (modelo aleatório)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - base de {base}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def descritiva(df_, var, vresp='ENCERROU_ATIVIDADE', max_classes=5):
    """
    Gera um gráfico descritivo da taxa de atividade empresarial por categoria da variável especificada.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    """
    
    df = df_.copy()
    
    if df[var].nunique()>max_classes:
        df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()

def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})


def diagnóstico(df_, var, vresp='survived', pred = 'pred', max_classes=5):
    """
    Gera um gráfico descritivo da taxa de sobreviventes por categoria da variável especificada.
    
    Parâmetros:
    df : DataFrame - Base de dados a ser analisada.
    var : str - Nome da variável categórica a ser analisada.
    """
    
    df = df_.copy()
    
    if df[var].nunique()>max_classes:
        df[var] = pd.qcut(df[var], max_classes, duplicates='drop')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1)
    sns.pointplot(data=df, y=pred, x=var, ax=ax1, color='red', linestyles='--', ci=None)
    
    # Criar o segundo eixo y para a taxa de sobreviventes
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, palette='viridis', alpha=0.5, ax=ax2)
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)  # Tornar o fundo do eixo 1 transparente
    
    # Exibir o gráfico
    plt.show()

def transformar_categoricas_dummies(df):
    for coluna in df.select_dtypes(include = ['object','string']).columns:
        dummies = pd.get_dummies(df[coluna], prefix = coluna, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(coluna, axis=1)
    return df
    
# Calcula a diferença de datas usando relativedelta
def calculaDiferencaDatasMeses(data_inicial, data_final):
    # Converte as strings para objetos datetime
    formatoData = "%d/%m/%Y %H:%M:%S"
    data1 = datetime.strptime(data_inicial, formatoData)
    data2 = datetime.strptime(data_final, formatoData)
    return relativedelta(data2, data1).months

def calculaDiferencaDatasDias(data_inicial, data_final):
    # Converte as strings para objetos datetime
    formatoData = "%d/%m/%Y %H:%M:%S"
    data1 = datetime.strptime(data_inicial, formatoData)
    data2 = datetime.strptime(data_final, formatoData)
    return relativedelta(data2, data1).days

def calculaDiferencaDatasHoras(data_inicial, data_final):
    # Converte as strings para objetos datetime
    formatoData = "%d/%m/%Y %H:%M:%S"
    data1 = datetime.strptime(data_inicial, formatoData)
    data2 = datetime.strptime(data_final, formatoData)
    return relativedelta(data2, data1).hours

def calculaDiferencaDatasMinutos(data_inicial, data_final):
    # Converte as strings para objetos datetime
    formatoData = "%d/%m/%Y %H:%M:%S"
    data1 = datetime.strptime(data_inicial, formatoData)
    data2 = datetime.strptime(data_final, formatoData)
    return relativedelta(data2, data1).minutes

def coverterDataStringDateTime(data_str):
    # Converte as strings para objetos datetime
    formatoData = "%d/%m/%Y %H:%M:%S"
    dataConvertida = datetime.strptime(data_str, formatoData)    
    return dataConvertida.date()

# Gerar os indicadores de desempenho
def gerarIndicadores(df):

    df['DataHomologacaoAno'] = df['DATA_HOMOLOGACAO_REGISTRO'].dt.year
    
    # calcular o Tempo Médio de Abertura de Empresa/Contribuintes, antes e depois da RedeSim.
    df['tempo_abertura'] = df['DATA_HOMOLOGACAO_REGISTRO'] - df['DATA_SOLICITACAO_REGISTRO']
    
    media_abertura = df.groupby('CADASTRO_VIA_REDESIM')['tempo_abertura'].mean().reset_index(name='tempo')
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
    df['tempo_sobrevivencia_meses'] = np.where(df['DATA_ENCERRAMENTO_ATIVIDADE'].notna(), 
                                              (df['DATA_ENCERRAMENTO_ATIVIDADE'] - df['DATA_INICIO_ATIVIDADE']).dt.days / 30, # Cálculo
                                               np.nan)           # Valor se for nulo
    
    media_sobrevivencia = df.groupby('CADASTRO_VIA_REDESIM')['tempo_sobrevivencia_meses'].mean().reset_index(name='tempo')
    tempo1 = int(media_sobrevivencia.loc[media_abertura['CADASTRO_VIA_REDESIM'] == 'N', 'tempo'])
    tempo2 = int(media_sobrevivencia.loc[media_abertura['CADASTRO_VIA_REDESIM'] == 'S', 'tempo'])
    print(f"Tempo médio de sobrevivência até o encerramento da atividade empresarial (antes RedeSim) = {tempo1} meses")
    print(f"Tempo médio de sobrevivência até o encerramento da atividade empresarial (após RedeSim)  = {tempo2}  meses")
    
    
    # Calcular a quantidade de abertura de empresa: antes e após a integração com a RedeSim
    # 1.Agrupar por ano e indicador de cadastro via RedeSim
    dfAgrupamento = df.groupby(['DataHomologacaoAno', 'CADASTRO_VIA_REDESIM']).size() #.unstack().fillna(0)
    
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
    dfAgrupamento = df.groupby(['DataHomologacaoAno', 'MUNICIPIO']).size().reset_index(name='quantidade')
    
    
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
    df.drop(columns=['DataHomologacaoAno', 'tempo_abertura', 'tempo_sobrevivencia_meses'], inplace=True)
    
   