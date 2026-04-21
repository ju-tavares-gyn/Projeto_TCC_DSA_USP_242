# -*- coding: utf-8 -*-
"""
Created em 12/02/2026
@author: Juliano Tavares
"""

#%%  funções de ajuda

import pandas as pd
import numpy as np
import category_encoders as ce

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score
import shap as shap
    
import seaborn as sns
import matplotlib.pyplot as plt

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
    
def gerarMetricasModelo(predicts, observado, cutoff=None, base='Treino'):
    
    values = predicts.values
    
    predicao_binaria = []
    
    if cutoff is None:
        predicao_binaria = predicts
        print('parâmetro cutoff é nulo')
    else:
        print(f'parâmetro cutoff igual a {cutoff}')
        for item in values:
            if item < cutoff:
                predicao_binaria.append(0)
            else:
                predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format='d')# Formatar como inteiro os valores da matriz de confusão
    plt.title('Matriz de Confusão - Base de ' + base)    
    plt.xlabel('Observado(Real)')
    plt.ylabel('Classificado')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Calculando AUC
    auc_score = roc_auc_score(observado, predicts)
       
    # Visualização dos principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Base':[base],
                                'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia],
                                'AUC-ROC':[auc_score]})
        
    # print(f"GINI: {(2*auc_score-1):.2%}")
    print(indicadores)
    
    # Relatório de classificação do Scikit
    print('\n', classification_report(observado, predicao_binaria, digits=6))
    
    # Gerar a Curva ROC
    fpr, tpr, thresholds = roc_curve(observado, predicts)
    
    # Plotar a Curva ROC
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {auc_score:.6f})')
    plt.plot([1, 0], [1, 0], color='red', linestyle='--')  # Linha de referência (modelo aleatório)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(f"Curva ROC - base de {base}")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


# Criação da função 'espec_sens' para a construção de um dataset com diferentes
# valores de cutoff, sensitividade e especificidade:
def espec_sens(observado,predicts):
    
    # Adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # Range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.01)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado

# Gerar gráfico do algoritmo SHAP que é uma das melhores técnicas para explicar modelos de classificação com o XGBoost.
# Será usado o algoritmo Tree SHAP, específico para árvores.
def gerarGraficoSHAP(modelo, df_teste):
    explainerTreeShap = shap.TreeExplainer(modelo, df_teste)

    # Calcula os valores SHAP para o conjunto de teste
    # O resultado será uma matriz com o impacto de cada feature para cada amostra.
    shap_values = explainerTreeShap.shap_values(df_teste)

    feature_names = df_teste.columns.tolist()

    # Cria um objeto Explanation com os nomes
    shap_exp = shap.Explanation(values=shap_values, 
                                base_values=explainerTreeShap.expected_value, 
                                data=df_teste, 
                                feature_names=feature_names)

    # Gera o gráfico de barras (bar plot) com os nomes corretos
    shap.plots.bar(shap_exp, max_display=30, show=False)
    
    
    ax = plt.gca()
    ax.set_xlabel("Impacto no Modelo (Valor médio SHAP)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Features", fontsize=12, fontweight='bold')
    ax.set_title("Importância das Features - SHAP", fontsize=16, pad=20)

    # Define o tamanho da figura: (largura em polegadas, altura em polegadas)
    plt.figure(figsize=(20, 15))
    
    # Adiciona rodapé
    plt.figtext(0.5, -0.05, "Valores positivos aumentam a previsão da classe positiva", 
                ha="center", fontsize=9, style='italic')
    
    plt.tight_layout()
        
    # Exibe o gráfico
    plt.show()    

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

def transformar_categoricas_dummies(df):
    for coluna in df.select_dtypes(include = ['object','string']).columns:
        dummies = pd.get_dummies(df[coluna], prefix = coluna, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(coluna, axis=1)
    return df