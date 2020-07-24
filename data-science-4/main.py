#!/usr/bin/env python
# coding: utf-8

# # Desafio 6
# 
# Neste desafio, vamos praticar _feature engineering_, um dos processos mais importantes e trabalhosos de ML. Utilizaremos o _data set_ [Countries of the world](https://www.kaggle.com/fernandol/countries-of-the-world), que contém dados sobre os 227 países do mundo com informações sobre tamanho da população, área, imigração e setores de produção.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk


# In[2]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

#from IPython.core.pylabtools import figsize


#figsize(12, 8)

#sns.set()


# In[3]:


countries = pd.read_csv("countries.csv")


# In[4]:


new_column_names = [
    "Country", "Region", "Population", "Area", "Pop_density", "Coastline_ratio",
    "Net_migration", "Infant_mortality", "GDP", "Literacy", "Phones_per_1000",
    "Arable", "Crops", "Other", "Climate", "Birthrate", "Deathrate", "Agriculture",
    "Industry", "Service"
]

countries.columns = new_column_names

countries.head(5)


# ## Observações
# 
# Esse _data set_ ainda precisa de alguns ajustes iniciais. Primeiro, note que as variáveis numéricas estão usando vírgula como separador decimal e estão codificadas como strings. Corrija isso antes de continuar: transforme essas variáveis em numéricas adequadamente.
# 
# Além disso, as variáveis `Country` e `Region` possuem espaços a mais no começo e no final da string. Você pode utilizar o método `str.strip()` para remover esses espaços.

# ## Inicia sua análise a partir daqui

# #### Característica dos dados

# In[5]:


countries_copy = countries.copy()
print(f'Linhas: {countries_copy.shape[0]} | Colunas: {countries_copy.shape[1]}')


# In[6]:


print(f'Tipos dos dados: {countries_copy.dtypes.unique()}')


# #### Informações dos dados

# In[7]:


countries_copy.info()


# #### Estatística dos dados

# In[8]:


countries_copy.describe()


# #### Pré-processamento dos dados

# In[9]:


countries_copy[['Country', 'Region']] = countries_copy[['Country', 'Region']].apply(lambda x: x.str.strip())


# #### Conversão de tipos

# In[10]:


cols = ['Pop_density', 'Coastline_ratio', 'Net_migration', 'Infant_mortality', 'Literacy', 'Phones_per_1000', 'Arable', 
        'Crops', 'Other', 'Climate', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']

countries_copy[cols] = countries_copy[cols].apply(lambda x: x.str.replace(',', '.').astype('float'))


# #### Dados nulos

# In[11]:


data_missing = pd.DataFrame({'nomes': countries_copy.columns, 
                             'tipos': countries_copy.dtypes, 
                             'NA #': countries_copy.isna().sum(),
                             'NA %': (countries_copy.isna().sum() / countries_copy.shape[0]) * 100
                            })

data_missing[data_missing['NA #'] > 0].set_index('nomes')


# ## Questão 1
# 
# Quais são as regiões (variável `Region`) presentes no _data set_? Retorne uma lista com as regiões únicas do _data set_ com os espaços à frente e atrás da string removidos (mas mantenha pontuação: ponto, hífen etc) e ordenadas em ordem alfabética.

# In[12]:


def q1():
    regions = list(countries_copy['Region'].unique())
    return sorted(regions)


# In[13]:


q1()


# ## Questão 2
# 
# Discretizando a variável `Pop_density` em 10 intervalos com `KBinsDiscretizer`, seguindo o encode `ordinal` e estratégia `quantile`, quantos países se encontram acima do 90º percentil? Responda como um único escalar inteiro.

# In[14]:


from sklearn.preprocessing import KBinsDiscretizer


# In[15]:


def q2():
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile').fit_transform(countries_copy[['Pop_density']])
    quantile = np.quantile(discretizer, 0.90)
    return int(sum((discretizer > quantile)))


# In[16]:


q2()


# # Questão 3
# 
# Se codificarmos as variáveis `Region` e `Climate` usando _one-hot encoding_, quantos novos atributos seriam criados? Responda como um único escalar.

# In[17]:


def q3():
    return int(countries_copy['Region'].nunique() + countries_copy['Climate'].nunique()) + 1


# In[18]:


q3()


# ## Questão 4
# 
# Aplique o seguinte _pipeline_:
# 
# 1. Preencha as variáveis do tipo `int64` e `float64` com suas respectivas medianas.
# 2. Padronize essas variáveis.
# 
# Após aplicado o _pipeline_ descrito acima aos dados (somente nas variáveis dos tipos especificados), aplique o mesmo _pipeline_ (ou `ColumnTransformer`) ao dado abaixo. Qual o valor da variável `Arable` após o _pipeline_? Responda como um único float arredondado para três casas decimais.

# In[19]:


test_country = [
    'Test Country', 'NEAR EAST', -0.19032480757326514,
    -0.3232636124824411, -0.04421734470810142, -0.27528113360605316,
    0.13255850810281325, -0.8054845935643491, 1.0119784924248225,
    0.6189182532646624, 1.0074863283776458, 0.20239896852403538,
    -0.043678728558593366, -0.13929748680369286, 1.3163604645710438,
    -0.3699637766938669, -0.6149300604558857, -0.854369594993175,
    0.263445277972641, 0.5712416961268142
]


# In[20]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


cols = countries_copy.select_dtypes(['int', 'float']).columns.values

pipe = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), 
    ('scaler', StandardScaler())
])


# In[21]:


def q4():
    pipe.fit(countries_copy[cols])
    columnTransformer = pipe.transform([test_country[2:]])
    return float(columnTransformer[0][9].round(3))


# In[22]:


q4()


# ## Questão 5
# 
# Descubra o número de _outliers_ da variável `Net_migration` segundo o método do _boxplot_, ou seja, usando a lógica:
# 
# $$x \notin [Q1 - 1.5 \times \text{IQR}, Q3 + 1.5 \times \text{IQR}] \Rightarrow x \text{ é outlier}$$
# 
# que se encontram no grupo inferior e no grupo superior.
# 
# Você deveria remover da análise as observações consideradas _outliers_ segundo esse método? Responda como uma tupla de três elementos `(outliers_abaixo, outliers_acima, removeria?)` ((int, int, bool)).

# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = [10, 5]

sns.boxplot(y=countries_copy['Net_migration'])

plt.title('Analisando outliers', y=1.03, size=14, loc='left', x=-0.008)
plt.show()


# In[24]:


def q5():
    iqr = countries_copy['Net_migration'].quantile(0.75) - countries_copy['Net_migration'].quantile(0.25)
    q1 = countries_copy['Net_migration'].quantile(0.25) - 1.5 * iqr
    q3 = countries_copy['Net_migration'].quantile(0.75) + 1.5 * iqr
    
    outliers_abaixo = int((countries_copy['Net_migration'] < q1).sum())
    outliers_acima =  int((countries_copy['Net_migration'] > q3).sum())
    
    return outliers_abaixo, outliers_acima, False


# In[25]:


q5()


# ## Questão 6
# Para as questões 6 e 7 utilize a biblioteca `fetch_20newsgroups` de datasets de test do `sklearn`
# 
# Considere carregar as seguintes categorias e o dataset `newsgroups`:
# 
# ```
# categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
# newsgroup = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
# ```
# 
# 
# Aplique `CountVectorizer` ao _data set_ `newsgroups` e descubra o número de vezes que a palavra _phone_ aparece no corpus. Responda como um único escalar.

# In[26]:


from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

categories = ['sci.electronics', 'comp.graphics', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)


# In[27]:


count_vectorizer = CountVectorizer()
newsgroups_counts = count_vectorizer.fit_transform(newsgroups.data)
words_idx = sorted([count_vectorizer.vocabulary_.get(f"{word.lower()}") for word in [u"phone"]])


# In[28]:


def q6():
    frame = pd.DataFrame(newsgroups_counts[:, words_idx].toarray(), 
                         columns=np.array(count_vectorizer.get_feature_names())[words_idx])
    return int(frame['phone'].sum())


# In[29]:


q6()


# ## Questão 7
# 
# Aplique `TfidfVectorizer` ao _data set_ `newsgroups` e descubra o TF-IDF da palavra _phone_. Responda como um único escalar arredondado para três casas decimais.

# In[30]:


tfidf_transformer = TfidfTransformer()


# In[31]:


def q7():
    tfidf_transformer.fit(newsgroups_counts)
    newsgroups_tfidf = tfidf_transformer.transform(newsgroups_counts)

    frame = pd.DataFrame(newsgroups_tfidf[:, words_idx].toarray(), 
                             columns=np.array(count_vectorizer.get_feature_names())[words_idx])
    return float(frame['phone'].sum().round(3))


# In[32]:


q7()

