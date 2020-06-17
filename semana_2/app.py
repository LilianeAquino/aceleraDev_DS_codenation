import streamlit as st
import pandas as pd
import base64

def get_table_download_link(df):
    """Gera um link permitindo o download dos dados em um quadro de dados do pandas
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href

def main():
    st.image('logo.png', width=200)
    st.title('AceleraDev Data Science Codenation')
    st.subheader('Semana 2 - Pré-processamento de dados em Python')
    st.image('https://media3.giphy.com/media/z6xE1olZ5YP4I/giphy.gif?cid=ecf05e47eacefb9900b1f68bf34d120d8617da690eb3f604&rid=giphy.gif', width=200)
    
    file = st.file_uploader('Escolha os dados que deseja analisar (.csv)', type='csv')
    if file is not None:
        st.subheader('Analisando os dados: ')
        df = pd.read_csv(file)
        st.markdown('**Número de linhas:**')
        st.markdown(df.shape[0])
        st.markdown('**Número de colunas:**')
        st.markdown(df.shape[1])
        st.markdown('**Visualizando o dataframe**')
        number = st.slider('Escolha o numero de colunas que deseja ver', min_value=1, max_value=50)
        st.dataframe(df.head(number))
        st.markdown('**Nome das colunas:**')
        st.markdown(list(df.columns))
        exploracao = pd.DataFrame({'nomes' : df.columns, 'tipos' : df.dtypes, 'NA #': df.isna().sum(), 'NA %' : (df.isna().sum() / df.shape[0]) * 100})
        st.markdown('**Contagem dos tipos de dados:**')
        st.write(exploracao.tipos.value_counts())
        st.markdown('**Nomes das colunas do tipo int64:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'int64']['nomes']) or 0)
        st.markdown('**Nomes das colunas do tipo float64:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'float64']['nomes']) or 0)
        st.markdown('**Nomes das colunas do tipo object:**')
        st.markdown(list(exploracao[exploracao['tipos'] == 'object']['nomes']) or 0)
        st.markdown('**Tabela com coluna e percentual de dados faltantes :**')
        st.table(exploracao[exploracao['NA #'] != 0][['tipos', 'NA %']])
        st.subheader('Inputaçao de dados númericos :')
        percentual = st.slider('Escolha o limite de percentual faltante para as colunas vocë deseja inputar os dados', min_value=0, max_value=100)
        lista_colunas = list(exploracao[exploracao['NA %']  < percentual]['nomes'])
        select_method = st.radio('Escolha um metodo abaixo :', ('Média', 'Mediana'))
        st.markdown('Você selecionou : ' +str(select_method))
        if select_method == 'Média':
            df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
            exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns, 'tipos': df_inputado.dtypes, 'NA #': df_inputado.isna().sum(),
                                       'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
            st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
            st.subheader('Dados Inputados, faça o download abaixo : ')
            st.markdown(get_table_download_link(df_inputado), unsafe_allow_html=True)
        if select_method == 'Mediana':
            df_inputado = df[lista_colunas].fillna(df[lista_colunas].mean())
            exploracao_inputado = pd.DataFrame({'nomes': df_inputado.columns, 'tipos': df_inputado.dtypes, 'NA #': df_inputado.isna().sum(),
                                       'NA %': (df_inputado.isna().sum() / df_inputado.shape[0]) * 100})
            st.table(exploracao_inputado[exploracao_inputado['tipos'] != 'object']['NA %'])
            st.subheader('Dados Inputados faça download abaixo : ')
            st.markdown(get_table_download_link(df_inputado), unsafe_allow_html=True)

if __name__=='__main__':
    main()