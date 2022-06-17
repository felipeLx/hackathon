import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Detectar por Variáveis", page_icon="🧬")
st.markdown("# Análise de variáveis")
st.sidebar.header("Análise de variáveis")

#Titulo
st.title("""
Prevendo Câncer\n
App que utiliza machine learn para prever possível Câncer de mama de acordo com os dados de um exame.
""")

#dataset
df = pd.read_csv("pages/models/data.csv")
#print(df.isna().sum())
#print(df.info())
df = df.iloc[:,0:32]
# df['diagnisis_int'] = np.where(df['diagnosis'] == 'M', 1, 0)
# print(df.info())

#cabeçalho
st.subheader('Informações dos dados')

#dados de entrada
X= df.iloc[:, 2:32]
Y= df['diagnosis']

#Dados em treinamentos e teste
X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

#dados dos usuarios
radius_mean = st.sidebar.slider('Raio Médio', 0.284, 17.463, 12.146 )
texture_mean = st.sidebar.slider('Textura Média', 1.120, 21.604, 17.914)
perimeter_mean = st.sidebar.slider('Perímetro Médio', 2.000, 115.365, 78.075)
area_mean = st.sidebar.slider('Área Média', 21.135, 978.376, 462.790)
smoothness_mean = st.sidebar.slider('Suavidade Média', 0.006, 0.103, 0.092)
compactness_mean = st.sidebar.slider('Compacidade Média', 0.0214, 0.145, 0.080)
concavity_mean = st.sidebar.slider('Concavidade Média', 0.0259, 0.161, 0.046)
concave_points_mean = st.sidebar.slider('Pontos Côncavos Média', 0.009, 0.88, 0.025)
symmetry_mean = st.sidebar.slider('Simetria Média', 0.020, 0.193, 0.174)
fractal_dimension_mean = st.sidebar.slider('Dimensão Fractual Média', 0.003, 0.063, 0.060)
radius_se = st.sidebar.slider('Raio SE',  0.284, 17.463, 12.146)
texture_se = st.sidebar.slider('Textura SE', 1.120, 21.604, 17.914)
perimeter_se = st.sidebar.slider('Perímetro SE', 2.000, 115.365, 78.075)
area_se = st.sidebar.slider('Área SE', 21.135, 978.376, 462.790)
smoothness_se = st.sidebar.slider('Suavidade SE', 0.006, 0.103, 0.092)
compactness_se = st.sidebar.slider('Compacidade SE', 0.0214, 0.145, 0.080)
concavity_se = st.sidebar.slider('Concavidade SE', 0.0259, 0.161, 0.046)
concave_points_se = st.sidebar.slider('Pontos Côncavos SE', 0.009, 0.88, 0.025)
symmetry_se = st.sidebar.slider('Simetria SE', 0.020, 0.193, 0.174)
fractal_dimension_se = st.sidebar.slider('Dimensão Fractual SE', 0.003, 0.063, 0.060)
radius_worst = st.sidebar.slider('Raio Pior',  0.284, 17.463, 12.146)
texture_worst = st.sidebar.slider('Textura Pior', 1.120, 21.604, 17.914)
perimeter_worst = st.sidebar.slider('Perímetro Pior', 2.000, 115.365, 78.075)
area_worst = st.sidebar.slider('Área Pior',21.135, 978.376, 462.790)
smoothness_worst = st.sidebar.slider('Suavidade Pior', 0.006, 0.103, 0.092)
compactness_worst = st.sidebar.slider('Compacidade Pior', 0.0214, 0.145, 0.080)
concavity_worst = st.sidebar.slider('Concavidade Pior', 0.0259, 0.161, 0.046)
concave_points_worst = st.sidebar.slider('Pontos Côncavos Pior', 0.009, 0.88, 0.025)
symmetry_worst =  st.sidebar.slider('Simetria Pior', 0.020, 0.193, 0.174)
fractal_dimension_worst = st.sidebar.slider('Dimensão Fractual Pior', 0.003, 0.063, 0.060)

#dados dos usuarios com a função
def get_user_data(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst):
    user_data={
            'Raio Médio' : radius_mean,
            'Textura Média' : texture_mean,
            'Perímetro Médio' : perimeter_mean,
            'Área Média' : area_mean,
            'Suavidade Média' : smoothness_mean,
            'Compacidade Média' : compactness_mean,
            'Concavidade Média' : concavity_mean,
            'Pontos Concavidade Média' : concave_points_mean,
            'Simetria Média' : symmetry_mean,
            'Dimensão Fractual Média' : fractal_dimension_mean,
            'Raio SE' : radius_se,
            'Textura SE' : texture_se,
            'Perímetro SE' : perimeter_se,
            'Área SE' : area_se,
            'Suavidade SE' : smoothness_se,
            'Compacidade SE' : compactness_se,
            'Concavidade SE' : concavity_se,
            'Pontos Côncavos SE' : concave_points_se,
            'Simetria SE' : symmetry_se,
            'Dimensão Fractual SE' : fractal_dimension_se,
            'Raio Pior' : radius_worst,
            'Textura Pior' : texture_worst,
            'Perímetro Pior' : perimeter_worst,
            'Área Pior' : area_worst,
            'Suavidade Pior' : smoothness_worst,
            'Compacidade Pior' : compactness_worst,
            'Concavidade Pior' : concavity_worst,
            'Pontos Côncavos Pior' : concave_points_worst,
            'Simetria Pior' : symmetry_worst,
            'Dimensão Fractual Pior' : fractal_dimension_worst
            }

    features= pd.DataFrame(user_data, index=[0])
    return features

#Grafico
#graf = st.bar_chart(user_input_variables)
#st.subheader('Dados do usuário')
#st.write(user_input_variables)

dtc = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=0)
dtc.fit(X_train, y_train)

#Acuracia
# st.subheader('Acurácia do modelo')
# st.write(accuracy_score(y_test, dtc.predict(x_test)) * 100)

# result_report = classification_report(user_input_variables, y_test, target_names=target_name)
# print(result_report)


c = st.container()
c.subheader('Previsão: ')
generate_pred = c.button("Gerar Previsão")    
if generate_pred:
    user_input_variables= get_user_data(radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst)

    #Previsao
    prediction = dtc.predict(user_input_variables)
    # target_name = ['B', 'M']
    result_cross = str((dtc.predict_proba(user_input_variables)[:,0]* 100).round(2))
    result_cross = result_cross.replace('[', '').replace(']', '') 
    result_cross_str = str(result_cross) + '%'
    print('result_cross', result_cross_str)

    result = ''
    if prediction == 'B':
        result = 'Benigno'
    else:
        result = 'Maligno'

    c.metric("Resultado", result + ' (' + result_cross_str + ')')