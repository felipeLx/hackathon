from operator import concat
from pyexpat import features
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Titulo
st.title("""
Prevendo Câncer\n
App que utiliza machine learn para prever possível Câncer de mama
Fonte: 
""")

#dataset
df = pd.read_csv("C:/Users/mateu/Downloads/archive/data.csv")

#cabeçalho
st.subheader('Informações dos dados')

#Nome
user_input = st.sidebar.text_input('Digite seu nome')

st.write('Paciente: ', user_input)

#dados de entrada

X= df.drop(['diagnosis'], axis=1)
Y= df['diagnosis']

#Dados em treinamentos e teste
X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)

#dados dos usuarios com a função
def get_user_data():
    radius_mean = st.sidebar.slider('Raio Médio', 0.284, 17.463, 12.146 )
    texture_mean = st.sidebar.slider('Textura Média', 1.120, 21.604, 17.914)
    perimeter_mean = st.sidebar.slider('Perímetro Médio', 2.000, 115.365, 78.075)
    area_mean = st.sidebar.slider('Área Média', 21.135, 978.376, 462.790)
    smoothness = st.sidebar.slider('Suavidade', 0.006, 0.103, 0.092)
    compactness = st.sidebar.slider('Compacidade', 0.0214, 0.145, 0.080)
    concavity = st.sidebar.slider('Concavidade', 0.0259, 0.161, 0.046)
    concave_points = st.sidebar.slider('Pontos Côncavos', 0.009, 0.88, 0.025)
    symmetry = st.sidebar.slider('Simetria', 0.020, 0.193, 0.174)
    fractal_dimension = st.sidebar.slider('Dimensão Fractual', 0.003, 0.063, 0.060)

    user_data={
            'Raio Médio' : radius_mean,
            'Textura Média' : texture_mean,
            'Perímetro Médio' : perimeter_mean,
            'Área Média' : area_mean,
            'Suavidade' : smoothness,
            'Compacidade' : compactness,
            'Concavidade' : concavity,
            'Concavidade' : concave_points,
            'Pontos Côncavos' : concave_points,
            'Simetria' : symmetry,
            'Dimensão Fractual' : fractal_dimension
            }

    features= pd.DataFrame(user_data, index=[0])
    return features
user_input_variables= get_user_data()

#Grafico
graf = st.bar_chart(user_input_variables)
st.subheader('Dados do usuário')
st.write(user_input_variables)

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(X_train, y_train)

#Acuracia
st.subheader('Acurácia do modelo')
st.write(accuracy_score(y_test, dtc.predict(x_test)) * 100)
#Previsao
prediction = dtc.predict(user_input_variables)

st.subheader('Previsão: ')
st.write(prediction)