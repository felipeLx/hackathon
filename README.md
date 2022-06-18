# hackathon
Projeto Hackaton Ministério da Saúde + Campus Party

# aplicação online: 
https://share.streamlit.io/felipelx/hackathon/IDC_Detector.py

## Suporte na Detecção de Câncer de mama por imagem e por variável
Projeto é dividido em duas partes e três páginas, realizadas com streamlit.
A navegação é feita na barra lateral, sendo as duas páginas da análise da imagem com opções de subir arquivo de imagem. A página 
com as variáveis conta com 30 opções para o usuário definir os parámetros que são respondidos com o retorno das radiografias.

## Primeira parte - Variáveis
Foi considerado um dataset de centro de pesquisa em câncer que para o treinamento do Machine Learning considera 30 variáveis.
  'Raio Médio' : radius_mean, 'Textura Média' : texture_mean,  'Perímetro Médio' : perimeter_mean,  'Área Média' : area_mean,
  'Suavidade Média' : smoothness_mean,  'Compacidade Média' : compactness_mean,  'Concavidade Média' : concavity_mean,
  'Pontos Concavidade Média' : concave_points_mean,  'Simetria Média' : symmetry_mean, 'Dimensão Fractual Média' : fractal_dimension_mean,
  'Raio SE' : radius_se,  'Textura SE' : texture_se,  'Perímetro SE' : perimeter_se,  'Área SE' : area_se,  'Suavidade SE' : smoothness_se,
  'Compacidade SE' : compactness_se,  'Concavidade SE' : concavity_se,  'Pontos Côncavos SE' : concave_points_se,
  'Simetria SE' : symmetry_se,  'Dimensão Fractual SE' : fractal_dimension_se,  'Raio Pior' : radius_worst,
  'Textura Pior' : texture_worst,  'Perímetro Pior' : perimeter_worst,  'Área Pior' : area_worst,  'Suavidade Pior' : smoothness_worst,
  'Compacidade Pior' : compactness_worst,  'Concavidade Pior' : concavity_worst,  'Pontos Côncavos Pior' : concave_points_worst,
  'Simetria Pior' : symmetry_worst,  'Dimensão Fractual Pior' : fractal_dimension_worst

Para o treinamento do modelo é usado sklearn RandomForestClassifier por ter a melhor acurária depois das opções utilizadas/testadas.

## Segunda parte - Imagens
Foram utilizados dois modelos, nos quais o foco é Metástase e Carcimona.
Modelo foi contruido a partir da análise de aproximadamente 80 mil imagens, e categorizados por grupos: 0 Benéfico/Normal ou 1 Maligno.
Projeto utiliza Tensorflow/Keras para a criação do modelo e projeção da probabilidade.

O modelo deve ser criado de forma específica para cada caso, com testes do melhor processo e se possível sempre retro alimentado
para aumentar a base de dados e melhorar a acurária.
