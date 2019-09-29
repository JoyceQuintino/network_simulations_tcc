import warnings
warnings.filterwarnings('ignore')
import dataextraction as de
import machinelearning as ml
import resultsgeneration as rg
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

def mlp():
    #Resultados para a taxa de entrega
    n = 30
    flag = 1
    cam = "C:/Users/Joyce Quintino/joyce/dados-tcc/cenario1 - v4/"
    cam2 = "C:/Users/joyce/Documents/joyce/programsimulations/extraction_data/"

    a = '0.1'
    b = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', 
         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', 
         '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', 
         '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', 
         '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', 
         '105', '106', '107', '108', '109']
    c = '256'

    data_simulations, icmin, icmax = de.data_formulation(n, flag, cam, a, b, c)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x = data_simulations.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y = data_simulations.iloc[:, 3:4].values

    random.seed(1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()
    scaler_test_x = StandardScaler()
    scaler_test_y = StandardScaler()

    u = scaler_test_x.fit_transform(x_test)
    x = scaler_train_x.fit_transform(x_train)
    y = scaler_train_y.fit_transform(y_train)
    v = scaler_test_y.fit_transform(y_test)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(u)

    #Treinamento da MLP
    mlp = ml.optimize_parameters(MLPRegressor(), x, y, ml.definitions_algorithms('MLP'))

    mlp = MLPRegressor(alpha=mlp['alpha'],
                       activation=mlp['activation'], 
                       learning_rate_init=mlp['learning_rate_init'], 
                       solver=mlp['solver'], 
                       hidden_layer_sizes=mlp['hidden_layer_sizes'])

    score, error, mlp = ml.experiment_folds('mlp', mlp, x, y)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score)
    print("Média do erro -> ", error)
    print("Modelo -> ", mlp)
    '''

    #Pacote - 256

    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()

    x1 = scaler_train_x.fit_transform(data_x)
    y1 = scaler_train_y.fit_transform(data_y)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(x1)

    #Carregamento do modelo
    #print("Valores da simulação - conjunto de validação -> \n", scaler_train_y.inverse_transform(y1))
    simu1 = sorted(scaler_train_y.inverse_transform(y1))
    #print(simu1)
    #print("Valores da predição - MLP -> \n", scaler_train_y.inverse_transform(mlp.predict(x1)))
    #print("Valores da predição ordenados - MLP -> \n", sorted(scaler_train_y.inverse_transform(mlp.predict(x1))))
    pred1 = sorted(scaler_train_y.inverse_transform(mlp.predict(x1)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali = np.array(sorted([(x1[0],y1[0]) for x1,y1 in zip(set_test, simu1)]))
    predi1 = np.array(sorted([(x1[0],y1) for x1,y1 in zip(set_test, pred1)]))
    print(predi1[:,0])
    print(predi1[:,1])

    print('Resultados das métricas')
    print('Valor do Score - ', mlp.score(x1, y1))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y1, mlp.predict(x1))))
    
    icmin = np.array(icmin)
    icmax = np.array(icmax)

    min = vali[:,1]-icmin
    max = icmax-vali[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'mlp', 1, a, c, min, max, vali, predi1, pred1, set_test)

    #Pacote de 512
    c2 = '512'

    data_simulations2, icmin2, icmax2 = de.data_formulation(n, flag, cam, a, b, c2)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x2 = data_simulations2.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y2 = data_simulations2.iloc[:, 3:4].values

    random.seed(1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(data_x2, data_y2, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()
    scaler_test_x2 = StandardScaler()
    scaler_test_y2 = StandardScaler()

    u2 = scaler_test_x2.fit_transform(x_test2)
    x2 = scaler_train_x2.fit_transform(x_train2)
    y2 = scaler_train_y2.fit_transform(y_train2)
    v2 = scaler_test_y2.fit_transform(y_test2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(u2)

    #Treinamento da MLP
    mlp2 = ml.optimize_parameters(MLPRegressor(), x2, y2, ml.definitions_algorithms('MLP'))

    mlp2 = MLPRegressor(alpha=mlp2['alpha'],
                        activation=mlp2['activation'], 
                        learning_rate_init=mlp2['learning_rate_init'], 
                        solver=mlp2['solver'], 
                        hidden_layer_sizes=mlp2['hidden_layer_sizes'])

    score2, error2, mlp2 = ml.experiment_folds('mlp', mlp2, x2, y2)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score2)
    print("Média do erro -> ", error2)
    print("Modelo ->", mlp2)
    '''

    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()

    x3 = scaler_train_x2.fit_transform(data_x2)
    y3 = scaler_train_y2.fit_transform(data_y2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(x3)

    #Carregamento do modelo
    #print("Valores da simulação - conjunto de validação -> \n", scaler_train_y2.inverse_transform(y3))
    simu2 = sorted(scaler_train_y2.inverse_transform(y3))
    #print("Valores da predição - MLP -> \n", scaler_train_y2.inverse_transform(mlp2.predict(x3)))
    #print("Valores da predição ordenados - MLP -> \n", sorted(scaler_train_y2.inverse_transform(mlp2.predict(x3))))
    pred2 = sorted(scaler_train_y2.inverse_transform(mlp2.predict(x3)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali2 = np.array(sorted([(x3[0],y3[0]) for x3,y3 in zip(set_test2, simu2)]))
    predi2 = np.array(sorted([(x3[0],y3) for x3,y3 in zip(set_test2, pred2)]))

    print('Resultados das métricas')
    print('Valor do Score - ', mlp2.score(x3, y3))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y3, mlp2.predict(x3))))

    icmin2 = np.array(icmin2)
    icmax2 = np.array(icmax2)

    min2 = vali2[:,1]-icmin2
    max2 = icmax2-vali2[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'mlp', 2, a, c2, min2, max2, vali2, predi2, pred2, set_test2)

    #Pacote de 1024
    c3 = '1024'

    data_simulations3, icmin3, icmax3 = de.data_formulation(n, flag, cam, a, b, c3)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x3 = data_simulations3.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y3 = data_simulations3.iloc[:, 3:4].values

    random.seed(1)
    x_train3, x_test3, y_train3, y_test3 = train_test_split(data_x3, data_y3, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()
    scaler_test_x3 = StandardScaler()
    scaler_test_y3 = StandardScaler()

    u3 = scaler_test_x3.fit_transform(x_test3)
    x3 = scaler_train_x3.fit_transform(x_train3)
    y3 = scaler_train_y3.fit_transform(y_train3)
    v3 = scaler_test_y3.fit_transform(y_test3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(u3)

    #Treinamento da MLP
    mlp3 = ml.optimize_parameters(MLPRegressor(), x3, y3, ml.definitions_algorithms('MLP'))

    mlp3 = MLPRegressor(alpha=mlp3['alpha'],
                        activation=mlp3['activation'], 
                        learning_rate_init=mlp3['learning_rate_init'], 
                        solver=mlp3['solver'], 
                        hidden_layer_sizes=mlp3['hidden_layer_sizes'])

    score3, error3, mlp3 = ml.experiment_folds('mlp', mlp3, x3, y3)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score3)
    print("Média do erro -> ", error3)
    print("Modelo ->", mlp3)
    '''

    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()

    x4 = scaler_train_x3.fit_transform(data_x3)
    y4 = scaler_train_y3.fit_transform(data_y3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(x4)

    #Carregamento do modelo
    #print("Valores da simulação - conjunto de validação -> \n", scaler_train_y3.inverse_transform(y4))
    simu3 = sorted(scaler_train_y3.inverse_transform(y4))
    #print("Valores da predição - MLP -> \n", scaler_train_y3.inverse_transform(mlp3.predict(x4)))
    #print("Valores da predição ordenados - MLP -> \n", sorted(scaler_train_y3.inverse_transform(mlp3.predict(x4))))
    pred3 = sorted(scaler_train_y3.inverse_transform(mlp3.predict(x4)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali3 = np.array(sorted([(x4[0],y4[0]) for x4,y4 in zip(set_test3, simu3)]))
    predi3 = np.array(sorted([(x4[0],y4) for x4,y4 in zip(set_test3, pred3)]))

    print('Resultados das métricas')
    print('Valor do Score - ', mlp3.score(x4, y4))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y4, mlp3.predict(x4))))

    icmin3 = np.array(icmin3)
    icmax3 = np.array(icmax3)

    min3 = vali3[:,1]-icmin3
    max3 = icmax3-vali3[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'mlp', 3, a, c3, min3, max3, vali3, predi3, pred3, set_test3)

    rg.create_graphic_interval_confidence('Taxa de entrega %', 'mlp', 3, a, min, max, vali, predi1, pred1, set_test,
                                                                    min2, max2, vali2, predi2, pred2, set_test2,
                                                                    min3, max3, vali3, predi3, pred3, set_test3)

def dt():
    print('DT')
    #Resultados para a taxa de entrega
    n = 30
    flag = 1
    cam = "C:/Users/Joyce Quintino/joyce/dados-tcc/cenario1 - v4/"
    cam2 = "C:/Users/joyce/Documents/joyce/programsimulations/extraction_data/"

    a = '0.1'
    b = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', 
         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', 
         '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', 
         '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', 
         '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', 
         '105', '106', '107', '108', '109']
    c = '256'

    data_simulations, icmin, icmax = de.data_formulation(n, flag, cam, a, b, c)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x = data_simulations.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y = data_simulations.iloc[:, 3:4].values

    random.seed(1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()
    scaler_test_x = StandardScaler()
    scaler_test_y = StandardScaler()

    u = scaler_test_x.fit_transform(x_test)
    x = scaler_train_x.fit_transform(x_train)
    y = scaler_train_y.fit_transform(y_train)
    v = scaler_test_y.fit_transform(y_test)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(u)

    #Treinamento da DT
    dt = ml.optimize_parameters(DecisionTreeRegressor(), x, y, ml.definitions_algorithms('DT'))

    dt = DecisionTreeRegressor(criterion=dt['criterion'],
                               splitter=dt['splitter'])

    score, error, dt = ml.experiment_folds('dt', dt, x, y)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score)
    print("Média do erro -> ", error)
    print("Modelo ->", dt)
    '''

    #Pacote - 256

    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()

    x1 = scaler_train_x.fit_transform(data_x)
    y1 = scaler_train_y.fit_transform(data_y)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(x1)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y.inverse_transform(y1))
    simu1 = sorted(scaler_train_y.inverse_transform(y1))
    print("Valores da predição - DT -> \n", scaler_train_y.inverse_transform(dt.predict(x1)))
    print("Valores da predição ordenados - DT -> \n", sorted(scaler_train_y.inverse_transform(dt.predict(x1))))
    pred1 = sorted(scaler_train_y.inverse_transform(dt.predict(x1)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali = np.array(sorted([(x1[0],y1[0]) for x1,y1 in zip(set_test, simu1)]))
    predi1 = np.array(sorted([(x1[0],y1) for x1,y1 in zip(set_test, pred1)]))

    print('Resultados das métricas')
    print('Valor do Score - ', dt.score(x1, y1))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y1, dt.predict(x1))))

    icmin = np.array(icmin)
    icmax = np.array(icmax)

    min = vali[:,1]-icmin
    max = icmax-vali[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'dt', 1, a, c, min, max, vali, predi1, pred1, set_test)

    #Pacote de 512
    c2 = '512'

    data_simulations2, icmin2, icmax2 = de.data_formulation(n, flag, cam, a, b, c2)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x2 = data_simulations2.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y2 = data_simulations2.iloc[:, 3:4].values

    random.seed(1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(data_x2, data_y2, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()
    scaler_test_x2 = StandardScaler()
    scaler_test_y2 = StandardScaler()

    u2 = scaler_test_x2.fit_transform(x_test2)
    x2 = scaler_train_x2.fit_transform(x_train2)
    y2 = scaler_train_y2.fit_transform(y_train2)
    v2 = scaler_test_y2.fit_transform(y_test2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(u2)

    #Treinamento da DT
    dt2 = ml.optimize_parameters(DecisionTreeRegressor(), x2, y2, ml.definitions_algorithms('DT'))

    dt2 = DecisionTreeRegressor(criterion=dt2['criterion'],
                                splitter=dt2['splitter'])

    score2, error2, dt2 = ml.experiment_folds('dt', dt2, x2, y2)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score2)
    print("Média do erro -> ", error2)
    print("Modelo ->", dt2)
    '''

    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()

    x3 = scaler_train_x2.fit_transform(data_x2)
    y3 = scaler_train_y2.fit_transform(data_y2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(x3)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y2.inverse_transform(y3))
    simu2 = sorted(scaler_train_y2.inverse_transform(y3))
    print("Valores da predição - DT -> \n", scaler_train_y2.inverse_transform(dt2.predict(x3)))
    print("Valores da predição ordenados - DT -> \n", sorted(scaler_train_y2.inverse_transform(dt2.predict(x3))))
    pred2 = sorted(scaler_train_y2.inverse_transform(dt2.predict(x3)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali2 = np.array(sorted([(x3[0],y3[0]) for x3,y3 in zip(set_test2, simu2)]))
    predi2 = np.array(sorted([(x3[0],y3) for x3,y3 in zip(set_test2, pred2)]))

    print('Resultados das métricas')
    print('Valor do Score - ', dt2.score(x3, y3))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y3, dt2.predict(x3))))

    icmin2 = np.array(icmin2)
    icmax2 = np.array(icmax2)

    min2 = vali2[:,1]-icmin2
    max2 = icmax2-vali2[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'dt', 2, a, c2, min2, max2, vali2, predi2, pred2, set_test2)

    #Pacote de 1024
    c3 = '1024'

    data_simulations3, icmin3, icmax3 = de.data_formulation(n, flag, cam, a, b, c3)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x3 = data_simulations3.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y3 = data_simulations3.iloc[:, 3:4].values

    random.seed(1)
    x_train3, x_test3, y_train3, y_test3 = train_test_split(data_x3, data_y3, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()
    scaler_test_x3 = StandardScaler()
    scaler_test_y3 = StandardScaler()

    u3 = scaler_test_x3.fit_transform(x_test3)
    x3 = scaler_train_x3.fit_transform(x_train3)
    y3 = scaler_train_y3.fit_transform(y_train3)
    v3 = scaler_test_y3.fit_transform(y_test3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(u3)

    #Treinamento da DT
    dt3 = ml.optimize_parameters(DecisionTreeRegressor(), x3, y3, ml.definitions_algorithms('DT'))

    dt3 = DecisionTreeRegressor(criterion=dt3['criterion'],
                                splitter=dt3['splitter'])

    score3, error3, dt3 = ml.experiment_folds('dt', dt3, x3, y3)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score3)
    print("Média do erro -> ", error3)
    print("Modelo ->", dt3)
    '''

    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()

    x4 = scaler_train_x3.fit_transform(data_x3)
    y4 = scaler_train_y3.fit_transform(data_y3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(x4)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y3.inverse_transform(y4))
    simu3 = sorted(scaler_train_y3.inverse_transform(y4))
    print("Valores da predição - DT -> \n", scaler_train_y3.inverse_transform(dt3.predict(x4)))
    print("Valores da predição ordenados - DT -> \n", sorted(scaler_train_y3.inverse_transform(dt3.predict(x4))))
    pred3 = sorted(scaler_train_y3.inverse_transform(dt3.predict(x4)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali3 = np.array(sorted([(x4[0],y4[0]) for x4,y4 in zip(set_test3, simu3)]))
    predi3 = np.array(sorted([(x4[0],y4) for x4,y4 in zip(set_test3, pred3)]))

    print('Resultados das métricas')
    print('Valor do Score - ', dt3.score(x4, y4))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y4, dt3.predict(x4))))

    icmin3 = np.array(icmin3)
    icmax3 = np.array(icmax3)

    min3 = vali3[:,1]-icmin3
    max3 = icmax3-vali3[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'dt', 3, a, c3, min3, max3, vali3, predi3, pred3, set_test3)

    rg.create_graphic_interval_confidence('Taxa de entrega %', 'dt', 3, a, min, max, vali, predi1, pred1, set_test,
                                                                   min2, max2, vali2, predi2, pred2, set_test2,
                                                                   min3, max3, vali3, predi3, pred3, set_test3)

def rf():
    print('RF')
    #Resultados para a taxa de entrega
    n = 30
    flag = 1
    cam = "C:/Users/Joyce Quintino/joyce/dados-tcc/cenario1 - v4/"
    cam2 = "C:/Users/joyce/Documents/joyce/programsimulations/extraction_data/"

    a = '0.1'
    b = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', 
         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', 
         '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', 
         '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', 
         '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', 
         '105', '106', '107', '108', '109']
    c = '256'

    data_simulations, icmin, icmax = de.data_formulation(n, flag, cam, a, b, c)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x = data_simulations.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y = data_simulations.iloc[:, 3:4].values

    random.seed(1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()
    scaler_test_x = StandardScaler()
    scaler_test_y = StandardScaler()

    u = scaler_test_x.fit_transform(x_test)
    x = scaler_train_x.fit_transform(x_train)
    y = scaler_train_y.fit_transform(y_train)
    v = scaler_test_y.fit_transform(y_test)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(u)

    #Treinamento da RF
    rf = ml.optimize_parameters(RandomForestRegressor(), x, y, ml.definitions_algorithms('RF'))

    rf = RandomForestRegressor(criterion=rf['criterion'],
                               n_estimators=rf['n_estimators'])

    score, error, rf = ml.experiment_folds('rf', rf, x, y)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score)
    print("Média do erro -> ", error)
    print("Modelo ->", rf)
    '''

    #Pacote - 256

    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()

    x1 = scaler_train_x.fit_transform(data_x)
    y1 = scaler_train_y.fit_transform(data_y)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(x1)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y.inverse_transform(y1))
    simu1 = sorted(scaler_train_y.inverse_transform(y1))
    print("Valores da predição - RF -> \n", scaler_train_y.inverse_transform(rf.predict(x1)))
    print("Valores da predição ordenados - RF -> \n", sorted(scaler_train_y.inverse_transform(rf.predict(x1))))
    pred1 = sorted(scaler_train_y.inverse_transform(rf.predict(x1)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali = np.array(sorted([(x1[0],y1[0]) for x1,y1 in zip(set_test, simu1)]))
    predi1 = np.array(sorted([(x1[0],y1) for x1,y1 in zip(set_test, pred1)]))

    print('Resultados das métricas')
    print('Valor do Score - ', rf.score(x1, y1))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y1, rf.predict(x1))))

    icmin = np.array(icmin)
    icmax = np.array(icmax)

    min = vali[:,1]-icmin
    max = icmax-vali[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'rf', 1, a, c, min, max, vali, predi1, pred1, set_test)

    #Pacote de 512
    c2 = '512'

    data_simulations2, icmin2, icmax2 = de.data_formulation(n, flag, cam, a, b, c2)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x2 = data_simulations2.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y2 = data_simulations2.iloc[:, 3:4].values

    random.seed(1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(data_x2, data_y2, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()
    scaler_test_x2 = StandardScaler()
    scaler_test_y2 = StandardScaler()

    u2 = scaler_test_x2.fit_transform(x_test2)
    x2 = scaler_train_x2.fit_transform(x_train2)
    y2 = scaler_train_y2.fit_transform(y_train2)
    v2 = scaler_test_y2.fit_transform(y_test2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(u2)

    #Treinamento da RF
    rf2 = ml.optimize_parameters(RandomForestRegressor(), x2, y2, ml.definitions_algorithms('RF'))

    rf2 = RandomForestRegressor(criterion=rf2['criterion'],
                               n_estimators=rf2['n_estimators'])

    score2, error2, rf2 = ml.experiment_folds('rf', rf2, x2, y2)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score2)
    print("Média do erro -> ", error2)
    print("Modelo ->", rf2)
    '''

    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()

    x3 = scaler_train_x2.fit_transform(data_x2)
    y3 = scaler_train_y2.fit_transform(data_y2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(x3)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y2.inverse_transform(y3))
    simu2 = sorted(scaler_train_y2.inverse_transform(y3))
    print("Valores da predição - RF -> \n", scaler_train_y2.inverse_transform(rf2.predict(x3)))
    print("Valores da predição ordenados - RF -> \n", sorted(scaler_train_y2.inverse_transform(rf2.predict(x3))))
    pred2 = sorted(scaler_train_y2.inverse_transform(rf2.predict(x3)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali2 = np.array(sorted([(x3[0],y3[0]) for x3,y3 in zip(set_test2, simu2)]))
    predi2 = np.array(sorted([(x3[0],y3) for x3,y3 in zip(set_test2, pred2)]))

    print('Resultados das métricas')
    print('Valor do Score - ', rf2.score(x3, y3))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y3, rf2.predict(x3))))

    icmin2 = np.array(icmin2)
    icmax2 = np.array(icmax2)

    min2 = vali2[:,1]-icmin2
    max2 = icmax2-vali2[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'rf', 2, a, c2, min2, max2, vali2, predi2, pred2, set_test2)

    #Pacote de 1024
    c3 = '1024'

    data_simulations3, icmin3, icmax3 = de.data_formulation(n, flag, cam, a, b, c3)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x3 = data_simulations3.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y3 = data_simulations3.iloc[:, 3:4].values

    random.seed(1)
    x_train3, x_test3, y_train3, y_test3 = train_test_split(data_x3, data_y3, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()
    scaler_test_x3 = StandardScaler()
    scaler_test_y3 = StandardScaler()

    u3 = scaler_test_x3.fit_transform(x_test3)
    x3 = scaler_train_x3.fit_transform(x_train3)
    y3 = scaler_train_y3.fit_transform(y_train3)
    v3 = scaler_test_y3.fit_transform(y_test3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(u3)

    #Treinamento da RF
    rf3 = ml.optimize_parameters(RandomForestRegressor(), x3, y3, ml.definitions_algorithms('RF'))

    rf3 = RandomForestRegressor(criterion=rf3['criterion'],
                               n_estimators=rf3['n_estimators'])

    score3, error3, rf3 = ml.experiment_folds('rf', rf3, x3, y3)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score3)
    print("Média do erro -> ", error3)
    print("Modelo ->", rf3)
    '''

    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()

    x4 = scaler_train_x3.fit_transform(data_x3)
    y4 = scaler_train_y3.fit_transform(data_y3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(x4)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y3.inverse_transform(y4))
    simu3 = sorted(scaler_train_y3.inverse_transform(y4)) 
    print("Valores da predição - RF -> \n", scaler_train_y3.inverse_transform(rf3.predict(x4)))
    print("Valores da predição ordenados - RF -> \n", sorted(scaler_train_y3.inverse_transform(rf3.predict(x4))))
    pred3 = sorted(scaler_train_y3.inverse_transform(rf3.predict(x4)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali3 = np.array(sorted([(x4[0],y4[0]) for x4,y4 in zip(set_test3, simu3)]))
    predi3 = np.array(sorted([(x4[0],y4) for x4,y4 in zip(set_test3, pred3)]))

    print('Resultados das métricas')
    print('Valor do Score - ', rf3.score(x4, y4))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y4, rf3.predict(x4))))

    icmin3 = np.array(icmin3)
    icmax3 = np.array(icmax3)

    min3 = vali3[:,1]-icmin3
    max3 = icmax3-vali3[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'rf', 3, a, c3, min3, max3, vali3, predi3, pred3, set_test3)

    rg.create_graphic_interval_confidence('Taxa de entrega %', 'rf', 3, a, min, max, vali, predi1, pred1, set_test,
                                                                    min2, max2, vali2, predi2, pred2, set_test2,
                                                                    min3, max3, vali3, predi3, pred3, set_test3)

def svr():
    print('SVR')
    #Resultados para a taxa de entrega
    n = 30
    flag = 1
    cam = "C:/Users/Joyce Quintino/joyce/dados-tcc/cenario1 - v4/"
    cam2 = "C:/Users/joyce/Documents/joyce/programsimulations/extraction_data/"

    a = '0.1'
    b = ['50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', 
         '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', 
         '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', 
         '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', 
         '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', 
         '105', '106', '107', '108', '109']
    c = '256'

    data_simulations, icmin, icmax = de.data_formulation(n, flag, cam, a, b, c)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x = data_simulations.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y = data_simulations.iloc[:, 3:4].values

    random.seed(1)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()
    scaler_test_x = StandardScaler()
    scaler_test_y = StandardScaler()

    u = scaler_test_x.fit_transform(x_test)
    x = scaler_train_x.fit_transform(x_train)
    y = scaler_train_y.fit_transform(y_train)
    v = scaler_test_y.fit_transform(y_test)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(u)

    #Treinamento da SVR
    svr = ml.optimize_parameters(SVR(), x, y, ml.definitions_algorithms('SVR'))

    svr = SVR(kernel=svr['kernel'],
              gamma=svr['gamma'],
              C=svr['C'])

    score, error, svr = ml.experiment_folds('svr', svr, x, y)
    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score)
    print("Média do erro -> ", error)
    print("Modelo ->", svr)
    '''

    #Pacote - 256

    scaler_train_x = StandardScaler()
    scaler_train_y = StandardScaler()

    x1 = scaler_train_x.fit_transform(data_x)
    y1 = scaler_train_y.fit_transform(data_y)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca = PCA(n_components=1)
    set_test = pca.fit_transform(x1)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y.inverse_transform(y1))
    simu1 = sorted(scaler_train_y.inverse_transform(y1))
    print("Valores da predição - SVR -> \n", scaler_train_y.inverse_transform(svr.predict(x1)))
    print("Valores da predição ordenados - SVR -> \n", sorted(scaler_train_y.inverse_transform(svr.predict(x1))))
    pred1 = sorted(scaler_train_y.inverse_transform(svr.predict(x1)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali = np.array(sorted([(x1[0],y1[0]) for x1,y1 in zip(set_test, simu1)]))
    predi1 = np.array(sorted([(x1[0],y1) for x1,y1 in zip(set_test, pred1)]))

    print('Resultados das métricas')
    print('Valor do Score - ', svr.score(x1, y1))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y1, svr.predict(x1))))

    icmin = np.array(icmin)
    icmax = np.array(icmax)

    min = vali[:,1]-icmin
    max = icmax-vali[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'svm', 1, a, c, min, max, vali, predi1, pred1, set_test)

    #Pacote de 512
    c2 = '512'

    data_simulations2, icmin2, icmax2 = de.data_formulation(n, flag, cam, a, b, c2)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x2 = data_simulations2.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y2 = data_simulations2.iloc[:, 3:4].values

    random.seed(1)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(data_x2, data_y2, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()
    scaler_test_x2 = StandardScaler()
    scaler_test_y2 = StandardScaler()

    u2 = scaler_test_x2.fit_transform(x_test2)
    x2 = scaler_train_x2.fit_transform(x_train2)
    y2 = scaler_train_y2.fit_transform(y_train2)
    v2 = scaler_test_y2.fit_transform(y_test2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(u2)

    #Treinamento da SVR
    svr2 = ml.optimize_parameters(SVR(), x2, y2, ml.definitions_algorithms('SVR'))

    svr2 = SVR(kernel=svr2['kernel'],
               gamma=svr2['gamma'],
               C=svr2['C'])

    score2, error2, svr2 = ml.experiment_folds('svr', svr2, x2, y2)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score2)
    print("Média do erro -> ", error2)
    print("Modelo ->", svr2)
    '''

    scaler_train_x2 = StandardScaler()
    scaler_train_y2 = StandardScaler()

    x3 = scaler_train_x2.fit_transform(data_x2)
    y3 = scaler_train_y2.fit_transform(data_y2)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca2 = PCA(n_components=1)
    set_test2 = pca2.fit_transform(x3)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y2.inverse_transform(y3))
    simu2 = sorted(scaler_train_y2.inverse_transform(y3))
    print("Valores da predição - SVR -> \n", scaler_train_y2.inverse_transform(svr2.predict(x3)))
    print("Valores da predição ordenados- SVR -> \n", sorted(scaler_train_y2.inverse_transform(svr2.predict(x3))))
    pred2 = sorted(scaler_train_y2.inverse_transform(svr2.predict(x3)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali2 = np.array(sorted([(x3[0],y3[0]) for x3,y3 in zip(set_test2, simu2)]))
    predi2 = np.array(sorted([(x3[0],y3) for x3,y3 in zip(set_test2, pred2)]))

    print('Resultados das métricas')
    print('Valor do Score - ', svr2.score(x3, y3))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y3, svr2.predict(x3))))

    icmin2 = np.array(icmin2)
    icmax2 = np.array(icmax2)

    min2 = vali2[:,1]-icmin2
    max2 = icmax2-vali2[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'svm', 2, a, c2, min2, max2, vali2, predi2, pred2, set_test2)

    #Pacote de 1024
    c3 = '1024'

    data_simulations3, icmin3, icmax3 = de.data_formulation(n, flag, cam, a, b, c3)

    # x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
    data_x3 = data_simulations3.iloc[:, 0:3].values
    # y representa o atributo alvo de x
    data_y3 = data_simulations3.iloc[:, 3:4].values

    random.seed(1)
    x_train3, x_test3, y_train3, y_test3 = train_test_split(data_x3, data_y3, test_size = 0.2, random_state = 0)

    # Escalonamento de dados
    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()
    scaler_test_x3 = StandardScaler()
    scaler_test_y3 = StandardScaler()

    u3 = scaler_test_x3.fit_transform(x_test3)
    x3 = scaler_train_x3.fit_transform(x_train3)
    y3 = scaler_train_y3.fit_transform(y_train3)
    v3 = scaler_test_y3.fit_transform(y_test3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(u3)

    #Treinamento da SVR
    svr3 = ml.optimize_parameters(SVR(), x3, y3, ml.definitions_algorithms('SVR'))

    svr3 = SVR(kernel=svr3['kernel'],
               gamma=svr3['gamma'],
               C=svr3['C'])

    score3, error3, svr3 = ml.experiment_folds('svr', svr3, x3, y3)

    '''
    print("Resultados após treinamento dos algoritmos")
    print("Média do score -> ", score3)
    print("Média do erro -> ", error3)
    print("Modelo ->", svr3)
    '''

    scaler_train_x3 = StandardScaler()
    scaler_train_y3 = StandardScaler()

    x4 = scaler_train_x3.fit_transform(data_x3)
    y4 = scaler_train_y3.fit_transform(data_y3)

    #Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
    pca3 = PCA(n_components=1)
    set_test3 = pca3.fit_transform(x4)

    #Carregamento do modelo
    print("Valores da simulação - conjunto de validação -> \n", scaler_train_y3.inverse_transform(y4))
    simu3 = sorted(scaler_train_y3.inverse_transform(y4))
    print("Valores da predição - SVR -> \n", scaler_train_y3.inverse_transform(svr3.predict(x4)))
    print("Valores da predição ordenados- SVR -> \n", sorted(scaler_train_y3.inverse_transform(svr3.predict(x4))))
    pred3 = sorted(scaler_train_y3.inverse_transform(svr3.predict(x4)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali3 = np.array(sorted([(x4[0],y4[0]) for x4,y4 in zip(set_test3, simu3)]))
    predi3 = np.array(sorted([(x4[0],y4) for x4,y4 in zip(set_test3, pred3)]))

    print('Resultados das métricas')
    print('Valor do Score - ', svr3.score(x4, y4))
    print('Valor da RMSE - ', np.sqrt(mean_squared_error(y4, svr3.predict(x4))))

    icmin3 = np.array(icmin3)
    icmax3 = np.array(icmax3)

    min3 = vali3[:,1]-icmin3
    max3 = icmax3-vali3[:,1]

    rg.create_graphic_interval_confidence2('Taxa de entrega %', 'svm', 3, a, c3, min3, max3, vali3, predi3, pred3, set_test3)

    rg.create_graphic_interval_confidence('Taxa de entrega %', 'svm', 3, a, min, max, vali, predi1, pred1, set_test,
                                                                    min2, max2, vali2, predi2, pred2, set_test2,
                                                                    min3, max3, vali3, predi3, pred3, set_test3)