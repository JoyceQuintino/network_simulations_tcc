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
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

#Resultados para a taxa de entrega
u = 30
flag = 1
cam = "C:/Users/joyce/Documents/joyce/dados-tcc/"
cam2 = "C:/Users/joyce/Documents/joyce/programsimulations/extraction_data/"

a = '0.01'
b = ['50', '60', '70', '80', '90', '95', '100']
c = '256'

data_simulations, icmin, icmax = de.data_formulation(u, flag, cam, a, b, c)

print(icmin)
print(icmax)

# x representa o dataset de treino e possui 23 colunas sem o atributo alvo 
data_x = data_simulations.iloc[:, 0:3].values
# y representa o atributo alvo de x
data_y = data_simulations.iloc[:, 3:4].values

random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size = 0.2, random_state = 0)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape) 
print(y_test.shape)

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
set_test

#Treinamento da MLP
mlp = ml.optimize_parameters(MLPRegressor(), x, y, ml.definitions_algorithms('MLP'))

mlp = MLPRegressor(alpha=mlp['alpha'],
                   activation=mlp['activation'], 
                   learning_rate_init=mlp['learning_rate_init'], 
                   solver=mlp['solver'], 
                   hidden_layer_sizes=mlp['hidden_layer_sizes'])

score, error, mlp = ml.experiment_folds2('mlp', mlp, x, y)

print("Resultados após treinamento dos algoritmos")
print("Média do score -> ", score)
print("Média do erro -> ", error)
print("Modelo ->", mlp)

#Pacote - 256

scaler_train_x = StandardScaler()
scaler_train_y = StandardScaler()

x1 = scaler_train_x.fit_transform(data_x)
y1 = scaler_train_y.fit_transform(data_y)

#Uso de PCA no conjunto de teste para exibicao dos resultados de forma grafica
pca = PCA(n_components=1)
set_test = pca.fit_transform(x1)
set_test

#Carregamento do modelo
print("Valores da simulação - conjunto de validação -> \n", scaler_train_y.inverse_transform(y1))
simu1 = sorted(scaler_train_y.inverse_transform(y1))
print("Valores da predição - MLP -> \n", sorted(scaler_train_y.inverse_transform(mlp.predict(x1))))
pred1 = sorted(scaler_train_y.inverse_transform(mlp.predict(x1)))

#Comparação entre valores da simulação e valores preditos - Taxa de entrega
vali = np.array(sorted([(x1[0],y1[0]) for x1,y1 in zip(set_test, simu1)]))
predi1 = np.array(sorted([(x1[0],y1) for x1,y1 in zip(set_test, pred1)]))

vali = np.array(sorted([(x1[0],y1[0]) for x1,y1 in zip(set_test, simu1)]))

#print('\nIntervalo de confianca') 

#ml.interval_confidence(vali)

rg.create_graphic_interval_confidence('Taxa de entrega', 'mlp', icmin, icmax, vali, predi1, pred1, set_test)
#rg.teste(icmin, icmax, vali)