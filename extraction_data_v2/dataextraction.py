import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import random
import machinelearning as ml
import numpy as np
from collections import OrderedDict

#dicionário auxiliar para casos em que o fluxo não foi iniciado
'''
dic_aux = {'DeliveryRate':'0.0 %','Throughput':'0.0 Mbps','TxBytes':'0.0','RxBytes':'0.0',
               'TxPackets':'0.0','RxPackets':'0.0','LostPackets':'0.0','DelaySum':'0.0 s','DelayMean':'0.0 s',
               'JitterSum':'0.0 s','JitterMean':'0.0 s','TimeFirstTxPacket':'0.0 s','TimeLastTxPacket':'0.0 s',
               'TimeFirstRxPacket':'0.0 s','TimeLastRxPacket':'0.0 s','MeanTransmittedPacketSize':'0.0 byte','MeanTransmittedBitrate':'0.0 bit/s','MeanHopCount':'0.0',
               'PacketLossRatio':'0.0','MeanReceivedPacketSize':'0.0 byte','MeanReceivedBitrate':'0.0 bit/s'}
'''
#leitura de arquivos e conversão de tipos
def read_file_convert(path):
    path2 = 'C:/Users/Joyce Quintino/joyce/dados-tcc/cenario1 - v3/auxiliar/logmeshsimulation.txt'
    file = open(path, 'r')
    text = file.readlines()
    file.close()
    simulations_data = []
    dic_data = {}
    if text[9:10] == []:
        file = open(path2, 'r')
        text = file.readlines()
        file.close()
        for row in text[9:30]:
            simulations_data.append(row)
        for e in simulations_data:
            (key, val) = e.split(":")
            dic_data[str(key)] = val
        return dic_data
    else:
        for row in text[9:30]:
            simulations_data.append(row)
        for e in simulations_data:
            (key, val) = e.split(":")
            dic_data[str(key)] = val
        return dic_data

#limpeza dos dados retirando todos os caracteres diferentes de zeros
def remove_caracteres(dataset, name_col):
    result = []
    for value in dataset[name_col]:
        result.append(str(value).strip(' s\n\nbit/s\t%byteMbps'))
    dataset[name_col] = result

#faz a extracao de dados
def extraction_data(u, a, b, c, flag, cam):
    u += 1
    x = range(1,u)
    p1 = [a]
    p2 = [b]
    p3 = [c]
    data = []
    data_simulations = pd.DataFrame()
    
    for j in p2:
        for i in p1:
            for k in p3:
                for n in x:
                    s = cam+"{}-traces/step{}-packetInterval{}-packetSize{}/rodada{}/logMeshSimulation.txt".format(flag, j,i, k, n)
                    dataset = pd.DataFrame([read_file_convert(s)])
                    dataset['DistanceBetweenNode'] = j
                    dataset['PacketInterval'] = i
                    dataset['packetSize'] = k
                    data.append(dataset)
                data_simulations = pd.concat(data, ignore_index=True)
                data_simulations.fillna(0)
    return data_simulations

def transfor_columns(dataset):
    dataset.set_axis(['DeliveryRate',
                      'Throughput',
                      'TxBytes',
                      'RxBytes',
                      'TxPackets',
                      'RxPackets',
                      'LostPackets',
                      'DelaySum',
                      'DelayMean',
                      'JitterSum',
                      'JitterMean',
                      'TimeFirstTxPacket',
                      'TimeLastTxPacket',
                      'TimeFirstRxPacket',
                      'TimeLastRxPacket',
                      'MeanTransmittedPacketSize',
                      'MeanTransmittedBitrate',
                      'MeanHopCount',
                      'PacketLossRatio',
                      'MeanReceivedPacketSize',
                      'MeanReceivedBitrate',
                      'DistanceBetweenNode',
                      'PacketInterval',
                      'packetSize'], 
                       axis=1, inplace=True)
    return dataset

def clear_columns(dataset):
    remove_caracteres(dataset, 'DeliveryRate')
    remove_caracteres(dataset, 'DelayMean')
    remove_caracteres(dataset, 'DelaySum')
    remove_caracteres(dataset, 'JitterMean')
    remove_caracteres(dataset, 'JitterSum')
    remove_caracteres(dataset, 'TimeFirstRxPacket')
    remove_caracteres(dataset, 'TimeFirstTxPacket')
    remove_caracteres(dataset, 'TimeLastRxPacket')
    remove_caracteres(dataset, 'TimeLastTxPacket')
    remove_caracteres(dataset, 'LostPackets')
    remove_caracteres(dataset, 'MeanHopCount')
    remove_caracteres(dataset, 'MeanReceivedBitrate')
    remove_caracteres(dataset, 'MeanReceivedPacketSize')
    remove_caracteres(dataset, 'MeanTransmittedBitrate')
    remove_caracteres(dataset, 'MeanTransmittedPacketSize')
    remove_caracteres(dataset, 'PacketLossRatio')
    remove_caracteres(dataset, 'RxBytes')
    remove_caracteres(dataset, 'RxPackets')
    remove_caracteres(dataset, 'Throughput')
    remove_caracteres(dataset, 'TxBytes')
    remove_caracteres(dataset, 'TxPackets')

    return dataset

def convert_columns(dataset):
    dataset['LostPackets'] = dataset['LostPackets'].astype('int64')
    dataset['MeanHopCount'] = dataset['MeanHopCount'].astype('int64')
    dataset['TxBytes'] = dataset['TxBytes'].astype('int64')
    dataset['TxPackets'] = dataset['TxPackets'].astype('int64')
    dataset['DistanceBetweenNode'] = dataset['DistanceBetweenNode'].astype('int64')
    dataset['packetSize'] = dataset['packetSize'].astype('int64')
    dataset['MeanTransmittedPacketSize'] = dataset['MeanTransmittedPacketSize'].astype('int64')
    dataset['DeliveryRate'] = dataset['DeliveryRate'].astype('float')
    dataset['DelayMean'] = dataset['DelayMean'].astype('float')
    dataset['DelaySum'] = dataset['DelaySum'].astype('float')
    dataset['JitterMean'] = dataset['JitterSum'].astype('float')
    dataset['Throughput'] = dataset['Throughput'].astype('float')
    dataset['TimeFirstRxPacket'] = dataset['TimeFirstRxPacket'].astype('float')
    dataset['TimeFirstTxPacket'] = dataset['TimeFirstTxPacket'].astype('float')
    dataset['TimeLastRxPacket'] = dataset['TimeLastRxPacket'].astype('float')
    dataset['TimeLastTxPacket'] = dataset['TimeLastTxPacket'].astype('float')
    dataset['PacketInterval'] = dataset['PacketInterval'].astype('float')
    dataset['MeanReceivedBitrate'] = dataset['MeanReceivedBitrate'].astype('float')
    dataset['MeanReceivedPacketSize'] = dataset['MeanReceivedPacketSize'].astype('float')
    dataset['MeanTransmittedBitrate'] = dataset['MeanTransmittedBitrate'].astype('float')

    return dataset

def selection_variables(dataset):
    data = OrderedDict(
    {
        'distanceBetweenNode': dataset['DistanceBetweenNode'],
        'packetInterval': dataset['PacketInterval'],
        'packetSize': dataset['packetSize'],
        'deliveryRate': dataset['DeliveryRate']
    })

    df = pd.DataFrame(data)
    
    '''
    df['lostPackets'] = df['lostPackets'].astype('float')
    df['timeFirstTxPacket'] = df['timeFirstTxPacket'].astype('float')
    df['timeLastTxPacket'] = df['timeLastTxPacket'].astype('float')
    df['timeFirstRxPacket'] = df['timeFirstRxPacket'].astype('float')
    df['timeLastRxPacket'] = df['timeLastRxPacket'].astype('float')
    df['meanReceivedPacketSize'] = df['meanReceivedPacketSize'].astype('float')
    df['meanReceivedBitrate'] = df['meanReceivedBitrate'].astype('float')
    '''

    df['distanceBetweenNode'] = df['distanceBetweenNode'].astype('float')
    df['packetSize'] = df['packetSize'].astype('float')
    df['packetInterval'] = df['packetInterval'].astype('float')
    df['deliveryRate'] = df['deliveryRate'].astype('float')

    return df

def create_dataset(u, a, b, c, flag, cam, i):
    dataset = extraction_data(u, a, b[i], c, flag, cam)
    dataset = clear_columns(transfor_columns(dataset))
    df = selection_variables(dataset)
    df2 = df.iloc[0, 0:3].values
    data = OrderedDict(
    {
        '''
        'lostPackets': df2[0],
        'timeFirstTxPacket': df2[1],
        'timeLastTxPacket': df2[2],
        'timeFirstRxPacket': df2[3],
        'timeLastRxPacket': df2[4],
        'meanReceivedPacketSize': df2[5],
        'meanReceivedBitrate': df2[4],
        '''

        'distanceBetweenNode': df2[0],
        'packetInterval': df2[1],
        'packetSize': df2[2],
        'deliveryRate': sum(df.iloc[:,3].values)//df['deliveryRate'].count()
    })
    new_data_aux = pd.DataFrame(data, index=[i])
    return new_data_aux, [df.iloc[:, 3].values]

def data_formulation(u, flag, cam, a, b, c):
    i = 1
    icmin = []
    icmax = []
    data_simulations, data2 = create_dataset(u, a, b, c, flag, cam, 0)
    min, max = ml.confidence_interval(data2)
    icmin.append(min)
    icmax.append(max)
    while(i < len(b)):
        dataset_aux, data2 = create_dataset(u, a, b, c, flag, cam, i)
        min, max = ml.confidence_interval(data2)
        icmin.append(min)
        icmax.append(max)
        data_simulations = pd.concat([data_simulations, dataset_aux])
        i += 1
    return data_simulations, icmin, icmax

def create_datatest():
    n = 30
    flag = 1
    cam = "C:/Users/Joyce Quintino/joyce/dados-tcc/"
    cam2 = "C:/Users/joyce/Documents/joyce/programsimulations/extraction_data/"

    a = '0.01'
    b = ['50', '60', '70', '80', '90', '95', '100']
    c = '256'

    data_simulations, icmin, icmax = data_formulation(n, flag, cam, a, b, c)

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
    print("Valores da predição - MLP -> \n", sorted(scaler_train_y.inverse_transform(mlp.predict(x1))))
    pred1 = sorted(scaler_train_y.inverse_transform(mlp.predict(x1)))

    #Comparação entre valores da simulação e valores preditos - Taxa de entrega
    vali = np.array(sorted([(x1[0],y1[0]) for x1,y1 in zip(set_test, simu1)]))
    predi1 = np.array(sorted([(x1[0],y1) for x1,y1 in zip(set_test, pred1)]))

    icmin = np.array(icmin)
    icmax = np.array(icmax)

    min = vali[:,1]-icmin
    max = icmax-vali[:,1]

    return min, max, vali, set_test