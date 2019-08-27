import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
from collections import OrderedDict

#leitura de arquivos e conversao de tipos
def read_file_convert(path):
    file = open(path, 'r')
    text = file.readlines()
    file.close()
    simulations_data = []
    val_data = []
    dic_data = {}
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
        result.append(value.strip(' s\n\nbit/s\t%byteMbps'))
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
    remove_caracteres(dataset, 'DelayMean')
    remove_caracteres(dataset, 'DelaySum')
    remove_caracteres(dataset, 'DeliveryRate')
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

def selection_variables(dataset):
    data = OrderedDict(
    {
        'distanceBetweenNode': dataset['DistanceBetweenNode'],
        'packetInterval': dataset['PacketInterval'],
        'packetSize': dataset['packetSize'],
        'deliveryRate': dataset['DeliveryRate']
    })

    df = pd.DataFrame(data)

    df['distanceBetweenNode'] = df['distanceBetweenNode'].astype('int64')
    df['packetSize'] = df['packetSize'].astype('int64')
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
        'distanceBetweenNode': df2[0],
        'packetInterval': df2[1],
        'packetSize': df2[2],
        'deliveryRate': sum(df.iloc[:,3].values)//df['deliveryRate'].count()
    })
    new_data_aux = pd.DataFrame(data, index=[i])
    return new_data_aux

def data_formulation(u, flag, cam, a, b, c):
    i = 1
    data_simulations = create_dataset(u, a, b, c, flag, cam, 0)
    while(i < len(b)):
        dataset_aux = create_dataset(u, a, b, c, flag, cam, i)
        data_simulations = pd.concat([data_simulations, dataset_aux])
        i += 1
    return data_simulations