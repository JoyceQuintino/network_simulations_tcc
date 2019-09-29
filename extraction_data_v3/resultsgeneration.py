import matplotlib.pyplot as plt
import seaborn as sns
import machinelearning as ml
import numpy as np
import plotly.graph_objects as go

def create_graphic(mod, med, set_test, simu, pred, predi, vali):
    plt.scatter(set_test, simu)
    plt.scatter(set_test, pred)
    plt.title('Simulação x Predição - Modelo: '+mod)
    plt.plot(vali[:,0], vali[:,1], color='blue')
    plt.plot(predi[:,0], predi[:,1], color='red')
    plt.legend(('Simulação', 'Predição'), loc='lower left')
    plt.xlabel('Principal componente')
    plt.ylabel(med)
    plt.savefig('result_'+mod+'.png')

def create_graphic_interval_confidence(med, name_model, num, name_inter, min, max, vali, predi1, pred1, set_test,
                                                        min2, max2, vali2, predi2, pred2, set_test2,
                                                        min3, max3, vali3, predi3, pred3, set_test3):
    yone100 = []

    for i in range(0, 130, 20):
        yone100.append(i)

    x = []

    for i in range(-1, 3):
        x.append(i)

    fig1 = plt.figure(figsize=(15, 9))

    yerror = [min, max]
    yerror2 = [min2, max2]
    yerror3 = [min3, max3]

    line,caps,bars = plt.errorbar(
        vali[:,0],
        vali[:,1],
        yerr=yerror,
        fmt='^-',
        linewidth=2,
        elinewidth=1.5,
        ecolor='blue',
        capsize=7,
        capthick=5
    )

    line2,caps,bars = plt.errorbar(
        vali2[:,0],
        vali2[:,1],
        yerr=yerror2,
        fmt='^-',
        linewidth=2,
        elinewidth=1.5,
        ecolor='red',
        capsize=7,
        capthick=5
    )

    line3,caps,bars = plt.errorbar(
        vali3[:,0],
        vali3[:,1],
        yerr=yerror3,
        fmt='^-',
        linewidth=2,
        elinewidth=1.5,
        ecolor='green',
        capsize=7,
        capthick=5
    )

    plt.scatter(set_test, pred1)
    plt.scatter(set_test2, pred2)
    plt.scatter(set_test3, pred3)
    #plt.title('Simulação x Predição - Modelo: '+name_model+'\nIntervalo entre pacotes: '+name_inter, fontsize=18)
    plt.title('Intervalo entre pacotes: '+name_inter, fontsize=26)
    plt.plot(predi1[:,0], predi1[:,1], 'o-', color='blue', markersize=10)
    plt.plot(predi2[:,0], predi2[:,1], 'o-', color='red', markersize=10)
    plt.plot(predi3[:,0], predi3[:,1], 'o-', color='green', markersize=10)
    plt.setp(line,label="Simulação", color='blue', markersize=10)
    plt.setp(line2,label="Simulação", color='red', markersize=10)
    plt.setp(line3,label="Simulação", color='green', markersize=10)
    plt.legend(('Valores reais da simulação - tamanho do pacote: 256',
                'Valores reais da simulação - tamanho do pacote: 512',
                'Valores reais da simulação - tamanho do pacote: 1024',
                'Valores da predição - tamanho do pacote: 256', 
                'Valores da predição - tamanho do pacote: 512', 
                'Valores da predição - tamanho do pacote: 1024'), 
                loc=('lower left'), fontsize=16)
    plt.xlim((-1.3,1.9))
    plt.xticks(x, fontsize=26)
    plt.yticks(yone100, fontsize=26)
    plt.xlabel('Principal componente', fontsize=26)
    plt.ylabel(med, fontsize=26)
    plt.savefig(name_model+str(num)+'.png')

def create_graphic_interval_confidence2(med, name_model, num, name_inter, tam_pac, min, max, vali, predi1, pred1, set_test):

    yone100 = []

    for i in range(0, 130, 20):
        yone100.append(i)

    x = []

    for i in range(-1, 3):
        x.append(i)

    fig1 = plt.figure(figsize=(15, 9))

    yerror = [min, max]

    line,caps,bars = plt.errorbar(
        vali[:,0],
        vali[:,1],
        yerr=yerror,
        fmt='^-',
        linewidth=2,
        elinewidth=1.5,
        ecolor='red',
        capsize=7,
        capthick=5
    )

    plt.scatter(set_test, pred1)
    #plt.title('Simulação x Predição - Modelo: '+name_model+'\nIntervalo entre pacotes: '+name_inter, fontsize=18)
    plt.title('Intervalo entre pacotes: '+name_inter, fontsize=26)
    plt.plot(predi1[:,0], predi1[:,1], 'o-', color='blue', markersize=10)
    plt.setp(line,label="Simulação", color='red', markersize=10)
    plt.legend(('Valores reais da simulação - tamanho do pacote: '+tam_pac,
                'Valores da predição - tamanho do pacote: '+tam_pac), 
                loc=('lower left'), fontsize=16)
    plt.xlim((-1.3,1.9))
    plt.xticks(x, fontsize=26)
    plt.yticks(yone100, fontsize=26)
    plt.xlabel('Principal componente', fontsize=26)
    plt.ylabel(med, fontsize=26)
    plt.savefig(name_model+'_'+tam_pac+'_'+str(num)+'.png')