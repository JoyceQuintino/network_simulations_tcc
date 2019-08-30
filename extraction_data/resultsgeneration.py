import matplotlib.pyplot as plt
import seaborn as sns
import machinelearning as ml
import numpy as np

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

def create_graphic_interval_confidence(med, name_model, yerrormin, yerrormax, vali, predi1, pred1, set_test):
    yone100 = []

    for i in range(0, 115, 5):
        yone100.append(i)

    x = []

    for i in range(-1, 3):
        x.append(i)

    fig1 = plt.figure(figsize=(15, 9))
    std = np.std(vali)

    yerror = [yerrormin, yerrormax]

    line,caps,bars = plt.errorbar(
        vali[:,0],
        vali[:,1],
        yerr= yerror,
        fmt="rs-",
        linewidth=1,
        elinewidth=0.5,
        ecolor='k',
        capsize=5,
        capthick=0.5,
    )

    plt.scatter(set_test, pred1)
    plt.title('Simulação x Predição - Modelo: '+name_model)
    plt.plot(predi1[:,0], predi1[:,1], color='blue')
    plt.setp(line,label="Simulação")
    plt.legend(('Simulação', 'Predição'), loc=('lower center'))
    plt.xlim((-1.3,1.9))
    plt.xticks(x)
    plt.yticks(yone100)
    plt.xlabel('Principal componente')
    plt.ylabel(med)
    plt.savefig(name_model+'.png')

def teste(yerrormin, yerrormax, vali):
    yerror = [yerrormin, yerrormax]
    plt.errorbar(vali[:,0], vali[:,1], yerr= yerror, fmt="o")
    plt.savefig('teste.png')