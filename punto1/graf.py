import numpy as np
import matplotlib.pyplot as plt

M=8 #cantidad de cadenas

means=np.zeros(M) #guarda los promedios
stdvs=np.zeros(M) # guarda las desviaciones
for i in range(M):
    data=np.loadtxt(str(i))
    mean_est=np.mean(data)
    std_est=np.std(data)
    
    N=len(data)
    
    means[i]=mean_est
    stdvs[i]=std_est
    plt.hist(data, bins=60,density=True)
    plt.title('Num ='+str(len(data))+' stdev='+str(std_est) + ' mu='+str(mean_est), alpha=0.6)
    plt.savefig('histograms')

B= (N/(M-1))*np.sum((np.mean(means)-means)**2)
W=(1/M)*np.sum(stdvs**2)

V= ((N-1)/N)*W + ((M+1)/(M*N))* B

print('El factor Gelman-Rubin es: ',V)