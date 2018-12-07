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

B=np.ones(N)
W=np.ones(N)
figue1=plt.figure()
for i in range(1,N): 
    B[i]= (i/(M-1))*np.sum((np.mean(means[:i])-means[:i])**2)
    W[i]=(1/M)*np.sum(stdvs[:i]**2)

V= ((N-1)/N)*W + ((M+1)/(M*N))* B

plt.plot(V)
figue1.savefig('gr.png')