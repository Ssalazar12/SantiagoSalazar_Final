
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = np.loadtxt('datos_observacionales.txt')
sig=1.0


# In[3]:


import numpy as np 
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3d

#condiciones inciales dadas por data[0,1:]
#metodo de euler y+1=y0+hf(x,y)

#obs[0] = x asociado a sigma , obs[1]=y asociado a sigma rho obs[2]=z asociado a beta

#define el tamaño de cada paso de la gradiente
step = 0.004
delta = 0.1 #tamaño del paso en el leapfrog
m=100.0

step_sig=10
step_rh=10
step_bet=10

def funx(sigma,obs):
    return sigma*(obs[1]-obs[0])

def funy(rh,obs):
    return obs[0]*(rh-obs[2])-obs[1]

def funz(bet,obs):
    return obs[0]*obs[1]-bet*obs[2]

#calcula el loglike, llama a modelo. Se asume una likelihood en forma de multiplicatoria-->log es una suma
#sig = sigma de los datos (error bars)
def loglike(derv,obs,param):
    #pram es el parametro a probar
    #obs son x y o lo que sea observados
    #es las derivada de la coordenana requerida por model
    modelo=np.asarray([funx(param[0],obs),funy(param[1],obs),funz(param[2],obs)])
    t = (derv-modelo)
    t=t/sig
    return -0.5*np.sum(t**2)

#saca el hamiltoniano
def H(p,derv,obs,param):
    m=100.0
    #model es la funcion que vamos a probar para cada param
    K = 0.5 * np.sum(p**2)/m
    #la distribucion que queremos samplear es la energia potencias
    #la posterior del modelo y el prior van aca
    U = -loglike(derv,obs, param)
    return K + U


#calcula la derivada respecto a cada parametro
def grad_loglike(derv,obs,param):
    #derv es un array con las derivadas en cada direccion
    ret_grad = np.zeros(3)
    #movemos sigma
    param_sig=param
    param_sig[0]=param_sig[0]+step
    #movemos rho
    param_rho=param
    param_rho[0]=param_rho[0]+step
    #movemos el beta
    param_beta=param
    param_beta[0]=param_beta[0]+step
    #el step me avanza al siguiente parametro como la definicion de derivada
    ret_grad[0]= (loglike(derv[0],obs, param_sig) - loglike(derv[0],obs, param))/step
    ret_grad[1] = (loglike(derv[1],obs, param_rho) - loglike(derv[1],obs, param))/step
    ret_grad[2] = (loglike(derv[2],obs, param_beta) - loglike(derv[2],obs, param))/step

    return ret_grad
            
#genera el nuevo punto por leapfrog hay un momento asociado a cada uno de los parametros
def leapfrog(derv,obs,param, p_in):
    #paramson los parametros
    #delta es el step del leapfrog
    #sig es el error que se le manda a la loglikelihood
    p_new = p_in 
    sig_new =np.copy(param[0])
    rh_new =np.copy(param[1])
    bet_new =np.copy(param[2])
        
    n_iter = 5

    
    for i in range(1,n_iter):
        #primer half-step
        p_new = p_in + 0.5 * delta * grad_loglike(derv,obs,param)  
        #para sacar el nuevo parametro, tenemos que actualizar con la masa param * m=p_param
        sig_new = sig_new + delta * p_new[0]/m
        rh_new = rh_new + delta *p_new[1]/m
        bet_new = bet_new + delta *p_new[2]/m
        #actualiza el momento con los nuevos parametros(el segundo half-step)
        p_new = p_new + 0.5 * delta * grad_loglike(derv,obs,param)
        ######
        #no olvidar el -1
        #####
        p_new = - p_new

    param_new=np.asarray([sig_new,rh_new,bet_new])
    
    return p_new, param_new


# In[4]:


def estimado(n_steps):
    sigma_param=0.4
    parametros=np.zeros((n_steps,3))
    p= np.zeros((n_steps,3))
    #primer elemento de mi array de logposterior
    #log_post=[loglike(x,z,sigma_z,a_e[0],b_e[0],c_e[0])]
    parametros[0,:]=np.random.normal(loc=0.0,scale=sigma_param,size=3)
    p[0]=np.random.normal(loc=0.0,scale=sigma_param,size=3)
   
    for i in range(1,n_steps):
        obs_old = data[i-1,1:]
        obs_new = data[i,1:]
        derivates = (obs_new-obs_old)/(data[i,0]-data[i-1,0])
        #generamos una propuesta por medio de leapfrog en vez de usar un numero aleatorio
        p_new, parametros_new = leapfrog(derivates,obs_new, parametros[i-1,:], p[i-1])
        #aca estamos sampleando la distribucion de energia con el hamiltoniano 
                #p,derv,obs,model,param
        E_new = H( p_new,derivates,obs_new,parametros_new) 
        E_old = H( p[i-1,:],derivates,obs_old, parametros[i-1,:])
        
        #Aca se aplica la condicion metropolis
         #recordemos que nuestra distribucion de proba es la distribucion de gibbs
        alpha = min(1,np.exp( - (E_new - E_old))) # Se comparan las dos energias
        
        beta = np.random.random()
         #acepta
        if(beta<alpha):
            parametros[i,:]=parametros_new
        #rechaza
        else:
            parametros[i,:]=parametros[i-1,:]

        #reiniciar el momento
        p[i]=(np.random.normal(size=3))
    
    return parametros


# In[6]:


parametros=estimado(len(data))


# In[10]:


plt.hist(parametros[0], bins=40)
plt.hist(parametros[1], bins=40)
plt.hist(parametros[2], bins=40)
plt.savefig('a.pdf')

