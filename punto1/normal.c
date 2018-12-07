#include<stdio.h> 
#include<omp.h>
#include <stdlib.h>
#include <math.h>

double func(float x){
    float sigma;
    float xx;
    sigma=1.0;
    
    xx=(1/sqrt(2*sigma))*exp(x/pow(sigma,2.0));
    return xx;
    
}

int main(int argc, char *argv[]){
    int N=1000;
    float step = 0.5; // step del montecarlo
#pragma omp parallel 
{ 
   //variables
    float samples[N]; //muestras de la normal
    float x_old; 
    float x_new;
    float a; //coeficiente de la condicion metropolis
    float normal_new; //nueva muestra
    float normal_old; // anterior muestra
    float r; //el factor aleatoreo comparar en MCMC
    int name; //nombre del archivo 
    int thread_id = omp_get_thread_num(); 
    int thread_count = omp_get_num_threads(); 
    int i;
    char name_str[12];
    
    FILE *fsamples;

    
    //inicializa
    name = thread_id;
    //pasa el name a string
    sprintf(name_str, "%d", name);
    
    //algoritmo metropolis hastings
    
    /*primera propuesta*/
    x_old= ((double) rand () / (double)RAND_MAX); 
    samples[0] = ((double) rand () / (double)RAND_MAX);
    
    for(i=0;i<N;i++){
        x_old = samples[i];
        x_new=x_old + step*(((double) rand () / (double)RAND_MAX)-0.5);
        normal_new = func(x_new);
        normal_old = func(x_old);
        a= normal_new/normal_old;
        //acepta
        if(a>=1){
            samples[i+1]=x_new;
        }
        //prueba otra vez
        else{
            r=((double) rand () / (double)RAND_MAX);
            //acepta
            if(r<a){
                samples[i+1]=x_new;
            }
            //rechaza
            else{
                samples[i+1]=x_old;
            }
        }   
    }
    
    //crea los archivos
    fsamples = fopen(name_str, "w+"); //opens the stream
	
	for(i=0; i<N; i++){
		fprintf(fsamples, "%f \n", samples[i]);
	}
	
	fclose(fsamples); //closes the stream
 } 
    
 
return 0;
}