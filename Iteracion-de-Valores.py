import numpy as np

X = [[1,-1,-1], [1, -1,1], [1, 1,-1], [1, 1,1]] #Variables
y = [-1, -1, -1, 1]#valores que tiene que dar la compuerta AND
w = np.array([16,10,1])#Pesos
Res = [0,0,0,0]#variable donde voy a almacenar los resutados que predice la neurona
iteraciones=0
while Res !=y:  #Ejecuto la neurona hasta que la variable resultados sea igual a la salida de la compuerta AND
    for i in range(len(X)):     #Itero la predicion de la neurona y la actualizacion de los pesos 4 veces
        Y = np.sign(np.dot(w.T,X[i]))#Predicion de la variable 
        #print(Y)
        w = w + np.array(np.multiply(0.5*(y[i]-Y), X[i])) #Actualizar los pesos
        Res[i]=Y #Almaceno la predicion en la variable
        #print(w)
    iteraciones +=1 #le sumo 1 cada ves que se necesite volver a iterar una epoca
print("El resultado de la compureta es")
for i in Res:
    print(i)
    
print("A la neurona le tomo ",iteraciones," iteraciones aprender")
print("los pesos finales son",w)