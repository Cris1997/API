"""
Author: Cristian Rosales Deloya

"""

#Pandas, librería para el manejo de los dataframe dentro del programa
import pandas as pd
import operator
#Librerias de funciones matemáticas
import math
import numpy as np
import matplotlib.pyplot as plt
# from scipy.spatial import distance


# Ponderaciones para cada atributo de las 

w_variedad = 0.40
w_region = 0.05
w_tipo = 0.10
w_sabor = 0.05
w_aroma = 0.15
w_precio = 0.15
w_guarda = 0.10



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum() 

#Funcion que calcula la distancia euclidina entre dos vectores númericos
def distancia_euclidiana(x,y):
    return 1/(1+np.sqrt(np.sum((x - y) ** 2)))


# Leer el Dataframe de vinos
def abrir_datafrane():
    vinos_df = pd.read_csv('data_vinos.csv')
    return vinos_df


def clean_data(vinos_df):
    # Eliminar espacios en blanco para no tener redundancias en las categoria
    vinos_df['variedad'] = vinos_df.variedad.str.replace(" ","")
    vinos_df['region'] = vinos_df.region.str.replace(" ","")
    vinos_df['color'] = vinos_df.color.str.replace(" ","")
    vinos_df['sabor'] = vinos_df.sabor.str.replace(" ","")
    vinos_df['aroma'] = vinos_df.aroma.str.replace(" ","")
    return vinos_df

def estructurar_datos(vinos_df):
    vinos_df['variedad'] = vinos_df.variedad.str.split(',')
    vinos_df['aroma'] = vinos_df.aroma.str.split(',')
    vinos_df['region'] = vinos_df.region.str.split(',')
    vinos_df['color'] = vinos_df.color.str.split(',')
    vinos_df['sabor'] = vinos_df.sabor.str.split(',')
    return vinos_df
    
# Función para generar los descriptores de cada aspecto para cada vino
def generar_descriptor(vinos_df,aspecto):
    for index, row in vinos_df.iterrows():
        for aspect in row[aspecto]:
            vinos_df.at[index, aspect] = 1
    vinos_df= vinos_df.fillna(0)
    vinos_df = vinos_df.drop(['nombre', 'variedad','porcen_alch','pais','region','guarda','temp_consumo','color','aroma','maridaje','sabor','precio'], axis=1)
    descriptor = vinos_df.drop(['id'],axis=1)
    # Aplicar Softmax a cada fila del descriptor de vinos, para variedad y aromas
    arreglos_numpy = descriptor.to_numpy()
    for array in arreglos_numpy:
        temporal = []
        for item in array:
            if item == 1:
                temporal.append(item)     
        y = softmax(temporal)
        for i in range(len(array)):
            if array[i] == 1:
                array[i] = y[0]
    descriptor[:] = arreglos_numpy
    descriptor.to_csv(aspecto + ".csv")#Guardar el descriptor del aspecto en un archivo CSV

#Funcion ha ser llamada por la API para entregar las recomendaciones al cliente
def obtener_recomendaciones(id):
    #Leer los descriptores
    desc_variedad = pd.read_csv('variedad.csv', index_col=0).to_numpy()
    desc_region = pd.read_csv('region.csv',  index_col=0).to_numpy()
    desc_tipo = pd.read_csv('color.csv', index_col=0 ).to_numpy()
    desc_sabor = pd.read_csv('sabor.csv',  index_col=0).to_numpy()
    desc_aroma = pd.read_csv('aroma.csv',  index_col=0).to_numpy()
    desc_precio = pd.read_csv('precio.csv',index_col=0).to_numpy()
    desc_guarda = pd.read_csv('guarda.csv',index_col=0).to_numpy()

    dist_variedad = []
    dist_region = []
    dist_tipo = []
    dist_sabor = []
    dist_aroma = []
    dist_precio = []
    dist_guarda = []
    #Restar 1 al ID que se envía de la API, ya que el índice en los descriptores comienza en 0
    identificador  = int(id) - 1
    for i in range(len(desc_variedad)):
        # Distancia variedad
        dist_variedad.append(distancia_euclidiana(desc_variedad[identificador],desc_variedad[i])* w_variedad)
        # Distancia region
        dist_region.append(distancia_euclidiana(desc_region[identificador],desc_region[i]) *  w_region)
        # Distancia tipo
        dist_tipo.append(distancia_euclidiana(desc_tipo[identificador],desc_tipo[i]) * w_tipo )
        # Distancia sabor
        dist_sabor.append(distancia_euclidiana(desc_sabor[identificador],desc_sabor[i]) * w_sabor )
        # Distancia aroma
        dist_aroma.append(distancia_euclidiana(desc_aroma[identificador],desc_aroma[i]) * w_aroma)
        # Distancia precio
        dist_precio.append(distancia_euclidiana(desc_precio[identificador],desc_precio[i]) * w_precio )
        # Distancia entre la guarda de los vinos
        dist_guarda.append(distancia_euclidiana(desc_guarda[identificador],desc_guarda[i]) * w_guarda)
    
    # Sumatoria de los arreglos
    uno = np.asarray(dist_variedad, dtype=np.float32)
    tres = np.asarray(dist_region, dtype=np.float32)
    cuatro = np.asarray(dist_tipo, dtype=np.float32)
    cinco = np.asarray(dist_sabor, dtype=np.float32)
    seis = np.asarray(dist_aroma, dtype=np.float32)
    siete = np.asarray(dist_precio, dtype=np.float32)
    ocho = np.asarray(dist_guarda,dtype=np.float32)
    
    scores = uno  + tres + cuatro + cinco + seis + siete + ocho
    #Ordenar en un diccionario los puntajes obtenidos del calculo de similitud
    diccionario = { i : scores[i] for i in range(0, len(scores) ) }
    #Eliminar el elemento de igual identificador
    if(identificador < 32):#Para acercar los resultados a los del sommelier, no considerar los vinos de concha y toro cuando alguno de ellos se solicita la recomendación
        for i in range(0,33):
            del diccionario[i]
    else:
        del diccionario[identificador]
    diccionario_sort = sorted(diccionario.items(), key=operator.itemgetter(1), reverse=True)
    #Entregar al usuario los cinco vinos con mayor puntaje de similitud
    lista_similares = diccionario_sort[:5]
    lista_ids= []
    for i in range(len(lista_similares)):
        lista_ids.append(lista_similares[i][0] + 1)
    #Retornar los identificadores de los vinos a recomendar, mismos que servirán para recuperar información de la base de datos 
    return lista_ids

# Funcion principal usada cuando recién el motor de recomendaciones es creado
def funcion_main():
    # Abrir dataframe
    vinos_df = abrir_datafrane()    
    # Eliminar espacios en blanco para evitar la redundancia
    vinos_df = clean_data(vinos_df)
    # Convertir a lista los campos que son multivalor (aroma y variedad)
    vinos_df = estructurar_datos(vinos_df)
    # Aspectos a ser considerados al realizar la recomendación
    aspectos = ['variedad','region','color','sabor','aroma']
    # Generar los descriptores
    for aspecto in aspectos:
        df = vinos_df.copy()
        generar_descriptor(df ,aspecto)
    # Generar el archivo CSV con la información de los vinos
    vinos_df['precio'].to_csv("precio.csv",header ="precio")
    vinos_df['guarda'].to_csv("guarda.csv",header ="guarda")
    # Leer los descriptores
    desc_variedad = pd.read_csv('variedad.csv', index_col=0).to_numpy()
    desc_region = pd.read_csv('region.csv',  index_col=0).to_numpy()
    desc_tipo = pd.read_csv('color.csv', index_col=0 ).to_numpy()
    desc_sabor = pd.read_csv('sabor.csv',  index_col=0).to_numpy()
    desc_aroma = pd.read_csv('aroma.csv',  index_col=0).to_numpy()
    desc_precio = pd.read_csv('precio.csv',index_col=0).to_numpy()
    desc_guarda = pd.read_csv('guarda.csv',index_col=0).to_numpy()
    # Calculas la distancia euclidiana 
    #identificador = 16
    for identificador in range(0,33):
        dist_variedad = []
        dist_region = []
        dist_tipo = []
        dist_sabor = []
        dist_aroma = []
        dist_precio = []
        dist_guarda = []
        for i in range(len(desc_variedad)):
            if(i > 32):
                # Distancia variedad
                dist_variedad.append(distancia_euclidiana(desc_variedad[identificador],desc_variedad[i])* w_variedad)
                # Distancia region
                dist_region.append(distancia_euclidiana(desc_region[identificador],desc_region[i]) *  w_region)
                # Distancia tipo
                dist_tipo.append(distancia_euclidiana(desc_tipo[identificador],desc_tipo[i]) * w_tipo )
                # Distancia sabor
                dist_sabor.append(distancia_euclidiana(desc_sabor[identificador],desc_sabor[i]) * w_sabor )
                # Distancia aroma
                dist_aroma.append(distancia_euclidiana(desc_aroma[identificador],desc_aroma[i]) * w_aroma)
                # Distancia precio
                dist_precio.append(distancia_euclidiana(desc_precio[identificador],desc_precio[i]) * w_precio )
                # Distancia entre la guarda de los vinos
                dist_guarda.append(distancia_euclidiana(desc_guarda[identificador],desc_guarda[i]) * w_guarda)
            
    # Sumatoria de los arreglos
        uno = np.asarray(dist_variedad, dtype=np.float32)
        tres = np.asarray(dist_region, dtype=np.float32)
        cuatro = np.asarray(dist_tipo, dtype=np.float32)
        cinco = np.asarray(dist_sabor, dtype=np.float32)
        seis = np.asarray(dist_aroma, dtype=np.float32)
        siete = np.asarray(dist_precio, dtype=np.float32)
        ocho = np.asarray(dist_guarda,dtype=np.float32)
        
        scores = uno  + tres + cuatro + cinco + seis + siete + ocho
        # print(scores)
        
        diccionario2 = {}
        print("SCORES:", len(scores))
        j = 0 
        for i in range(33,101):
            diccionario2[i] = scores[j]
            j = j + 1 
        diccionario = { i : scores[i] for i in range(32, len(scores) ) }
        print("Los vinos similares a ", vinos_df.iloc[identificador]['nombre'], "son: \n\n")
        #del diccionario[identificador]
        diccionario_sort = sorted(diccionario2.items(), key=operator.itemgetter(1), reverse=True)
        i = 0
        #for vino in enumerate(diccionario_sort):
            #   i =  vino[1][0]
            #  print(vino[1][0], '--', diccionario2[vino[1][0]], "--", vinos_df.iloc[i]['nombre'])

        lista_similares = diccionario_sort[:5]
        lista_ids= []
        for i in range(len(lista_similares)):
            lista_ids.append(lista_similares[i][0] + 1)
        print(lista_ids)
