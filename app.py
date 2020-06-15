#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cristianrosales
"""

#Librerías necesarias
import os
import tensorflow as tf
import PIL
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import operator
import math
import zipfile
import pytesseract
import json 
from PIL import Image
from flask import Flask, jsonify, request
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


#Librerias para la base de datos
import pandas as pd
import pprint
import bson

from bson.json_util import dumps
from pymongo import MongoClient
#Fin de las librerias para la base de datos
# -*- coding: utf-8 -*-

#Archivos externos de python donde se encuentran las funciones de los algoritmos de busqueda, ocr y sistema de recomendacion
import similitud_texto
import cartaocr
import sistema_recomendacion

IMAGE_SIZE = (12, 8) #Tamano de la imagen
NUM_CLASSES = 33 #Clases de los vinos que pueden ser detectados con el modelo 
PATH_TO_CKPT = 'frozen.pb' #Ruta del modelo congelado

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map.pbtxt' #Archivo que contiene las categorias de las 33 clases de vino
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#Directorio donde se almacenan las imágenes que llegan para ser clasificadas (foto de botellas de vino)
UPLOAD_FOLDER = '/imagenes/' 
#Directorio donde se almacenan las imágenes que llegan al servidor para encontrar vinos en la carta
UPLOAD_FOLDER_OCR = '/img_ocr/' 
#Extensiones de archivos  que pueden ser aceptadas
ALLOWED_EXTENSIONS = {'png','jpeg','jpg'}


app = Flask(__name__)#Iniciar flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#End point de inicio (solamente de prueba)
@app.route("/")
def hello():
    return jsonify(ok=True, description="Inicio del servicio Pocket Sommelier", status_code=200)

#Retorna un vino de la base de datos
@app.route('/foundone/<string:id>',methods=['GET'] )
def enviar_uno(id):
    #Busca un vino en la base de datos
    data  = get_info(int(id))
    #El valor almacenado en data es decodificado en formato JSON valido para otorgar respuesta al cliente
    return dumps(data,ensure_ascii=False).encode('utf8')

#Retorna todos los vinos de la  base de datos
@app.route('/allwines/', methods=['GET'] )
def enviar_todos():
    data = get_all() #Consulta a la base de datos Mongo
    return dumps(data,ensure_ascii=False).encode('utf8') #Se codifican los datos en JSON y se envia la respuesta al cliente

#===========================
#API Recomendador 
#===========================
#End point para obtener recomendaciones de productos similares
@app.route('/recomendador', methods = ['GET'] )
def genera_recomendacion():
    #Obtenemos el parametro ID y ejecutamos la funcion del sistema de recomendacion para obtener recomendaciones
    list_similares_id  = sistema_recomendacion.obtener_recomendaciones(request.args['id'])
    list_similares_data = []
    #Recuperar los datos de cada uno de los vinos que se encuentran en las recomendaciones
    for id_vino in list_similares_id:
        #Consulta a la base de datos para obtener la informacion de un vino
        data =vinos_collection.find_one({"_id": id_vino})
        list_similares_data.append(data)
    #Codificamos los datos en JSON y enviamos la respuesta al cliente
    return dumps(list_similares_data,ensure_ascii=False).encode('utf8')

#===========================
#API OCR
#===========================
#Endpoint para obtener la informacion de los vinos que se encuentran en una lista o menu
@app.route('/ocr',methods=['POST'])
def algoritmo_ocr():
    #Verificamos que la peticion sea de tipo Post y que este incluido el archivo
    if request.method == 'POST':
        if 'file' not in request.files:
            return "error" 
        user_file = request.files['file']
        if user_file.filename == '':
            return "error"
        else:
            path = os.path.join(os.getcwd() + UPLOAD_FOLDER_OCR + user_file.filename)
            #El archivo es guardado en el directorio para imagenes a ser procesadas por el OCR
            user_file.save(path)
            #Ingresamos la fotografia al OCR y guardamos el texto en la variable text
            text = cartaocr.main_ocr(path)
            #Con el texto rescatado del OCR se ejecutara la funcion encargada de encontrar los vinos
            lista_vinos = similitud_texto.encontrarVinos(text)
        #Si las funcionnes anteriores no encontraron algun vino entonces retornamos un error indicando que no hay resultados
        if len(lista_vinos) == 0:
            return "error"
        lista_final = []
        #Eliminamos elementos repetidos
        for id in lista_vinos:
            if id not in lista_final:
                lista_final.append(id)
        list_vinos_data = []
        #Si existen valores en la lista que regresa la funcion de busqueda es necesario extraer toda la informacion de cada vino, cuyo identificador se encuentre en la lista
        for id_vino in lista_final:
            data =vinos_collection.find_one({"_id": id_vino})
            list_vinos_data.append(data)
        return dumps(list_vinos_data,ensure_ascii=False).encode('utf8')
    
#===========================
#API Deep Learning
#===========================   
@app.route('/clasificar',methods=['POST'])
def predict():
    #Verificamos que la peticion sea de tipo Post y que este incluida la fotografia de la botella del vino
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Algo salio mal"
        user_file = request.files['file']
        if user_file.filename == '':
            return "No se encontro el nombre del archivo"
        else:
            path = os.path.join(os.getcwd() + app.config['UPLOAD_FOLDER'] + user_file.filename)
            #Guardamos la fotografia en el directorio para poder cargarla al modelo
            user_file.save(path)
            #Por medio de la siguiente funcion reducimos el tamaño de la imagen
            reducirImagen(path,user_file.filename)
            #Realizamos la prediccion de la clasificacion de la fotografia
            porcentaje, clase = prediccionVino(path)
            #Evaluamos el porcentaje con el que se clasifico la fotografia 
            if int(porcentaje) > 70:
                #Si el porcentaje es mayor a 70% enntonces la tomamos como valida y extraemos la informacion de la base de datos
                data  = get_info(int(clase))
                #Decodificamos a formato JSON y enviamos la respuesta al cliente 
                return dumps(data,ensure_ascii=False).encode('utf8')
            else:
                return "error"


#Funcion para clasificar la fotografia que mando el usuario desde la aplicacion y fue guardada en un directorio
def prediccionVino(path):
    image = Image.open(path) # Cargar la fotografia
    image_np = load_image_into_numpy_array(image) #convertir la imagen en un arreglo de valores numericos
    image_np_expanded = np.expand_dims(image_np, axis=0) #incrementa la dimension del arreglo obtenido
    output_dict = run_inference_for_single_image(image_np, detection_graph) # se ingresa la imagen al modelo entrenado
    porcentaje = output_dict['detection_scores'][0]*100 #del resultado obtenido 
    clase  =  output_dict['detection_classes'][0]    
    return porcentaje, clase


def reducirImagen(path,name_image):
    img = Image.open(path)#Abrir la imagen 
    #Obtener el ancho y el alto de la imagen
    width = img.size[0] 
    heigh = img.size[1]
    if width > heigh:
        basewidth = 400 
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img = img.rotate(270,0,1)
        img.save("imagenes/" + name_image.lower()) 
    else: #Cuando la foto es vertical la imagen no es rotada
        baseheight = 400
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), PIL.Image.ANTIALIAS)
        img.save("imagenes/" + name_image.lower())

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.compat.v1.Session() as sess:
    # with tf.Session() as sess:
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:

        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
 
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
      output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

#===========================
#Fucnion para la manipulacion de base de datos 
#===========================
def insert_all(dataframe_vinos):
    #Debido a que el campo variedad y aroma es multivalor deben ser almacenados en la base de datos como una lista
    dataframe_vinos['aroma'] = dataframe_vinos.aroma.str.split(',')
    dataframe_vinos['variedad'] = dataframe_vinos.variedad.str.split(',')
#    print(type(dataframe_vinos['variedad'][100]))
    i = 0 
    for i in range(i,len(dataframe_vinos)):
        vino = {"_id": i + 1,
              "nombre": dataframe_vinos['nombre'][i],
              "variedad": dataframe_vinos['variedad'][i],
              "porcen_alch":dataframe_vinos['porcen_alch'][i] ,
              "pais":dataframe_vinos['pais'][i],
              "region":dataframe_vinos['region'][i],
              "guarda":dataframe_vinos['guarda'][i],
              "temp_consumo":dataframe_vinos['temp_consumo'][i],
              "color":dataframe_vinos['color'][i],
              "aroma":dataframe_vinos['aroma'][i],
              "sabor":dataframe_vinos['sabor'][i],
              "maridaje":dataframe_vinos['maridaje'][i],
              "name_image":dataframe_vinos['name_image'][i]} 
        #vinos_collection.insert_one(vino).inserted_id
        
def get_info(id_vino):
    #Verificamos que el argumento sea de tipo entero
    if type(id_vino) is str:
        int(id)# Convertir el valor String a Integer
    #Consulta a la base de datos para obtener la informacion de un vino
    data =vinos_collection.find_one({"_id": id_vino})
    return data
    
def get_all():
    #Hacemos una lectura en la base de datos para obtener toda la informacion de los vinos
    data = vinos_collection.find()
    return data

def distancia_euclidiana(x,y):
    return 1/(1+np.sqrt(np.sum((x - y) ** 2)))


if __name__ == '__main__':
    #Iniciar base de datos
    #Conexion con la base de datos
    cliente = MongoClient("mongodb://localhost:27017/")
    database = cliente["vinos"] #Nombre de la base de datos
    vinos_collection = database["data_vinos"] #Nombre de la coleccion de mongodb donde esta la informacion de los vinos
    #Cargar freeze model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef() 
        # od_graph_def = tf.GraphDef()
        with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        # with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')

    app.run(debug = True, host = '192.168.1.70', port = 4000)
