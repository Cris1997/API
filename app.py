#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cristianrosales
"""

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

#Mis librerias
#import cartaocr 
import similitud_texto
import cartaocr
import sistema_recomendacion

IMAGE_SIZE = (12, 8) #Tamano de la imagen
NUM_CLASSES = 33 #Clases de los vinos que pueden ser detectados con el modelo 
PATH_TO_CKPT = 'frozen.pb' #Ruta del modelo congelado

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'label_map.pbtxt' #Archivo que contiene las categorias de las 33 clases de vino
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

UPLOAD_FOLDER = '/imagenes/'
UPLOAD_FOLDER_OCR = '/img_ocr/'
ALLOWED_EXTENSIONS = {'png','jpeg','jpg'}


app = Flask(__name__)#Iniciar flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
    return jsonify(ok=True, description="Probando el CI de Github y Azure :D", status_code=200)
#Retorna un vino de la base de datos
@app.route('/foundone/<string:id>',methods=['GET'] )
def enviar_uno(id):
    data  = buscar_vino(int(id))
#    print(data)
    return dumps(data,ensure_ascii=False).encode('utf8')

#Retorna todos los vinos de la  base de datos
@app.route('/allwines/', methods=['GET'] )
def enviar_todos():
    #data = get_all()
    data =BASEDATOS
    return dumps(data,ensure_ascii=False).encode('utf8')

#===========================
#API Recomendador 
#===========================
@app.route('/recomendador/<string:id>', methods = ['GET'] )
def genera_recomendacion(id):
    list_similares_id  = sistema_recomendacion.obtener_recomendaciones(id)
    #print(list_similares_id)
    list_similares_data = []
    for id_vino in list_similares_id:
        data = buscar_vino(id_vino)
        #data =vinos_collection.find_one({"_id": id_vino}) MONGODBB 
        list_similares_data.append(data)
    return dumps(list_similares_data,ensure_ascii=False).encode('utf8')

#===========================
#API OCR
#===========================
@app.route('/ocr',methods=['POST'])
def algoritmo_ocr():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "error" 
        user_file = request.files['file']
        if user_file.filename == '':
            return "error"
        else:
            path = os.path.join(os.getcwd() + UPLOAD_FOLDER_OCR + user_file.filename)
            user_file.save(path)
            text = cartaocr.main_ocr(path)
            lista_vinos = similitud_texto.encontrarVinos(text)
        
        if len(lista_vinos) == 0:
            return "error"
        lista_final = []
        for id in lista_vinos:
            if id not in lista_final:
                lista_final.append(id)
        list_vinos_data = []
        for id_vino in lista_final:
            data  = buscar_vino(id_vino)
            #data =vinos_collection.find_one({"_id": id_vino}) BD MONGODB
            list_vinos_data.append(data)
        return dumps(list_vinos_data,ensure_ascii=False).encode('utf8')
    
#===========================
#API Deep Learning
#===========================   
@app.route('/clasificar',methods=['POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "Algo salio mal"
        
        user_file = request.files['file']
        if user_file.filename == '':
            return "No se encontro el nombre del archivo"
        else:
            path = os.path.join(os.getcwd() + app.config['UPLOAD_FOLDER'] + user_file.filename)
#            print(path)
            user_file.save(path)
            reducirImagen(path,user_file.filename)
            porcentaje, clase = prediccionVino(path)
#            print(int(porcentaje))
#            print(clase)
            if int(porcentaje) > 70:
                data = buscar_vino(int(clase))
                #data  = get_info(int(clase)) BD MongoDB
#               print(data)
                return dumps(data,ensure_ascii=False).encode('utf8')
                # return str(clase)
                #return "error"
            else:
                return "error"
                # return str(clase)
                #info_vino = get_info(int(clase))
                #return dumps(info_vino,ensure_ascii=False).encode('utf8')
#            print(info_vino)


def prediccionVino(path):
    image = Image.open(path)

    image_np = load_image_into_numpy_array(image)

    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    porcentaje = output_dict['detection_scores'][0]*100
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

#Funcion para transforar 
def class_int_to_text(row_label):
  if row_label ==1:
    return 'reservado cabernet sauvignon'
  elif row_label == 2:
    return 'reservado merlot'
  elif row_label == 3:
    return 'reservado carmenere'
  elif row_label == 4:
    return 'reservado malbec'
  elif row_label == 5:
    return 'resrvado shiraz'
  elif row_label == 6:
    return 'reservado rose'
  elif row_label == 7:
    return 'reservado white zinfandel'
  elif row_label == 8:
    return 'reservado sauvignon blanc'
  elif row_label == 9:
    return 'frontera merlot'
  elif row_label == 10:
    return 'frontera carmenere'
  elif row_label == 11:
    return 'frontera cabernet sauvignon'
  elif row_label == 12:
    return 'frontera chardonnay'
  elif row_label == 13:
    return 'casillero del diablo red blend'
  elif row_label == 14:
    return 'casillero del diablo cabernet sauvignon'
  elif row_label == 15:
    return 'casillero del diablo merlot'
  elif row_label == 16:
    return 'casillero del diablo devils collection rojo'
  elif row_label == 17:
    return 'casillero del diablo chardonnay'
  elif row_label == 18:
    return 'casillero del diablo pinot noir'
  elif row_label == 19:
    return 'casillero del diablo carmenere'
  elif row_label == 20:
    return 'casillero del diablo devils collecition verde'
  elif row_label == 21:
    return 'casillero del diablo malbec'
  elif row_label == 22:
    return 'marques de casa concha merlot'
  elif row_label == 23:
    return 'marques de casa concha carmenere'
  elif row_label == 24:
    return 'marques de casa concha chardonnay'
  elif row_label == 25:
    return 'trio merlot'
  elif row_label == 26:
    return 'trio chardonnay'
  elif row_label == 27:
    return 'trio cabernet sauvignonn'
  elif row_label == 28:
    return 'brut'
  elif row_label == 29:
    return 'diablo dark red'
  elif row_label == 30:
    return 'casillero del diablo rose'
  elif row_label == 31:
    return 'casillero del diablo sauvignon blanc'
  elif row_label == 32:
    return 'casillero del diablo reserva privada cabernet sauvignon'
  elif row_label == 33:
    return 'reservado sweet red'
  else:
    print("no se encontro")


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


      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

#===========================
#Aplicacion de base de datos 
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
    if type(id_vino) is str:
        int(id)
#    print(vinos_collection.find_one({"_id": id_vino}))
   # data =vinos_collection.find_one({"_id": id_vino})
   #return data
    
def get_all():
  print("HOLA")
    #data = vinos_collection.find()
#    for x in vinos_collection.find():
#        print(x) 
    #return data

def distancia_euclidiana(x,y):
    return 1/(1+np.sqrt(np.sum((x - y) ** 2)))

def buscar_vino(id):
    info = BASEDATOS[id-1]
    return info 
#Arrancar el servidor 

if __name__ == '__main__':
    BASEDATOS = [] 
    with open('file.json') as f:
      for line in f:
          BASEDATOS.append(json.loads(line))

    #Conexion con la base de datos
    #cliente = MongoClient("mongodb://localhost:27017/")
    #database = cliente["vinos"] #Nombre de la base de datos
    #vinos_collection = database["data_vinos"] #Nombre de la coleccion de mongodb donde esta la informacion de los vinos
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

    #Cargar las etiquetas de las categorias
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    app.run()
