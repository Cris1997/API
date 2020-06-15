import pytesseract
import PIL
import numpy as np 
import cv2
from PIL import Image


#Funcion que aplica el OCR a la imagen en escala de grises
def ocr_function(imagen):
    text = pytesseract.image_to_string(imagen)
    return text 

#Convertir imagen RGB a escala de grises
def gray_scale_opencv(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    return gray

#Este método es llamado desde la API. Recibe como argumento la ruta donde se encuentra la imagen
def main_ocr(path):
   #Cargar la imagen
   imagen = cv2.imread(path)
   #Convertir la imagen a escala de grises
   image_gray = gray_scale_opencv(imagen)
   #Ingresar la fotografia al OCR y obtener su texto
   text = ocr_function(image_gray)
   #regresar el texto a la funcion donde fue llamada.
   return text