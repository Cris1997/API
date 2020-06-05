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

def main_ocr(path):
   #print("Pre-procesamiento de imágenes")
   imagen = cv2.imread(path)
   image_gray = gray_scale_opencv(imagen)
   text = ocr_function(image_gray)
   return text