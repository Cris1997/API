<p align="center">
  <img src="https://github.com/Cris1997/GUIS/blob/master/13.png">
</p>

<p align="center">
  En este repositorio se encuentra la implementación de la Rest-API que proporciona las principales funcionalidades a la aplicación pocket sommelier.
</p>

## 1.- Operaciones 

En la siguiente tabla se describen brevemente los cuatro servicios que brinda el API.

| Nombre | Descripción |
| ------------- | ------------- |
| Carta de vinos  | Es capaz de extraer el texto de la fotografía del menú, tratarlo y encontrar aquellos vinos que están en la base de datos.  |
| Etiqueta de vino  | Por medio de la solicitud realizada por la aplicación, este servicio recibe una fotografía que es evaluada por la red neural entrenada para asignarle una clase. Si el resultado es satisfactio otorga la información del vino identificado.  |
| Sistema de recomendacion  | Otorga como resultado una serie de recomendaciones solicitadas por la aplicación móvil (cliente), basándose en un vino que es proporcionado por esta última. |
| Todos los vinos  | Como su nombre lo indica, proporciona a la parte del cliente toda la información de los vinos que se encuentran en la base de datos.  |

## 2.- Especificación de las operaciones

  A lo largo de la implementación del API existen cuatro puntos donde se alojan los servicios que se describieron en el punto anterior. 
  
  ### 2.1 .- Carta de vinos
  
  |  | Tipo de operación |
  | ------------- | ------------- |
  
  
  
  ### 2.2 .- Etiqueta de vino
  ### 2.3 .- Carta de vinos
  ### 2.4 .- Carta de vinos


## 3.- Descripción de los ficheros

Con el fin de orientar más al usuario, la siguiente tabla proporciona una descripción de los principales archivos que se encuentran en el repositorio.

| Nombre | Descripción |
| ------------- | ------------- |
| <a href ="https://github.com/Cris1997/API/blob/master/app.py">app.py</a>| Es el archivo principal, la API que levanta los servicios que se describieron en el primer punto.  |
| <a href ="https://github.com/Cris1997/API/blob/master/cartaocr.py">cartaocr.py</a>| Este programa contiene las funciones que modifican la fotografía y la ingresan al OCR para obtener el texto que se encuentra en ella.|
| <a href ="https://github.com/Cris1997/API/blob/master/similitud_texto.py">similitud_texto.py</a>| A partir del texto obtenido del OCR, la función de este programa lo analiza y trata de encontrar las coincidencias con los vinos que se encuentran en el sistema.
| <a href ="https://github.com/Cris1997/API/blob/master/sistema_recomendacion.py">sistema_recomendacion.py</a>|Ejecuta las operaciones necesarias con la información de los productos para generar recomendaciones.|

## 4.- Uso de las operaciones

Para probar que el API funciona adecuadamente se utilizó el programa Postman, que gracias a su interfaz permite corrobar que cada uno de los endpoint desempeña su función. Las siguientes capturas muestran los resultados que deben mostrarse cuando se ejecutan las funciones con éxito.

### 4.1 .- Etiqueta de vino

<p align = "center"> 
  <img src="https://github.com/Cris1997/GUIS/blob/master/endpoint1.png" height="600" width = "800">
</p>

### 4.2 .- Lista de vinos
<p align = "center"> 
  <img src="https://github.com/Cris1997/GUIS/blob/master/endpoint2.png" height="600" width = "800">
</p>


### 4.3 .- Obtención de recomendaciones
<p align = "center"> 
  <img src="https://github.com/Cris1997/GUIS/blob/master/endpoint3.png" height="600" width = "800">
</p>

### 4.3 .- Información de vinos 
<p align = "center"> 
  <img src="https://github.com/Cris1997/GUIS/blob/master/endpoint4.png" height="600" width = "800">
</p>

## 5.- Ejemplos usados en el punto anterior

Las siguientes dos imágenes corresponden a los ejemplos que se usaron para verificar la funcionalidad de los endpoint de etiqueta de vino y lista de vinos. Si deseas descargar alguno de los ejemplares, favor de dar clic sobre la imagen.

<p align = "center">
  <a href = "https://github.com/Cris1997/GUIS/blob/master/vinoDiablo.jpg">
  <img src="https://github.com/Cris1997/GUIS/blob/master/vinoDiablo.jpg" height="600" width = "800">
  </a>
</p>

<p align = "center">
  <a href = "https://github.com/Cris1997/GUIS/blob/master/foto_menu.png">
  <img src="https://github.com/Cris1997/GUIS/blob/master/foto_menu.png" height="600" width = "800">
  </a>
</p>



