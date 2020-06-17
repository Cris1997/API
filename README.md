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

## 2.- Uso de las operaciones

Esta apartado describe como usar cada una de las operaciones 


## 3.- Descripción de los ficheros

Con el fin de orientar más al usuario, la siguiente tabla proporciona una descripción de los principales archivos que se encuentran en el repositorio.

| Nombre | Descripción |
| ------------- | ------------- |
| <a href ="https://github.com/Cris1997/API/blob/master/app.py">app.py</a>| Es el archivo principal, la API que levanta los servicios que se describieron en el primer punto.  |
| <a href ="https://github.com/Cris1997/API/blob/master/cartaocr.py">cartaocr.py</a>| Este programa contiene las funciones que modifican la fotografía y la ingresan al OCR para obtener el texto que se encuentra en ella.|
| <a href ="https://github.com/Cris1997/API/blob/master/similitud_texto.py">similitud_texto.py</a>| A partir del texto obtenido del OCR, la función de este programa lo analiza y trata de encontrar las coincidencias con los vinos que se encuentran en el sistema.
| <a href ="https://github.com/Cris1997/API/blob/master/sistema_recomendacion.py">sistema_recomendacion.py</a>|Ejecuta las operaciones necesarias con la información de los productos para generar recomendaciones.|

