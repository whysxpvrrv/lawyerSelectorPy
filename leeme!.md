

█░░ █▀▀█ █░░░█ █░░█ █▀▀ █▀▀█ 　 █▀▀ █▀▀ █░░ █▀▀ █▀▀ ▀▀█▀▀ █▀▀█ █▀▀█
█░░ █▄▄█ █▄█▄█ █▄▄█ █▀▀ █▄▄▀ 　 ▀▀█ █▀▀ █░░ █▀▀ █░░ ░░█░░ █░░█ █▄▄▀
▀▀▀ ▀░░▀ ░▀░▀░ ▄▄▄█ ▀▀▀ ▀░▀▀ 　 ▀▀▀ ▀▀▀ ▀▀▀ ▀▀▀ ▀▀▀ ░░▀░░ ▀▀▀▀ ▀░▀▀
                       by Fernando Parra        


*-------------------Lanzamiento de Server---------------------*

1. Entra en la carpeta ejemplo desde cmd
2. Ejecuta python manage.py runserver, (debes de tener instalado
python y nodejs)
3. Accede a http://127.0.0.1:8000/ desde tu navegador favorito
4. Si quieres chequear el status de la red neuronal a la hora de
la ejecucion, consulta cmd


*---------------Archivos y rutas Relevantes-----------------*

* Ejemplo > synapses.json
Pesos sinapticos de las palabras utilizadas para el entrenamiento
de la red neuronal

* Ejemplo > lawyerSelector > settings.py
Archivo python que contiene distintos parametros acerca de la
configuracion del proyecto

* Ejemplo > lawyerSelectorApp > views.py
Archivo python en el que se encuentran la redireccion a las templates,
y el algoritmo a la red neuronal

* Ejemplo > lawyerSelectorApp > templates
Ruta en la que se encuentran las distintas templates

* Ejemplo > lawyerSelectorApp > static
Ruta en la que se encuentran todos los archivos estaticos usados en
las templates

* Ejemplo > lawyerSelectorApp > static > css > style.css
Archivo css con los estilos generales


* Ejemplo > lawyerSelectorApp > static > css > lawyers.css
Archivo css con los estilos de las templates de los abogados


* Ejemplo > lawyerSelectorApp > static > js > main.js
Archivo js con scripts para uso generico de la app

