![banner](bin/logo.png)

# Trabajo de título

MEMORIA PARA OPTAR AL TÍTULO DE INGENIERO CIVIL ELÉCTRICO

TÍTULO                  : REPOSICIÓN AUTOMÁTICA EN EMPRESA DE RETAIL MEDIANTE ALGORITMOS DE OPTIMIZACIÓN

ESTUDIANTE              : ERICK FELIPE SALOMÓN PÉREZ FLORES

PROFESOR GUÍA           : DAVID VALENZUELA U.

PROFESOR CO-GUÍA        : FRANCISCO RIVERA S.

MIEMBRO DE LA COMISIÓN  : DANIEL POLA C.


ENERO 2022

# Resumen

El presente Trabajo de Título corresponde a la memoria realizada en una compañía de retail, realizando modelamiento matemático y optimización. Para las compañías de retail es clave reponer productos en el momento y lugar preciso para tener un inventario equilibrado según los estándares de la companía y, particularmente, no tener problemas de stock. Como problemas de stock se destacan los quiebre de stock (falta de productos en tiendas) y sobre-stock (exceso de productos en tiendas), este último puede generar que productos entren al proceso de liquidación, reduciendo el margen de utilidad. Generar un calendario de reposición eficiente es una tarea compleja debido a la gran cantidad de SKUs (Stock Keeping Units) y tiendas que manejan las grandes compañías, por lo que la planificación se genera mediante softwares especializados o incluso planificaciones manuales rudimentarias. Muchos de estos softwares no son capaces de satisfacer todas las necesidades particulares de cada compañía, por lo que una buena dirección de desarrollo de los retailers es internalizar estos procesos y generar herramientas computacionales propias.

Por lo anterior, el trabajo se enfoca en desarrollar un modelo simplificado para obtener un calendario de reposición, y herramientas computacionales que permitan poner a prueba este modelo. El modelo se desarrolla como un problema  de optimización lineal de enteros, donde la función objetivo contempla un término de contribución de utilidades, tratando de maximizarlas. Adicionalmente, se agregan 3 términos a la función objetivo con el fin de moldear la solución encontrada de reposición, considerando reposiciones una vez por semana. Se simula el comportamiento del modelo mediante scripts programados en lenguaje Python, que permiten realizar la optimización con los solvers lineales existentes PuLP CBC y Gurobi. Se prueba con parámetros simulados, donde el forecast de la demanda es el input más destacable.

Como resultados más importantes se tiene que la implementación de varios términos en la función objetivo, junto con las restricciones utilizadas, influencian la optimización moldeando la solución encontrada. Lo anterior se ve como una buena herramienta para personalizar la  planificación de la reposición, dirigiendo el output del modelo a distribuciones de stock acordes a la realidad de la compañía. La implementación por ventanas permite anticiparse a peaks de demanda, reduciendo el número de quiebres de stock. Se simula y evalúa el error que puede presentar el forecast de demanda, obteniendo perdidas de utilidades del orden del 1.8% para un MAPE del 16% y la disminución de utilidades es de un 6.1% para un MAPE del 40%.

Se obtiene un modelo robusto que entrega reposiciones con características deseables para la compañía de retail. Las herramientas computacionales utilizadas permiten realizar optimizaciones, y se presentan de buena manera los resultados computacionales. El trabajo realizado permite avanzar en la dirección correcta para desarrollar herramientas propias que permitan obtener un calendario de reposición utilizando solvers lineales.
