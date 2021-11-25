v01: 	- nuevo modelo con gurobi

v02: 	- se agregan costos por almacenamiento en tiendas y en centro de distribucion
	- se agregan restricción por limite de transporte
	- se elimina restriccion de minimo de exhcibición
	- se cambian parametros como stock maximo conjunto en tienda, SCD y demanda

v03:	- añadir ventanas de tiempo con total de 15 semanas
	- buscar parámetros con sentido para mostrar
	- crear curvas de precio 15 semanas
	- agregar minimo de exhibicion (tener ojo con máximo en tiendas)(quizas no considerar la exhibicion en el maximo en tiendas)

v04:	- Se generalizan las curvas, más faciles de setear.
	- Se ordena todo en funciones

v05:	- Se mejoran los algoritmos
	- Se crean graficos de barra de la reposición semanal

v06:	- Se cambia parametro en función demanda
	- Se busca obtener mismos resultados que en PuLP
	- Retorna valor del óptimo

v07:	- Se añaden quiebres de stock modificando parámetros.
	- Se agraga grafico que muestre quiebres
	- Se cambia restricción defectuosa de minimo de exhibición




