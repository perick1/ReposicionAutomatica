![banner](bin/logo.png)

# Reposicion Automatica

Para una empresa de retail es muy importante tener presente las preferencias e intereses de los compradores para así poder ofrecer los productos correctos, en el momento correcto y en el lugar correcto. Dependiendo de los artículos que se tengan en los estantes de las tiendas, se tiene un mayor o menor ingreso, lo que hace relevante buscar estrategias para generar el mayor beneficio posible para la empresa, donde este beneficio está asociado al ingreso por venta de productos.

Se define entonces que el problema a tratar es la reposición óptima de productos en tiendas con el objetivo de generar un mayor beneficio económico para la empresa de retail. Con esto en mente, se puede escribir de forma cualitativa un problema de optimización.

MAX. SUMA (sobre i) de BENEFICIO (del producto i)
S.A. RESTRICCIONES DE STOCK

Definir la forma en que se modelan los beneficios no es una tarea trivial, más bien es un desafío importante dentro del problema. Usualmente, para definir los beneficios se toman en consideración factores como pricing, predicciones de ventas, caracterización de artículos similares, sumado a un modelamiento de las preferencias de los compradores. También, se consideran los costos asociados a cada producto y el costo por tener mucho tiempo un producto en bodega y no en tiendas.

La variable de optimización dentro del problema es la cantidad que se debe repartir de cada producto $i$ en la tienda $j$ en la semana $t$. Del problema de optimización se busca entonces obtener la cantidad exacta a repartir de cada producto, en cada tienda y en qué momento.

En este repositorio se tienen las herramientas para implementar algoritmos de optimización para resolver el problema descrito. Esto se hará de 2 maneras: PSO y metodos exactos para ILP como CPLEX + B&B.

## Implementación de solución exacta mediante PULP

se utiliza coin BB

## Implementación de PSO via pyswarm

bla
