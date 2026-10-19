# Census Income KDD — Exploración, limpieza y transformación

Script de Spark (Scala) para el análisis y preprocesado del conjunto de datos
**Census Income KDD**, correspondiente al primer entregable de la asignatura.

Cubre las etapas de exploración, limpieza, selección de atributos y transformación
hasta obtener DataFrames con las columnas `features` y `label`. **No** incluye la
construcción de ningún clasificador.

## Requisitos

- Apache Spark (4.1.3) con `spark-shell` (Scala 2.13+)
- El fichero de datos `census-income.data`, que **no está incluido** en el repositorio

## Configuración: el PATH está fijado en el código

La ruta a los datos está **escrita directamente en el script** (no se lee de argumentos
ni de variables de entorno). Antes de ejecutarlo, edita estas líneas al principio de
`exploracion.scala`:

```scala
val PATH="./"
val DATA="census-income.data"
```

- `PATH` debe apuntar al directorio que contiene el fichero de datos. El valor por
  defecto, `"./"`, es el directorio desde el que se lanza `spark-shell`, **no** el
  directorio donde reside el script.
- La misma variable `PATH` se usa para **escribir** los ficheros Parquet resultantes
  (`census_train.parquet` y `census_test.parquet`), así que debe ser una ruta con
  permisos de escritura.

Ejemplo para una ruta absoluta:

```scala
val PATH="/home/usuario/uva_clasificadores/"
```

Ten en cuenta que la barra final es necesaria: la ruta se construye concatenando
`PATH + DATA`.

## Ejecución

Desde el directorio del proyecto:

```bash
spark-shell -i exploracion.scala
```

O bien, desde dentro de un `spark-shell` ya abierto:

```scala
:load exploracion.scala
```

Ambas formas funcionan: todas las expresiones encadenadas del script están envueltas
en paréntesis, de modo que el REPL no las interrumpe al procesar el fichero línea a
línea (que es lo que hace `:load`, a diferencia de `-i`).

## Salida

Por consola se muestran las tablas de cada etapa del análisis: estadísticas de los
atributos, valores ausentes, atípicos, matrices de correlación de Pearson y Spearman,
y el ranking de asociación con la clase mediante el coeficiente V de Cramér.

Además se generan dos directorios en `PATH`:

| Fichero | Contenido |
|---|---|
| `census_train.parquet` | 66 % de las instancias, con columnas `features` y `label` |
| `census_test.parquet` | 34 % restante, con la misma estructura |

La partición es estratificada por `label`, de forma que ambos conjuntos conservan la
proporción original de clases (aproximadamente un 6 % de instancias positivas). Si los
directorios ya existen, se sobrescriben.
