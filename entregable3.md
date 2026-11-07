# Entregable 3 — Creación, selección y evaluación de modelos

**Contexto:** Census Income KDD (clasificación **binaria**, ~6 % de positivos).
**Entorno:** Spark 4.1.3, scripts Scala sin parámetros, biblioteca `ml` (`mllib` solo para métricas no disponibles en `ml`).

---

## 1. Preparación de datos (reutilizar entregables 1 y 2)

- [ ] Reutilizar **las mismas particiones** `census_train.parquet` / `census_test.parquet` (66/34, estratificada, misma semilla).
- [ ] Verificar que el test **no se toca** en ninguna fase de creación/selección.
- [ ] Revisar la **limpieza** del entregable 1: corregir errores o mejoras sustantivas (ausentes, atípicos, duplicados, capado).
- [ ] Revisar la **transformación**: solo recodificaciones (OneHot vs. índice, escalado, binning), **sin** añadir/eliminar atributos.
- [ ] Decidir transformaciones **por modelo** (árboles → índices; modelos lineales → OneHot + escalado).
- [ ] Extraer limpieza/transformación común a un script reutilizable (p. ej. `comun.scala`) cargado por los demás.
- [ ] Documentar y **resaltar en la memoria** cada modificación respecto a entregables previos.

## 2. Modelo A — clasificador básico

- [ ] Elegir algoritmo (`DecisionTreeClassifier` o `LogisticRegression`).
- [ ] Definir `Pipeline` (transformaciones + clasificador).
- [ ] Exploración previa de "grano grueso" para acotar rangos razonables de parámetros.
- [ ] Construir `ParamGridBuilder` con pocos parámetros y rangos justificados.
  - DT: `maxDepth`, `maxBins`, `impurity`, `minInstancesPerNode`.
  - LR: `regParam`, `elasticNetParam`, `maxIter`.
- [ ] Configurar `CrossValidator` (validación cruzada **interna** sobre train, `numFolds` 3–5, semilla fija).
- [ ] Elegir métrica de selección adecuada al desbalanceo (`BinaryClassificationEvaluator` → **areaUnderPR** o AUC-ROC).
- [ ] Refinar la rejilla ("grano fino") alrededor del óptimo de la primera búsqueda.
- [ ] Registrar: proceso seguido, parámetros finales, valor de la métrica en CV.
- [ ] Guardar el modelo en `./modeloDT` (o `./modeloLR`).

## 3. Modelo B — clasificador avanzado (obligatorio)

- [ ] Elegir algoritmo (`RandomForestClassifier`, `GBTClassifier`, `MultilayerPerceptron`, `LinearSVC`).
- [ ] Repetir todos los pasos de la sección 2 con su propia rejilla.
  - RF: `numTrees`, `maxDepth`, `featureSubsetStrategy`, `subsamplingRate`.
  - GBT: `maxIter`, `maxDepth`, `stepSize`, `subsamplingRate`.
- [ ] Vigilar el **coste**: la rejilla crece exponencialmente; búsqueda en dos fases.
- [ ] Registrar: proceso, parámetros finales, métrica de CV.
- [ ] Guardar el modelo en `./modeloRF` (o `./modeloGBT`).

> Métricas adicionales de aceptación/rechazo solo si se obtienen del **train** por método honesto (sin solape train/validación).

## 4. Evaluación sobre el conjunto de prueba (binario)

Para **cada** modelo, calcular sobre el test:

- [ ] Tasa de error.
- [ ] Desviación estándar del error: `sqrt(err*(1-err)/n)`.
- [ ] Intervalo de confianza al 95 %: `err ± 1.96 * sd`.
- [ ] Matriz de confusión.
- [ ] Tasa de ciertos positivos (TPR).
- [ ] Tasa de falsos positivos (FPR).
- [ ] Área bajo la curva ROC.
- [ ] Curva ROC (puntos de `BinaryClassificationMetrics.roc`, exportados/graficados).
- [ ] Área bajo la curva PR.

Notas de implementación:
- [ ] Usar `ml` (`BinaryClassificationEvaluator`, `MulticlassMetrics` vía `mllib`) y `mllib.evaluation.BinaryClassificationMetrics` para ROC/PR.
- [ ] Construir el RDD `(score, label)` a partir de la columna `probability`.

## 5. Entregables software (mínimo 4 scripts)

- [ ] `crea_modeloXX_A.scala` — crea y selecciona el modelo básico; guarda en `./modeloXX`.
- [ ] `crea_modeloYY_B.scala` — crea y selecciona el modelo avanzado; guarda en `./modeloYY`.
- [ ] `eval_modeloXX.scala` — **lee** `./modeloXX` (no lo entrena), transforma el test y muestra todas las métricas.
- [ ] `eval_modeloYY.scala` — ídem para `./modeloYY`.
- [ ] Todos los scripts **sin parámetros** (rutas fijadas en el código) y ejecutables con `spark-shell -i script.scala`.
- [ ] Los scripts de evaluación deben funcionar sobre **cualquier subconjunto** de datos con el mismo formato.
- [ ] Entregar también las carpetas de los **dos modelos** generados.

## 6. Memoria

- [ ] Tabla de contenidos.
- [ ] Breve introducción.
- [ ] Resumen ejecutivo de los datos.
- [ ] Descripción del trabajo, un apartado por fase:
  - [ ] Conjuntos de entrenamiento y prueba (cambios resaltados).
  - [ ] Limpieza de datos (cambios resaltados).
  - [ ] Transformación de datos (cambios resaltados).
  - [ ] Creación y selección de modelos: proceso de búsqueda, parámetros finales y métrica de CV por modelo.
  - [ ] Evaluación de modelos finales: tablas de métricas y curva ROC.
- [ ] Conclusiones y comparación entre los dos modelos.
- [ ] Porcentaje de trabajo y horas por estudiante.
- [ ] Referencias bibliográficas.
- [ ] Apéndices (rejillas completas, salidas de consola).

## 7. Comprobación final

- [ ] El test no se ha usado en selección de parámetros.
- [ ] Los scripts de evaluación cargan el modelo desde disco y no lo reentrenan.
- [ ] Todas las métricas exigidas aparecen en la salida de los scripts y en la memoria.
- [ ] Ejecución limpia de los 4 scripts desde cero en Spark 4.1.3.
