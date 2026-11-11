/* =============================================================================
   CREACIÓN Y SELECCIÓN DEL MODELO GBT (Gradient-Boosted Trees) — Census Income KDD

   Uso:  :load gbt.scala    (sin parámetros; solo hay que ajustar PATH y FILE)

   1. Lee el fichero EN BRUTO (PATH + FILE) y hace la MISMA partición estratificada
      66/34 que preparation.scala (percent_rank por clase con la semilla SEED).
   2. Limpieza, selección y codificación aprendidas SOLO con entrenamiento (mismas
      reglas que preparation.scala: moda/mediana, capado q3 + 3*IQR, |r| >= 0.1 y
      Cramér's V >= 0.1 con redundancia). El conjunto de prueba no se usa para nada.
   3. Selección de parámetros con CrossValidator (validación cruzada interna de
      K_FOLDS particiones estratificadas) y métrica área bajo la curva PR, adecuada con
      un 6% de positivos (la tasa de error premia predecir siempre "<50k").
        Fase 1 (grano grueso): maxDepth x maxIter.
        Fase 2 (grano fino):   maxDepth vecinos x stepSize, con el mejor maxIter.
   4. Umbral de decisión: con las predicciones FUERA DE MUESTRA de la validación
      cruzada (cada instancia la predice el submodelo que no la vio) se elige el
      umbral sobre P(>50k) que maximiza el F1 de la clase positiva. Se guarda dentro
      del modelo con setThresholds, así que el script de evaluación lo aplica solo.
   5. Guarda un único PipelineModel autocontenido en ./modeloGBT (datos en bruto ->
      limpieza SQL -> StringIndexer -> VectorAssembler -> GBT) y la partición de
      prueba en bruto en PATH + "census-test.data" para evaluarModeloGBT.scala.
   ============================================================================= */

import org.apache.spark.sql.{Column, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, StringIndexerModel, SQLTransformer}
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.classification.{GBTClassifier, GBTClassificationModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.functions.vector_to_array


/* =============================================================================
   FASE 0 — Configuración y carga
   ============================================================================= */

val PATH = "./"
val FILE = "census-income.data"
val MODELO = "./modeloGBT"

val SEED = 0L
val TRAIN_FRAC = 0.66            // misma partición que en los entregables anteriores
val UMBRAL_AUSENTES = 0.40
val UMBRAL_RELEVANCIA = 0.10
val UMBRAL_REDUNDANCIA = 0.80
val K_IQR = 3.0
val K_FOLDS = 3                  // validación cruzada interna
val PARALELISMO = 2              // modelos que CrossValidator entrena a la vez

// Rejilla de grano grueso (fase 1). La fase 2 se construye alrededor del mejor valor.
val DEPTHS_GRUESO = Array(3, 5, 7)
val ITERS_GRUESO = Array(100, 300)
val STEP_GRUESO = 0.1
val STEPS_FINO = Array(0.05, 0.1, 0.2)
val SUBSAMPLING = 0.8            // Stochastic Gradient Boosting (diapositiva 40)

// GBT es iterativo: sin checkpoint, el linaje de cada iteración crece y con muchas
// iteraciones puede provocar StackOverflowError. Se guarda en el temporal del sistema.
spark.sparkContext.setCheckpointDir(System.getProperty("java.io.tmpdir") + "/checkpoint_gbt")

val censusSchema = StructType(Seq(
  "age", "class_of_worker", "detailed_industry_recode", "detailed_occupation_recode",
  "education", "wage_per_hour", "enroll_in_edu_inst_last_wk", "marital_stat",
  "major_industry_code", "major_occupation_code", "race", "hispanic_origin", "sex",
  "member_of_a_labor_union", "reason_for_unemployment", "full_or_part_time_employment_stat",
  "capital_gains", "capital_losses", "dividends_from_stocks", "tax_filer_stat",
  "region_of_previous_residence", "state_of_previous_residence",
  "detailed_household_and_family_stat", "detailed_household_summary_in_household",
  "instance_weight", "migration_code_change_in_msa", "migration_code_change_in_reg",
  "migration_code_move_within_reg", "live_in_this_house_1_year_ago",
  "migration_prev_res_in_sunbelt", "num_persons_worked_for_employer",
  "family_members_under_18", "country_of_birth_father", "country_of_birth_mother",
  "country_of_birth_self", "citizenship", "own_business_or_self_employed",
  "fill_inc_questionnaire_for_veterans_admin", "veterans_benefits", "weeks_worked_in_year",
  "year", "income" // clase: '- 50000.' o '50000+.'
).map(StructField(_, StringType, true)))

val rawCensusDF = spark.read.format("csv").schema(censusSchema).load(PATH + FILE)

val classCol = "income"
val contCols = Seq("age", "wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks", "instance_weight")
val discCols = Seq("num_persons_worked_for_employer", "weeks_worked_in_year")
val numCols = contCols ++ discCols
val nomCols = rawCensusDF.columns.filterNot(c => numCols.contains(c) || c == classCol).toSeq
val ceroInfladas = Seq("wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks")

def esPositivo(c: Column): Column = trim(c) === "50000+."


/* =============================================================================
   FASE 1 — Partición estratificada 66/34 (idéntica a preparation.scala), ANTES de
   aprender nada de los datos
   ============================================================================= */

val conAzar = (rawCensusDF
  .withColumn("label_strat", when(esPositivo(col(classCol)), 1).otherwise(0))
  .withColumn("azar", rand(SEED))
  .withColumn("rango", percent_rank().over(Window.partitionBy("label_strat").orderBy("azar")))
  .cache())

val trainRaw = conAzar.filter(col("rango") < TRAIN_FRAC).drop("label_strat", "azar", "rango").cache()
val testRaw = conAzar.filter(col("rango") >= TRAIN_FRAC).drop("label_strat", "azar", "rango")

val nTrain = trainRaw.count()
println(s"\nInstancias de entrenamiento: $nTrain (el conjunto de prueba no se usa en este script)")


/* =============================================================================
   FASE 2 — Limpieza aprendida con entrenamiento (mismas reglas que preparation.scala)
   ============================================================================= */

val colsNoAtributo = Seq("instance_weight", "year")

val filaAus = trainRaw.select(nomCols.map(c =>
  avg(when(col(c).isNull || trim(col(c)) === "?", 1.0).otherwise(0.0)).as(c)): _*).first()
val pctAusentes: Map[String, Double] = nomCols.map(c => c -> filaAus.getAs[Double](c)).toMap

val dropCols = colsNoAtributo ++ nomCols.filter(c => pctAusentes(c) > UMBRAL_AUSENTES)
val nomRestantes = nomCols.filterNot(dropCols.contains)
val numRestantes = numCols.filterNot(dropCols.contains)
val nomImputar = nomRestantes.filter(c => pctAusentes(c) > 0)

val modas: Map[String, String] = nomImputar.map { c =>
  val moda = (trainRaw
    .select(trim(col(c)).as("valor"))
    .filter(col("valor").isNotNull && col("valor") =!= "?")
    .groupBy("valor")
    .count()
    .orderBy(desc("count"))
    .first()
    .getString(0))
  c -> moda
}.toMap

def limpiar(datos: DataFrame): DataFrame = {
  val nominales = nomRestantes.map { c =>
    val v = trim(col(c))
    (if (nomImputar.contains(c)) when(v.isNull || v === "?", lit(modas(c))).otherwise(v) else v).as(c)
  }
  val numericos = numRestantes.map(c => trim(col(c)).cast("double").as(c))
  val etiqueta = when(esPositivo(col(classCol)), 1.0).otherwise(0.0).as("label")
  datos.select(nominales ++ numericos :+ etiqueta: _*)
}

val trainLimpio = limpiar(trainRaw)
val medianas: Map[String, Double] = numRestantes.map(c =>
  c -> trainLimpio.stat.approxQuantile(c, Array(0.5), 0.001)(0)).toMap

val capCols = ceroInfladas.filter(numRestantes.contains)
val trainImp = trainLimpio.na.fill(medianas)
val limites: Map[String, Double] = capCols.map { c =>
  val q = trainImp.filter(col(c) > 0).stat.approxQuantile(c, Array(0.25, 0.75), 0.001)
  c -> (q(1) + K_IQR * (q(1) - q(0)))
}.toMap

def capar(datos: DataFrame): DataFrame = capCols.foldLeft(datos) { (df, c) =>
  df.withColumn(c, least(col(c), lit(limites(c))))
}

val trainPre = capar(trainImp).cache()


/* =============================================================================
   FASE 3 — Selección de atributos (solo entrenamiento, mismas reglas)
   ============================================================================= */

def cramersVCells(cells: Map[(String, String), Double]): Double = {
  val rowTotals = cells.groupBy(_._1._1).map { case (k, v) => k -> v.values.sum }
  val colTotals = cells.groupBy(_._1._2).map { case (k, v) => k -> v.values.sum }
  val n = cells.values.sum
  val dof = (rowTotals.size - 1) * (colTotals.size - 1)
  var chi2 = 0.0
  for ((rk, rt) <- rowTotals; (ck, ct) <- colTotals) {
    val esperado = rt * ct / n
    var diff = cells.getOrElse((rk, ck), 0.0) - esperado
    if (dof == 1) diff = math.signum(diff) * math.max(0.0, math.abs(diff) - 0.5)
    chi2 += diff * diff / esperado
  }
  val k = math.min(rowTotals.size, colTotals.size) - 1
  if (k == 0) 0.0 else math.sqrt(chi2 / (n * k))
}

def cramersV(data: DataFrame, x: String, y: String): Double = {
  val cells = (data
    .select(coalesce(col(x).cast("string"), lit("NULO")).as("x"), coalesce(col(y).cast("string"), lit("NULO")).as("y"))
    .groupBy("x", "y")
    .count()
    .collect()
    .map(r => ((r.getString(0), r.getString(1)), r.getLong(2).toDouble))
    .toMap)
  cramersVCells(cells)
}

val corrCols = numRestantes :+ "label"
val corrVec = (new VectorAssembler()
  .setInputCols(corrCols.toArray)
  .setOutputCol("v")
  .transform(trainPre)
  .select("v"))
val pM = Correlation.corr(corrVec, "v", "pearson").head.getAs[Matrix](0)
val sM = Correlation.corr(corrVec, "v", "spearman").head.getAs[Matrix](0)
val iLabel = corrCols.size - 1

val selNumCols = (numRestantes.zipWithIndex
  .filter { case (_, i) => math.max(math.abs(pM(i, iLabel)), math.abs(sM(i, iLabel))) >= UMBRAL_RELEVANCIA }
  .map(_._1))

val candidatosNom = nomRestantes.filter(c => cramersV(trainPre, c, "label") >= UMBRAL_RELEVANCIA)

val paresSospechosos = Seq(
  ("detailed_industry_recode", "major_industry_code", "major_industry_code"),
  ("detailed_occupation_recode", "major_occupation_code", "major_occupation_code"),
  ("detailed_household_and_family_stat", "detailed_household_summary_in_household", "detailed_household_and_family_stat"),
  ("detailed_household_and_family_stat", "marital_stat", "detailed_household_and_family_stat"),
  ("race", "hispanic_origin", "hispanic_origin"),
  ("country_of_birth_self", "country_of_birth_father", "country_of_birth_father"),
  ("country_of_birth_father", "country_of_birth_mother", "country_of_birth_mother"),
  ("country_of_birth_self", "citizenship", "country_of_birth_self"),
  ("state_of_previous_residence", "region_of_previous_residence", "state_of_previous_residence"))

val descartados = (paresSospechosos
  .filter { case (a, b, _) => candidatosNom.contains(a) && candidatosNom.contains(b) }
  .filter { case (a, b, _) => cramersV(trainPre, a, b) >= UMBRAL_REDUNDANCIA }
  .map(_._3)
  .toSet)

val selNomCols = candidatosNom.filterNot(descartados.contains)

println(s"Nominales seleccionados (${selNomCols.size}): ${selNomCols.mkString(", ")}")
println(s"Numéricos seleccionados (${selNumCols.size}): ${selNumCols.mkString(", ")}")


/* =============================================================================
   FASE 4 — Transformación para árboles: limpieza en SQL (para poder aplicarla a
   datos en bruto desde el modelo guardado) + label encoding + VectorAssembler.
   Los árboles no necesitan 1 de K ni escalado; los índices conservan los metadatos
   nominales y el GBT parte por subconjuntos de categorías.
   ============================================================================= */

def q(c: String): String = "`" + c + "`"

val sqlNominales = selNomCols.map { c =>
  val v = s"trim(${q(c)})"
  if (nomImputar.contains(c))
    s"CASE WHEN $v IS NULL OR $v = '?' THEN '${modas(c).replace("'", "''")}' ELSE $v END AS ${q(c)}"
  else s"$v AS ${q(c)}"
}

val sqlNumericos = selNumCols.map { c =>
  val v = s"COALESCE(CAST(trim(${q(c)}) AS DOUBLE), CAST(${medianas(c)} AS DOUBLE))"
  if (capCols.contains(c)) s"LEAST($v, CAST(${limites(c)} AS DOUBLE)) AS ${q(c)}" else s"$v AS ${q(c)}"
}

val sqlEtiqueta = s"CASE WHEN trim(${q(classCol)}) = '50000+.' THEN CAST(1.0 AS DOUBLE) ELSE CAST(0.0 AS DOUBLE) END AS label"

val preprocesadoSQL = (new SQLTransformer()
  .setStatement("SELECT " + (sqlNominales ++ sqlNumericos :+ sqlEtiqueta).mkString(", ") + " FROM __THIS__"))

val idxCols = selNomCols.map(_ + "_idx")
val indexer = (new StringIndexer()
  .setInputCols(selNomCols.toArray)
  .setOutputCols(idxCols.toArray)
  .setStringOrderType("frequencyDesc")
  .setHandleInvalid("keep"))
val vaArbol = (new VectorAssembler()
  .setInputCols((idxCols ++ selNumCols).toArray)
  .setOutputCol("features"))

val prepModel = new Pipeline().setStages(Array[PipelineStage](preprocesadoSQL, indexer, vaArbol)).fit(trainRaw)

// maxBins >= nº de categorías del nominal con más valores (incluido el reservado)
val maxBinsMin = prepModel.stages(1).asInstanceOf[StringIndexerModel].labelsArray.map(_.length + 1).max
val MAX_BINS = math.max(64, maxBinsMin)
println(s"maxBins = $MAX_BINS (mínimo necesario: $maxBinsMin)")

// Particiones de la validación cruzada estratificadas por clase (columna "fold")
val trainDF = (prepModel.transform(trainRaw)
  .select("features", "label")
  .withColumn("fold", least(floor(percent_rank().over(Window.partitionBy("label").orderBy(rand(SEED + 1))) * K_FOLDS),
    lit(K_FOLDS - 1)).cast("int"))
  // la ventana deja todos los datos en 2 particiones (una por clase): se reparten entre
  // todos los núcleos, si no el GBT entrena en 2 hilos y es muchísimo más lento
  .repartition(spark.sparkContext.defaultParallelism)
  .cache())
trainDF.groupBy("fold").agg(count("*").as("n"), round(avg("label") * 100, 2).as("% positivos")).orderBy("fold").show()


/* =============================================================================
   FASE 5 — Creación y selección del modelo con CrossValidator
   ============================================================================= */

val gbt = (new GBTClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxBins(MAX_BINS)
  .setSubsamplingRate(SUBSAMPLING)
  .setCheckpointInterval(10)
  .setSeed(SEED))

val evaluadorPR = (new BinaryClassificationEvaluator()
  .setLabelCol("label")
  .setRawPredictionCol("rawPrediction")
  .setMetricName("areaUnderPR"))

def validar(grid: Array[ParamMap]): CrossValidatorModel = {
  val cv = (new CrossValidator()
    .setEstimator(gbt)
    .setEstimatorParamMaps(grid)
    .setEvaluator(evaluadorPR)
    .setNumFolds(K_FOLDS)
    .setFoldCol("fold")
    .setParallelism(PARALELISMO)
    .setCollectSubModels(true)
    .setSeed(SEED))
  cv.fit(trainDF)
}

def imprimirCV(titulo: String, m: CrossValidatorModel): Unit = {
  println(s"\n=== $titulo: AUC-PR medio en validación cruzada ($K_FOLDS particiones) ===")
  println("%8s %8s %8s %10s %10s".format("maxDepth", "maxIter", "stepSize", "AUC-PR", "desv."))
  println("-" * 48)
  // desviación entre particiones: se reevalúa cada submodelo en su partición de validación
  val validaciones = (0 until K_FOLDS).map(f => trainDF.filter(col("fold") === f).cache())
  val desv = m.getEstimatorParamMaps.indices.map { i =>
    val ms = (0 until K_FOLDS).map(f => evaluadorPR.evaluate(m.subModels(f)(i).transform(validaciones(f))))
    val media = ms.sum / K_FOLDS
    math.sqrt(ms.map(x => (x - media) * (x - media)).sum / (K_FOLDS - 1))
  }
  validaciones.foreach(_.unpersist())
  val filas = m.getEstimatorParamMaps.zip(m.avgMetrics).zip(desv).sortBy(t => -t._1._2)
  for (((pm, a), s) <- filas) {
    val d = pm(gbt.maxDepth)
    val it = pm(gbt.maxIter)
    val st = pm(gbt.stepSize)
    println(f"$d%8d $it%8d $st%8.2f $a%10.4f $s%10.4f")
  }
}

def mejorParams(m: CrossValidatorModel): ParamMap = m.getEstimatorParamMaps(m.avgMetrics.zipWithIndex.maxBy(_._1)._2)

// Fase 1: grano grueso (profundidad x número de árboles)
val gridGrueso = (new ParamGridBuilder()
  .addGrid(gbt.maxDepth, DEPTHS_GRUESO)
  .addGrid(gbt.maxIter, ITERS_GRUESO)
  .addGrid(gbt.stepSize, Array(STEP_GRUESO))
  .build())
val cvGrueso = validar(gridGrueso)
imprimirCV("Fase 1 (grano grueso)", cvGrueso)
val pGrueso = mejorParams(cvGrueso)
val dGrueso = pGrueso(gbt.maxDepth)
val itGrueso = pGrueso(gbt.maxIter)
println(s"Mejor de la fase 1: maxDepth = $dGrueso, maxIter = $itGrueso")

// Fase 2: grano fino (profundidades vecinas x learning rate), con el mejor maxIter
val depthsFino = Array(dGrueso - 1, dGrueso, dGrueso + 1).filter(d => d >= 2 && d <= 10)
val gridFino = (new ParamGridBuilder()
  .addGrid(gbt.maxDepth, depthsFino)
  .addGrid(gbt.maxIter, Array(itGrueso))
  .addGrid(gbt.stepSize, STEPS_FINO)
  .build())
val cvFino = validar(gridFino)
imprimirCV("Fase 2 (grano fino)", cvFino)

val iMejor = cvFino.avgMetrics.zipWithIndex.maxBy(_._1)._2
val pMejor = cvFino.getEstimatorParamMaps(iMejor)
val aucPRcv = cvFino.avgMetrics(iMejor)
println(f"\nParámetros elegidos: maxDepth = ${pMejor(gbt.maxDepth)}, maxIter = ${pMejor(gbt.maxIter)}, " +
  f"stepSize = ${pMejor(gbt.stepSize)}%.2f, subsamplingRate = $SUBSAMPLING, maxBins = $MAX_BINS")
println(f"AUC-PR en validación cruzada (métrica de selección) = $aucPRcv%.4f")


/* =============================================================================
   FASE 6 — Umbral de decisión con las predicciones fuera de muestra de la CV
   Con un 6% de positivos el umbral 0.5 deja escapar muchos >50k. Cada instancia de
   entrenamiento la predice el submodelo de la partición que NO la contenía.
   ============================================================================= */

val oof = ((0 until K_FOLDS).map { f =>
  (cvFino.subModels(f)(iMejor)
    .transform(trainDF.filter(col("fold") === f))
    .select(vector_to_array(col("probability")).getItem(1).as("p1"), col("label")))
}.reduce(_ union _)
  .collect()
  .map(r => (r.getDouble(0), r.getDouble(1))))

// Recorre los umbrales posibles (ordenando por probabilidad) y devuelve el de mayor F1
def mejorUmbral(a: Array[(Double, Double)]): (Double, Double) = {
  val s = a.sortBy(-_._1)
  val totPos = a.count(_._2 == 1.0).toDouble
  var tp = 0.0
  var fp = 0.0
  var mejor = (0.5, 0.0)
  var i = 0
  while (i < s.length) {
    if (s(i)._2 == 1.0) tp += 1 else fp += 1
    if (i == s.length - 1 || s(i + 1)._1 != s(i)._1) {
      val f1 = 2 * tp / (tp + fp + totPos)
      if (f1 > mejor._2) mejor = (s(i)._1, f1)
    }
    i += 1
  }
  mejor
}

def metricasEn(a: Array[(Double, Double)], u: Double): (Double, Double, Double, Double) = {
  val tp = a.count(r => r._1 >= u && r._2 == 1.0).toDouble
  val fp = a.count(r => r._1 >= u && r._2 == 0.0).toDouble
  val fn = a.count(r => r._1 < u && r._2 == 1.0).toDouble
  val prec = tp / (tp + fp)
  val rec = tp / (tp + fn)
  ((fp + fn) / a.length, prec, rec, 2 * tp / (2 * tp + fp + fn))
}

val (umbralBruto, _) = mejorUmbral(oof)
// Se redondea a 3 decimales hacia abajo para no depender del valor exacto de una instancia
val umbral = math.floor(umbralBruto * 1000) / 1000

println("\n=== Predicciones fuera de muestra de la validación cruzada (solo entrenamiento) ===")
println("%-18s %8s %10s %8s %8s".format("Umbral", "error", "precisión", "recall", "F1"))
println("-" * 56)
Seq(("0.5 (por defecto)", 0.5), (f"$umbral%.3f (elegido)", umbral)).foreach { case (nombre, u) =>
  val (e, p, r, f) = metricasEn(oof, u)
  println(f"$nombre%-18s $e%8.4f $p%10.4f $r%8.4f $f%8.4f")
}


/* =============================================================================
   FASE 7 — Modelo final y guardado
   cvFino.bestModel ya está reentrenado por CrossValidator con TODO el entrenamiento
   y los parámetros elegidos. Se le fija el umbral y se encadena con el preprocesado.
   ============================================================================= */

val mejorGBT = cvFino.bestModel.asInstanceOf[GBTClassificationModel]
mejorGBT.setThresholds(Array(1.0 - umbral, umbral))

// Todas las etapas ya están ajustadas: Pipeline.fit solo las encadena, no reentrena nada
val modeloGBT = new Pipeline().setStages(prepModel.stages :+ mejorGBT).fit(trainRaw)

println(s"\nModelo final: ${mejorGBT.getNumTrees} árboles, ${mejorGBT.totalNumNodes} nodos en total, umbral = $umbral")
println("Importancia de los atributos:")
((idxCols ++ selNumCols).zip(mejorGBT.featureImportances.toArray)
  .sortBy(-_._2)
  .foreach { case (c, v) => println(f"  ${c.stripSuffix("_idx")}%-45s $v%.4f") })

modeloGBT.write.overwrite().save(MODELO)
testRaw.coalesce(1).write.mode("overwrite").csv(PATH + "census-test.data")
println(s"\nGuardados: $MODELO (PipelineModel completo) y ${PATH}census-test.data (partición de prueba en bruto)")
