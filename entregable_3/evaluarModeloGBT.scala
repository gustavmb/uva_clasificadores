/* =============================================================================
   EVALUACIÓN DEL MODELO FINAL — GBT (Gradient-Boosted Trees)

   Uso:  :load evaluarModeloGBT.scala    (sin parámetros; solo hay que ajustar PATH y FILE)

   Lee un conjunto de datos EN BRUTO (mismo formato que census-income.data) y calcula
   las métricas del enunciado. NO crea el modelo: lo carga de ./modeloGBT (generado por
   gbt.scala), un PipelineModel que ya contiene todo el preprocesado aprendido con
   entrenamiento (imputación, capado, selección y codificación) y el umbral de
   decisión elegido en la validación cruzada.

   Por defecto evalúa la partición de prueba que guardó gbt.scala
   (PATH + "census-test.data"). Para evaluar otro subconjunto de los datos
   disponibles basta con cambiar PATH y FILE.
   ============================================================================= */

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.GBTClassificationModel
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

val PATH = "./"
val FILE = "census-test.data"
val MODELO = "./modeloGBT"

// Esquema del fichero en bruto: todo texto; el modelo hace las conversiones
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

def leerCensus(ruta: String): DataFrame = spark.read.format("csv").schema(censusSchema).load(ruta)

/* =============================================================================
   MÉTRICAS DE EVALUACIÓN (clasificador binario, según el enunciado)
   - Tasa de error, desviación estándar e intervalo de confianza al 95%
   - Matriz de confusión
   - Tasa de ciertos positivos (TPR) y de falsos positivos (FPR)
   - Área bajo la curva ROC y curva ROC
   - Área bajo la curva PR
   Además: precisión y F1 de la clase positiva (útiles con un 6% de positivos).
   Se usan ML (BinaryClassificationEvaluator) y MLlib (BinaryClassificationMetrics
   para los puntos de la curva ROC). Funciona con cualquier subconjunto, incluso si
   solo contiene una clase (entonces las áreas no están definidas).
   ============================================================================= */
def evaluarModelo(datos: DataFrame, predicciones: DataFrame, nombre: String, rutaROC: String): Unit = {
  val nInicial = datos.count()

  // probabilidad de la clase positiva (50000+.) como puntuación
  val pl = (predicciones
    .select(col("label").cast("double").as("label"), col("prediction"), vector_to_array(col("probability")).getItem(1).as("p1"))
    .cache())
  val n = pl.count()

  val conteo = (pl.groupBy("label", "prediction").count().collect()
    .map(r => ((r.getDouble(0), r.getDouble(1)), r.getLong(2)))
    .toMap)
  def celda(l: Double, p: Double): Long = conteo.getOrElse((l, p), 0L)
  def div(a: Double, b: Double): Double = if (b == 0) Double.NaN else a / b

  val tn = celda(0.0, 0.0)
  val fp = celda(0.0, 1.0)
  val fn = celda(1.0, 0.0)
  val tp = celda(1.0, 1.0)

  val tasaError = div(fp + fn, n)
  val sdError = math.sqrt(tasaError * (1 - tasaError) / n)
  val icInf = tasaError - 1.96 * sdError
  val icSup = tasaError + 1.96 * sdError
  val tpr = div(tp, tp + fn)
  val fpr = div(fp, fp + tn)
  val precision = div(tp, tp + fp)
  val f1 = div(2 * precision * tpr, precision + tpr)

  val ambasClases = (tp + fn) > 0 && (tn + fp) > 0
  val evaluador = (new BinaryClassificationEvaluator()
    .setLabelCol("label")
    .setRawPredictionCol("p1")
    .setNumBins(0)) // sin submuestreo: área exacta
  val aucROC = if (ambasClases) evaluador.setMetricName("areaUnderROC").evaluate(pl) else Double.NaN
  val aucPR = if (ambasClases) evaluador.setMetricName("areaUnderPR").evaluate(pl) else Double.NaN

  println("\n" + "=" * 64)
  println(s" EVALUACIÓN DEL MODELO $nombre")
  println("=" * 64)
  println(f"Instancias evaluadas: $n%d")
  println(f"Tasa de error:            $tasaError%.4f")
  println(f"Desviación estándar:      $sdError%.4f")
  println(f"IC 95%% de la tasa error:  [$icInf%.4f, $icSup%.4f]")
  println("\nMatriz de confusión (filas = clase real, columnas = predicción):")
  println("%-22s %14s %14s".format("", "pred. - 50000.", "pred. 50000+."))
  println("%-22s %14d %14d".format("real - 50000. (0)", tn, fp))
  println("%-22s %14d %14d".format("real 50000+.  (1)", fn, tp))
  println(f"\nTasa de ciertos positivos (TPR, recall): $tpr%.4f")
  println(f"Tasa de falsos positivos  (FPR):         $fpr%.4f")
  println(f"Precisión (clase positiva):              $precision%.4f")
  println(f"F1 (clase positiva):                     $f1%.4f")
  println(f"Área bajo la curva ROC:                  $aucROC%.4f")
  println(f"Área bajo la curva PR:                   $aucPR%.4f")

  // Curva ROC (MLlib). Se guarda completa en CSV para dibujarla en la memoria y se
  // muestra una selección de puntos.
  if (ambasClases) {
    val bm = new BinaryClassificationMetrics(pl.select("p1", "label").rdd.map(r => (r.getDouble(0), r.getDouble(1))), 100)
    val curva = bm.roc().collect()
    curva.toSeq.toDF("fpr", "tpr").coalesce(1).write.mode("overwrite").option("header", true).csv(rutaROC)
    println(s"\nCurva ROC (${curva.length} puntos, guardada en $rutaROC):")
    println("%10s %10s".format("FPR", "TPR"))
    val paso = math.max(1, curva.length / 20)
    curva.zipWithIndex
      .filter { case (_, i) => i % paso == 0 || i == curva.length - 1 }
      .foreach { case ((x, y), _) => println(f"$x%10.4f $y%10.4f") }
  } else {
    println("\nEl conjunto solo contiene una clase: la curva ROC y las áreas no están definidas.")
  }

  // Apéndice I: ninguna instancia se descarta en el preprocesado (los "?" se imputan,
  // los atípicos se capan), así que NTAIE = NTOE = 0 y no hay instancias sin clasificar.
  val NTAIE = 0L
  val NTOE = nInicial - n - NTAIE
  println(s"\nApéndice I: $nInicial -> $n instancias | NTAIE = $NTAIE | NTOE = $NTOE | " +
    s"tasa de no clasificados = ${(NTAIE + NTOE).toDouble / nInicial}")
  pl.unpersist()
}

val datosEval = leerCensus(PATH + FILE)
val modeloGBT = PipelineModel.load(MODELO)
val gbtFinal = modeloGBT.stages.last.asInstanceOf[GBTClassificationModel]
println(s"Modelo: ${gbtFinal.getNumTrees} árboles, maxDepth = ${gbtFinal.getMaxDepth}, " +
  s"stepSize = ${gbtFinal.getStepSize}, umbral sobre P(>50k) = ${gbtFinal.getThresholds(1)}")
val prediccionesGBT = modeloGBT.transform(datosEval)

evaluarModelo(datosEval, prediccionesGBT, "GBT (Gradient-Boosted Trees)", PATH + "curvaROC_GBT")
