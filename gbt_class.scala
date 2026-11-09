import org.apache.spark.ml.classification.{GBTClassifier}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.functions.vector_to_array
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute}
import scala.util.{Try, Success, Failure}

val PATH = "./"

// Conjuntos ya preprocesados (columnas: features, label) generados en exploracion.scala
val fullTrain = spark.read.parquet(PATH + "data/census_train.parquet").cache()
val test = spark.read.parquet(PATH + "data/census_test.parquet").cache()

// El test no interviene en la selección: se reserva un 20% del train como validación
val Array(trainSet, valid) = fullTrain.randomSplit(Array(0.8, 0.2), seed = 42L)
val train = trainSet.cache()
valid.cache()

// maxBins debe ser >= al número de categorías de cualquier atributo nominal
val maxCats = AttributeGroup.fromStructField(fullTrain.schema("features")).attributes.get.collect {
  case a: NominalAttribute => a.getNumValues.getOrElse(a.values.map(_.length).getOrElse(0))
}.foldLeft(0)(math.max)
println(s"Máximo de categorías en un atributo nominal: $maxCats")

val evaluator = (new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy"))

// Métrica de selección: F1 de la clase positiva (label = 1, ingreso >= 50K)
val f1PosEvaluator = (new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("fMeasureByLabel")
  .setMetricLabel(1.0))

val numTrees = Seq(20, 50, 100)
val depths   = Seq(5, 8, 12)
val bins     = Seq(64, 96, 128)

val results = for (n <- numTrees; d <- depths; b <- bins) yield {
  val r = Try {
    val m = new GBTClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxIter(n)
      .setMaxDepth(d)
      .setMaxBins(b)
      .setSeed(42L)
      .setThresholds(Array(0.7, 0.3))
      .fit(train)
    val trainF1 = f1PosEvaluator.evaluate(m.transform(train))
    val validF1 = f1PosEvaluator.evaluate(m.transform(valid))
    (m, trainF1, validF1)
  }
  (n, d, b, r)
}

println(f"${"numTrees"}%-9s ${"maxDepth"}%-9s ${"maxBins"}%-8s ${"trainF1"}%-10s ${"validF1"}%-10s")
results.foreach {
  case (n, d, b, Success((_, tr, vf))) =>
    println(f"$n%-9d $d%-9d $b%-8d $tr%-10.5f $vf%-10.5f")
  case (n, d, b, Failure(e)) =>
    println(f"$n%-9d $d%-9d $b%-8d INVALIDA: ${e.getMessage.take(80)}")
}

val ok = results.collect { case (n, d, b, Success((_, tr, vf))) => (n, d, b, tr, vf) }
val (bn, bd, bb, btr, bvf) = ok.maxBy(_._5)
println(s"\nMejor combinación (por F1 clase 1 en validación): numTrees=$bn, maxDepth=$bd, maxBins=$bb, F1Train=$btr, F1Validación=$bvf")

// Modelo final: reentrenado con todo el conjunto de entrenamiento
val finalModel = (new GBTClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxIter(bn)
  .setMaxDepth(bd)
  .setMaxBins(bb)
  .setThresholds(Array(0.7, 0.3))
  .setSeed(42L)
  .fit(fullTrain))

val finalTrainErr = 1.0 - evaluator.evaluate(finalModel.transform(fullTrain))

val testPredictions = finalModel.transform(test)
def metric(name: String): Double = evaluator.setMetricName(name).evaluate(testPredictions)
val accuracy  = metric("accuracy")
val precision = metric("weightedPrecision")
val recall    = metric("weightedRecall")
val f1        = metric("f1")
val testErr   = 1.0 - accuracy
// Métricas de la clase positiva (label = 1, ingreso >= 50K)
evaluator.setMetricLabel(1.0)
val precPos = metric("precisionByLabel")
val recPos  = metric("recallByLabel")
val f1Pos   = metric("fMeasureByLabel")
// Métricas de la clase negativa (label = 0, ingreso < 50K)
evaluator.setMetricLabel(0.0)
val precNeg = metric("precisionByLabel")
val recNeg  = metric("recallByLabel")
val f1Neg   = metric("fMeasureByLabel")
val f1Macro = (f1Pos + f1Neg) / 2.0

// AUC ROC y AUC PR sobre la probabilidad de la clase positiva
val pl = testPredictions.select(
  col("label").cast("double").as("label"),
  vector_to_array(col("probability")).getItem(1).as("p1"))
val hayPositivos = pl.filter(col("label") === 1.0).limit(1).count() > 0
val hayNegativos = pl.filter(col("label") === 0.0).limit(1).count() > 0
val tieneAmbas = hayPositivos && hayNegativos
val evaluadorAUC = (new BinaryClassificationEvaluator()
  .setLabelCol("label")
  .setRawPredictionCol("p1")
  .setNumBins(0)) // sin submuestreo: área exacta
val aucROC = if (tieneAmbas) evaluadorAUC.setMetricName("areaUnderROC").evaluate(pl) else Double.NaN
val aucPR  = if (tieneAmbas) evaluadorAUC.setMetricName("areaUnderPR").evaluate(pl) else Double.NaN

println("\n=========================================")
println("MODELO FINAL GBT\n")
println(s"numIter = $bn")
println(s"maxDepth = $bd")
println(s"maxBins  = $bb\n")
println(s"Error Train = $finalTrainErr\n")
println(s"Error Test  = $testErr\n")
println(s"Accuracy = $accuracy")
println(s"Precision (clase 1) = $precPos")
println(s"Recall (clase 1) = $recPos")
println(s"F1 (clase 1) = $f1Pos")
println(s"Precision (clase 0) = $precNeg")
println(s"Recall (clase 0) = $recNeg")
println(s"F1 (clase 0) = $f1Neg")
println(s"F1 macro (promedio clase 0 y 1) = $f1Macro")
println(s"AUC ROC = $aucROC")
println(s"AUC PR = $aucPR")
println(s"[Referencia, ponderadas] Precision = $precision, Recall = $recall, F1 = $f1")
println("=========================================")
println(s"Feature importances = ${finalModel.featureImportances}")

(finalModel.write
  .overwrite()
  .save(PATH + "ModeloGBT"))
