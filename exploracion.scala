import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.attribute.{AttributeGroup, NominalAttribute}
import scala.util.{Try, Success, Failure}

/* Directorio con los datos y conjunto de datos */
val PATH="./"
val DATA="census-income.data"

// Importamos las clases para definir el schema (esquema) del DataFrame:
//   StructType  -> el esquema completo: la lista ordenada de columnas (campos) de cada fila
//   StructField -> una columna: nombre, tipo de dato y si admite valores nulos
//   StringType  -> tipo texto; todos los atributos y la clase de car.data son categóricos,
//                  así que todas las columnas se declaran como String
//   DoubleType  -> tipo numérico (no se usa aquí; se usará más adelante, al pasar a índices numéricos)
val censusSchema = StructType(Array(
    StructField("age", StringType, true),
    StructField("class_of_worker", StringType, true),
    StructField("detailed_industry_recode", StringType, true),
    StructField("detailed_occupation_recode", StringType, true),
    StructField("education", StringType, true),
    StructField("wage_per_hour", StringType, true),
    StructField("enroll_in_edu_inst_last_wk", StringType, true),
    StructField("marital_stat", StringType, true),
    StructField("major_industry_code", StringType, true),
    StructField("major_occupation_code", StringType, true),
    StructField("race", StringType, true),
    StructField("hispanic_origin", StringType, true),
    StructField("sex", StringType, true),
    StructField("member_of_a_labor_union", StringType, true),
    StructField("reason_for_unemployment", StringType, true),
    StructField("full_or_part_time_employment_stat", StringType, true),
    StructField("capital_gains", StringType, true),
    StructField("capital_losses", StringType, true),
    StructField("dividends_from_stocks", StringType, true),
    StructField("tax_filer_stat", StringType, true),
    StructField("region_of_previous_residence", StringType, true),
    StructField("state_of_previous_residence", StringType, true),
    StructField("detailed_household_and_family_stat", StringType, true),
    StructField("detailed_household_summary_in_household", StringType, true),
    StructField("instance_weight", StringType, true),
    StructField("migration_code_change_in_msa", StringType, true),
    StructField("migration_code_change_in_reg", StringType, true),
    StructField("migration_code_move_within_reg", StringType, true),
    StructField("live_in_this_house_1_year_ago", StringType, true),
    StructField("migration_prev_res_in_sunbelt", StringType, true),
    StructField("num_persons_worked_for_employer", StringType, true),
    StructField("family_members_under_18", StringType, true),
    StructField("country_of_birth_father", StringType, true),
    StructField("country_of_birth_mother", StringType, true),
    StructField("country_of_birth_self", StringType, true),
    StructField("citizenship", StringType, true),
    StructField("own_business_or_self_employed", StringType, true),
    StructField("fill_inc_questionnaire_for_veterans_admin", StringType, true),
    StructField("veterans_benefits", StringType, true),
    StructField("weeks_worked_in_year", StringType, true),
    StructField("year", StringType, true),
    StructField("income", StringType, true) // class label: '- 50000.' or '50000+.'
))

// Leemos el DataFrame con la estructura definida
val rawCensusDF = spark.read.format("csv").schema(censusSchema).load(PATH + DATA)

// Muestra las primeras 5 columnas y 5 lineas
rawCensusDF.select(rawCensusDF.columns.take(5).map(col): _*).show(5)

// Lista de atributos
val classCol = "income"
val contCols = Seq("age", "wage_per_hour", "capital_gains",
  "capital_losses", "dividends_from_stocks", "instance_weight")
val discCols = Seq("num_persons_worked_for_employer", "weeks_worked_in_year")
val numCols = contCols ++ discCols
// these are codes, not numbers
val codeCols = Seq("detailed_industry_recode", "detailed_occupation_recode",
  "own_business_or_self_employed", "veterans_benefits", "year")

// Cantidad de classes positivas y negativas
val totalRows = rawCensusDF.count()
(rawCensusDF
  .groupBy(trim(col(classCol)).as(classCol))
  .count()
  .withColumn("label", when(col(classCol) === "50000+.", "positive").otherwise("negative"))
  .withColumn("percentage", round(col("count") * 100.0 / totalRows, 2))
  .orderBy(desc("count"))
  .show())

/*         
Los valores "NA" en hispanic_origin non son nulos.
Los verdaderos nulos son los "?" que hay en las columnas mencionadas abajo (qCols).
*/

// Cantidad de NA's en hispanic_origin
val naHispanic: Long = (rawCensusDF
  .filter(col("hispanic_origin").isNull || trim(col("hispanic_origin")) === "NA")
  .count())

println(s"NA's en hispanic_origin: $naHispanic de $totalRows (${math.round(naHispanic * 10000.0 / totalRows) / 100.0}%)")

// Cantidad de "?" por columnas (solo en las columnas que tienen "?")
val qCols = Seq(
  "state_of_previous_residence",
  "migration_code_change_in_msa",
  "migration_code_change_in_reg",
  "migration_code_move_within_reg",
  "migration_prev_res_in_sunbelt",
  "country_of_birth_father",
  "country_of_birth_mother",
  "country_of_birth_self"
)

val qCounts = rawCensusDF.select(qCols.map(c => sum(when(trim(col(c)) === "?", 1).otherwise(0)).as(c)): _*).first()

println("%-35s %11s".format("Nombre de Columna", "Incidencias"))
println("-" * 47)
qCols.foreach(c => println(f"$c%-35s ${qCounts.getAs[Long](c)}%11d"))

// La distribución de los atributos: 6 continuos, 2 discretos, 0 booleanos y 33 nominales
val nNominales = rawCensusDF.columns.length - 1 - numCols.size // sin la clase ni las numéricas

(Seq(("continuos", contCols.size), ("discretos", discCols.size), ("booleanos", 0), ("nominales", nNominales))
  .toDF("tipo", "count")
  .show())

// Instancias vacías, inválidas y duplicadas
// vacía: todas las columnas son null (el CSV lee los campos vacíos como null)
val nEmpty = totalRows - rawCensusDF.na.drop("all").count()
// inválida: la clase no es "- 50000." ni "50000+."
val nInvalid = rawCensusDF.filter(!trim(col(classCol)).isin("- 50000.", "50000+.")).count()
// duplicada: fila repetida (total - distintas)
val nDup = totalRows - rawCensusDF.distinct().count()

/*
Aunque haya varias columnas duplicadas, hemos decidido no eliminar, porque:
- Los datos de censo, dos personas diferentes pueden tener exactamente los 
mismos valores en todas las columnas.
*/

(Seq(("vacías", nEmpty), ("inválidas", nInvalid), ("duplicadas", nDup))
  .toDF("instancias", "count")
  .withColumn("percentage", round(col("count") * 100.0 / totalRows, 2))
  .show())

// Estadísticas de los atributos Numéricos (mín., máx., media, desviación, distribución, valores atípicos)
/*
Además de count, mean, std, min, quartis y max, se añaden:
- skew     = asimetría (cerca de 0 es simétrica, arriba de 1 muy inclinada a la derecha)
- %zeros   = porcentaje de valores iguales a 0
- n_unicos = cuántos valores distintos hay
*/
val numDF = rawCensusDF.select(numCols.map(c => trim(col(c)).cast("double").as(c)): _*)

// Una fila por atributo, con las estadísticas como columnas
val statsDF = numCols.map { c =>
  val x = col(c)
  numDF.agg(
    lit(c).as("atributo"),
    count(x).as("count"),
    mean(x).as("mean"),
    stddev(x).as("std"),
    min(x).as("min"),
    expr(s"percentile_approx(`$c`, array(0.25, 0.5, 0.75))").as("q"),
    max(x).as("max"),
    skewness(x).as("skew"),
    (avg(when(x === 0, 1.0).otherwise(0.0)) * 100).as("%zeros"),
    countDistinct(x).as("n_unicos"))
}.reduce(_ union _)

(statsDF
  .select(col("atributo"), col("count"), round(col("mean"), 2).as("mean"), round(col("std"), 2).as("std"),
    col("min"), col("q")(0).as("25%"), col("q")(1).as("50%"), col("q")(2).as("75%"), col("max"),
    round(col("skew"), 2).as("skew"), round(col("%zeros"), 2).as("%zeros"), col("n_unicos"))
  .show(false))

// Estadísticas de los atributos nominales (valores posibles, moda, frecuencia)
val nomCols = rawCensusDF.columns.filterNot(c => numCols.contains(c) || c == classCol).toSeq

val nomStats = nomCols.map { c =>
  rawCensusDF
    .groupBy(trim(col(c)).as("valor"))
    .count()
    .agg(lit(c).as("atributo"), count("*").as("valores_posibles"), max(struct(col("count"), col("valor"))).as("m"))
}.reduce(_ union _)

(nomStats
  .select(col("atributo"), col("valores_posibles"), col("m.valor").as("moda"), col("m.count").as("frecuencia"))
  .show(nomCols.size, false))

// Ausentes (?) y “Not in universe”
val ausentes = nomCols.map { c =>
  rawCensusDF.agg(
    lit(c).as("atributo"),
    sum(when(trim(col(c)) === "Not in universe", 1).otherwise(0)).as("not_in_universe"),
    sum(when(trim(col(c)) === "?", 1).otherwise(0)).as("interrogacion"))
}.reduce(_ union _)
/*
"?" es un valor ausente. "Not in universe" es una categoría válida y no cuenta como ausente
*/
(ausentes
  .filter(col("not_in_universe") > 0 || col("interrogacion") > 0)
  .withColumn("not_in_universe_%", round(col("not_in_universe") * 100.0 / totalRows, 2))
  .withColumn("interrogacion_%", round(col("interrogacion") * 100.0 / totalRows, 2))
  .select("atributo", "not_in_universe", "not_in_universe_%", "interrogacion", "interrogacion_%")
  .orderBy(desc("interrogacion"))
  .show(nomCols.size, false))

// Valores atípicos (outliers)
val outCols = contCols.filterNot(_ == "instance_weight") // quitamos por ser muestral
val ceroInfladas = outCols.filterNot(_ == "age")

val outliers = outCols.map { c =>
  val base = if (ceroInfladas.contains(c)) numDF.filter(col(c) > 0) else numDF
  val q = base.stat.approxQuantile(c, Array(0.25, 0.75), 0.001)
  val lim = q(1) + 3 * (q(1) - q(0))
  numDF.agg(
    lit(c).as("atributo"),
    lit(q(0)).as("q1"),
    lit(q(1)).as("q3"),
    lit(lim).as("limite_sup"),
    sum(when(col(c) > lim, 1).otherwise(0)).as("n_atipicos"))
}.reduce(_ union _)

(outliers
  .withColumn("atipicos_%", round(col("n_atipicos") * 100.0 / totalRows, 2))
  .show(false))


// Matriz de correlación (Pearson y Spearman) entre los atributos numéricos y la clase
/*
- Pearson  = relación lineal entre dos atributos
- Spearman = relación monótona (sobre los rangos), más robusta a atípicos y a las
             distribuciones muy inclinadas (skew) de este conjunto de datos
La clase se añade como "income_bin" (1 = "50000+.", 0 = "- 50000.") para ver la
correlación de cada atributo numérico con la variable objetivo.
*/
val corrCols = numCols :+ "income_bin"

val corrInputDF = (rawCensusDF
  .select(numCols.map(c => trim(col(c)).cast("double").as(c)) :+
    when(trim(col(classCol)) === "50000+.", 1.0).otherwise(0.0).as("income_bin"): _*)
  .na.drop(corrCols))

val corrVecDF = (new VectorAssembler()
  .setInputCols(corrCols.toArray)
  .setOutputCol("features")
  .transform(corrInputDF)
  .select("features"))

// La matriz (atributo x atributo) como DataFrame, redondeada a 2 decimales
def corrMatrixDF(m: Matrix, cols: Seq[String]): DataFrame =
  ((for (i <- cols.indices; j <- cols.indices) yield (i, cols(i), cols(j), m(i, j)))
    .toDF("orden", "atributo", "columna", "valor")
    .groupBy("orden", "atributo")
    .pivot("columna", cols)
    .agg(round(first("valor"), 2))
    .orderBy("orden")
    .drop("orden"))

val pearsonM = Correlation.corr(corrVecDF, "features", "pearson").head.getAs[Matrix](0)
val spearmanM = Correlation.corr(corrVecDF, "features", "spearman").head.getAs[Matrix](0)

println("Matriz de correlación de Pearson")
(corrMatrixDF(pearsonM, corrCols)
  .show(false))

println("Matriz de correlación de Spearman")
(corrMatrixDF(spearmanM, corrCols)
  .show(false))

// Pares de atributos ordenados por correlación absoluta (sin la diagonal ni repetidos)
val paresCorr = for {
  i <- corrCols.indices
  j <- (i + 1) until corrCols.size
} yield (corrCols(i), corrCols(j), pearsonM(i, j), spearmanM(i, j))

(paresCorr
  .toDF("atributo_1", "atributo_2", "pearson", "spearman")
  .withColumn("pearson", round(col("pearson"), 3))
  .withColumn("spearman", round(col("spearman"), 3))
  .orderBy(abs(col("spearman")).desc)
  .show(paresCorr.size, false))

// Limpieza: columnas que se eliminan y ausentes que se imputan
/*
Se eliminan las columnas que no aportan información:
- las de migración: ~50% de "?" (demasiados ausentes para imputar con fiabilidad)
  y además las cuatro describen lo mismo, que ya recoge live_in_this_house_1_year_ago
- instance_weight: es el peso muestral del censo, no un atributo de la persona
  (la documentación indica ignorarla; su correlación con todo lo demás es ~0)
- wage_per_hour: ~95% ceros y correlación nula con la clase (Pearson 0.02, Spearman -0.01)
- year: solo toma los valores 94 y 95; es metadato de la recogida de datos
*/
val dropCols = Seq(
  "migration_code_change_in_msa",
  "migration_code_change_in_reg",
  "migration_code_move_within_reg",
  "migration_prev_res_in_sunbelt",
  "instance_weight",
  "wage_per_hour",
  "year")

/*
En el resto de columnas los "?" son pocos (~2-3%), así que no se eliminan las filas:
se sustituyen por la moda (el valor más frecuente de cada columna, sin contar los "?").
Borrar esas filas descartaría instancias que son válidas en todos los demás atributos.
*/
val imputeCols = qCols.diff(dropCols)

/*      Partición en entrenamiento (66%) y prueba (34%)              */

/*
La partición se hace tras la exploración y antes de cualquier transformación que
aprenda de los datos (modas, límites de capado, selección de atributos, indexadores).
Así, el conjunto de prueba no influye en esos valores (no hay fuga de información):
se calculan solo con entrenamiento y se aplican tal cual a prueba.

La partición es estratificada: con solo un 6% de instancias positivas, un
randomSplit simple podría alterar la proporción de clases entre las particiones.
sampleBy toma el 66% de cada clase por separado, manteniendo el desbalance
original en ambos conjuntos. El test es el complementario (left_anti sobre el id).
*/
val conId = (rawCensusDF
  .withColumn("id", monotonically_increasing_id())
  .withColumn("strat", when(trim(col(classCol)) === "50000+.", 1.0).otherwise(0.0))
  .cache())

val trainIdDF = conId.stat.sampleBy("strat", Map(0.0 -> 0.66, 1.0 -> 0.66), 0L).cache()
val testIdDF = conId.join(trainIdDF.select("id"), Seq("id"), "left_anti").cache()

// Comprobamos que la proporción de clases se mantiene en ambas particiones
Seq(("train", trainIdDF), ("test", testIdDF), ("total", conId)).foreach { case (nombre, df) =>
  val n = df.count()
  val pos = df.filter(col("strat") === 1.0).count()
  println(f"$nombre%-6s n=$n%8d  positivos=$pos%7d (${pos * 100.0 / n}%.2f%%)")
}

val trainDF = trainIdDF.drop("id", "strat")
val testDF = testIdDF.drop("id", "strat")

// Moda de cada columna a imputar (solo con entrenamiento), ignorando los propios "?"
val modas: Map[String, String] = imputeCols.map { c =>
  val moda = trainDF
    .select(trim(col(c)).as("valor"))
    .filter(col("valor") =!= "?")
    .groupBy("valor")
    .count()
    .orderBy(desc("count"))
    .first()
    .getAs[String]("valor")
  c -> moda
}.toMap

println("%-35s %s".format("Columna", "Moda imputada"))
println("-" * 60)
imputeCols.foreach(c => println("%-35s %s".format(c, modas(c))))

// Se aplica igual a entrenamiento y prueba, con las modas de entrenamiento
def limpiar(df: DataFrame): DataFrame = (imputeCols
  .foldLeft(df.drop(dropCols: _*)) { (d, c) =>
    d.withColumn(c, when(trim(col(c)) === "?", modas(c)).otherwise(trim(col(c))))
  })

val cleanTrainDF = limpiar(trainDF)
val cleanTestDF = limpiar(testDF)

println(s"\nColumnas reducidas tras eliminacion de columnas irrelevantes: ${rawCensusDF.columns.length} -> ${cleanTrainDF.columns.length}")

// Comprobación: ya no quedan "?" en las columnas imputadas
println("\nComprobación de columnas imputadas (train y test). Valor '?' no presente")
Seq(("train", cleanTrainDF), ("test", cleanTestDF)).foreach { case (nombre, df) =>
  println(nombre)
  df.select(imputeCols.map(c => sum(when(col(c) === "?", 1).otherwise(0)).as(c)): _*).show(false)
}

// Tratamiento de los valores atípicos: capado (winsorización)
/*
Los atípicos no son errores de captura, sino valores reales y poco frecuentes.
No se eliminan las filas porque se concentran en la clase positiva, que ya es
minoritaria (~6%): descartarlas quitaría justamente las instancias más informativas.
En su lugar se capan al límite superior (q3 + 3*IQR), lo que conserva la instancia
y su orden de magnitud, pero neutraliza el efecto de la cola sobre media y varianza.
Los cuantiles se calculan sobre los valores > 0, porque son columnas cero-infladas
y de lo contrario q1 y q3 serían ambos 0.
age no se incluye: no presenta ningún valor atípico.
Los límites se calculan solo con entrenamiento y se aplican también a prueba.
*/
println("\nValores Atipicos (Outliers): ")
val capCols = Seq("capital_gains", "capital_losses", "dividends_from_stocks").diff(dropCols)

val numTrainDF = trainDF.select(numCols.map(c => trim(col(c)).cast("double").as(c)): _*)

val limites: Map[String, Double] = capCols.map { c =>
  val q = numTrainDF.filter(col(c) > 0).stat.approxQuantile(c, Array(0.25, 0.75), 0.001)
  c -> (q(1) + 3 * (q(1) - q(0)))
}.toMap

// Las columnas capadas pasan a ser numéricas (double)
def capar(df: DataFrame): DataFrame = capCols.foldLeft(df) { (d, c) =>
  val x = trim(col(c)).cast("double")
  d.withColumn(c, when(x > limites(c), lit(limites(c))).otherwise(x))
}

val cappedTrainDF = capar(cleanTrainDF)
val cappedTestDF = capar(cleanTestDF)

// Cuántos valores se han capado en cada columna
Seq(("train", cappedTrainDF), ("test", cappedTestDF)).foreach { case (nombre, df) =>
  println(nombre)
  df.select(capCols.map(c => sum(when(col(c) === limites(c), 1).otherwise(0)).as(c)): _*).show(false)
}

println("\n Aplicando windsorizacion:")
println("%-25s %12s".format("Columna", "Límite sup."))
println("-" * 38)
capCols.foreach(c => println("%-25s %12.1f".format(c, limites(c))))

// Selección de atributos: asociación de cada nominal con la clase (Cramér's V)
/*
Pearson y Spearman no sirven para los atributos nominales. Para medirlos usamos
Cramér's V, que normaliza el estadístico chi-cuadrado al intervalo [0, 1] y permite
comparar atributos con distinto número de categorías:
  0.00-0.05  sin asociación      0.05-0.10  débil
  0.10-0.20  moderada            > 0.20     fuerte
Se aplica la corrección de Yates cuando los grados de libertad son 1 (tablas 2x2).
Se calcula solo sobre entrenamiento, para que la selección no dependa de prueba.
*/
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
  math.sqrt(chi2 / (n * k))
}

// La tabla de contingencia se agrega en Spark; solo se recoge la tabla pequeña
def cramersV(data: DataFrame, x: String, y: String): Double = {
  val cells = data
    .select(coalesce(trim(col(x)), lit("NULO")).as("x"), coalesce(trim(col(y)), lit("NULO")).as("y"))
    .groupBy("x", "y")
    .count()
    .collect()
    .map(r => ((r.getString(0), r.getString(1)), r.getLong(2).toDouble))
    .toMap
  cramersVCells(cells)
}

// Nominales que quedan tras la limpieza
val nomColsClean = (cappedTrainDF.columns
  .filterNot(c => numCols.contains(c) || c == classCol)
  .toSeq)

// Ranking de asociación con la clase, junto al número de categorías de cada atributo
val cardinalidad = (cappedTrainDF
  .select(nomColsClean.map(c => countDistinct(trim(col(c))).as(c)): _*)
  .first())

val ranking = nomColsClean.map { c =>
  (c, cardinalidad.getAs[Long](c), cramersV(cappedTrainDF, c, classCol))
}

(ranking
  .toDF("atributo", "n_categorias", "cramers_v")
  .withColumn("cramers_v", round(col("cramers_v"), 3))
  .withColumn("asociacion", when(col("cramers_v") >= 0.2, "fuerte")
    .when(col("cramers_v") >= 0.1, "moderada")
    .when(col("cramers_v") >= 0.05, "debil")
    .otherwise("ninguna"))
  .orderBy(desc("cramers_v"))
  .show(nomColsClean.size, false))

// Redundancia entre pares de nominales sospechosos (V alto = informacion duplicada)
val paresRedundantes = (Seq(
  ("detailed_industry_recode", "major_industry_code"),
  ("detailed_occupation_recode", "major_occupation_code"),
  ("detailed_household_and_family_stat", "detailed_household_summary_in_household"),
  ("detailed_household_and_family_stat", "marital_stat"),
  ("race", "hispanic_origin"),
  ("country_of_birth_self", "country_of_birth_father"),
  ("country_of_birth_father", "country_of_birth_mother"),
  ("country_of_birth_self", "citizenship"),
  ("state_of_previous_residence", "region_of_previous_residence"))
  .filter { case (a, b) => nomColsClean.contains(a) && nomColsClean.contains(b) })

(paresRedundantes
  .map { case (a, b) => (a, b, cramersV(cappedTrainDF, a, b)) }
  .toDF("atributo_1", "atributo_2", "cramers_v")
  .withColumn("cramers_v", round(col("cramers_v"), 3))
  .orderBy(desc("cramers_v"))
  .show(false))

// Selección final de atributos
/*
De los 42 atributos originales se conservan 17 + la clase. Los criterios han sido:
- numéricos: correlación de Pearson/Spearman con la clase
- nominales: Cramér's V con la clase (se conservan los de asociación >= 0.14)
- redundancia: de cada par con V muy alto se conserva un solo atributo

Nota sobre la cardinalidad: Cramér's V se infla cuando el atributo tiene muchas
categorías, así que la ventaja de detailed_occupation_recode (47 cat., V=0.438)
sobre major_occupation_code (15 cat., V=0.366) es en parte sesgo y no señal real.
Como V entre ambos es 1.0 (el código detallado determina el agregado), se conserva
el detallado para modelos basados en árboles.
*/
val selNomCols = Seq(
  "detailed_occupation_recode",              // V=0.438
  "education",                               // V=0.390
  "detailed_industry_recode",                // V=0.293
  "class_of_worker",                         // V=0.261
  "tax_filer_stat",                          // V=0.229
  "detailed_household_summary_in_household", // V=0.227 (8 cat. frente a 38 de la detallada)
  "marital_stat",                            // V=0.198 (V=0.476 con el hogar: complementaria)
  "full_or_part_time_employment_stat",       // V=0.160
  "family_members_under_18",                 // V=0.159
  "sex",                                     // V=0.158
  "veterans_benefits")                       // V=0.144

/*
num_persons_worked_for_employer se mantiene por ser útil en árboles de decisión,
pero debe eliminarse en modelos lineales por su correlación de 0.88 con
weeks_worked_in_year.
*/
val selNumCols = Seq(
  "weeks_worked_in_year",            // Spearman 0.26
  "dividends_from_stocks",           // Spearman 0.27
  "capital_gains",                   // Pearson  0.24
  "capital_losses",                  // Pearson  0.15
  "age",                             // Spearman 0.16
  "num_persons_worked_for_employer") // Spearman 0.23

def seleccionar(df: DataFrame): DataFrame = df.select(
  selNomCols.map(c => trim(col(c)).as(c)) ++
  selNumCols.map(c => trim(col(c)).cast("double").as(c)) :+
  when(trim(col(classCol)) === "50000+.", 1.0).otherwise(0.0).as("label"): _*)

val finalTrainDF = seleccionar(cappedTrainDF)
val finalTestDF = seleccionar(cappedTestDF)

println(s"Atributos: ${rawCensusDF.columns.length} -> ${finalTrainDF.columns.length} (incluida la clase)")
finalTrainDF.printSchema()
finalTrainDF.show(5, false)

/*      Transformación a columnas features y label                  */

/*
Los 11 atributos nominales se pasan a índices numéricos con StringIndexer.
Los 6 numéricos NO se indexan: ya son double y pasan directos al VectorAssembler.
Indexarlos convertiría cada valor distinto en una categoría (90 edades -> 90
categorías) y destruiría la relación de orden entre ellos.
*/
val attributeColumns = selNomCols.toArray
val outputColumns = attributeColumns.map(_ + "-num")

/*
setStringOrderType("alphabetDesc"): el índice depende solo del valor y no de su
frecuencia, así que la asignación es la misma en cualquier partición.
setHandleInvalid("keep"): algunas categorías son muy poco frecuentes (códigos de
ocupación con unos pocos casos) y pueden aparecer en prueba sin haber aparecido en
entrenamiento. Con "keep" reciben un índice reservado en lugar de provocar un error.
*/
val siColumns = (new StringIndexer()
  .setInputCols(attributeColumns)
  .setOutputCols(outputColumns)
  .setStringOrderType("alphabetDesc")
  .setHandleInvalid("keep"))

val simColumns = siColumns.fit(finalTrainDF) // se ajusta solo con entrenamiento

def indexar(df: DataFrame): DataFrame = simColumns.transform(df).drop(attributeColumns: _*)

val trainNumericDF = indexar(finalTrainDF)
val testNumericDF = indexar(finalTestDF)

trainNumericDF.show(5, false)

/*
Los índices entran directamente en features.
Los árboles de Spark tratan de forma nativa los atributos categóricos indexados y
particionan por subconjuntos de categorías, sin necesidad de una codificación 1 de K.
*/
val vaNum = (new VectorAssembler()
  .setInputCols(outputColumns ++ selNumCols.toArray)
  .setOutputCol("features"))

val trainCensusDF = vaNum.transform(trainNumericDF).select("features", "label")
val testCensusDF = vaNum.transform(testNumericDF).select("features", "label")

trainCensusDF.show(5, false)

/*
Se guardan los conjuntos ya transformados para reutilizarlos en el entregable
individual sin repetir todo el preprocesado.
*/
trainCensusDF.write.mode("overwrite").parquet(PATH + "data/census_train.parquet")
testCensusDF.write.mode("overwrite").parquet(PATH + "data/census_test.parquet")

/*                      Árbol de decisión                           */

val fullTrain = trainCensusDF
// El test no interviene en la selección: se reserva un 20% del train como validación
val Array(trainSet, valid) = fullTrain.randomSplit(Array(0.8, 0.2), seed = 42L)
val train = trainSet.cache()
valid.cache()
val test = testCensusDF.cache()

// Mapeo índice -> variable, con tipo y número de categorías (según los metadatos de `features`)
val attrs = AttributeGroup.fromStructField(fullTrain.schema("features")).attributes.get
println(f"${"idx"}%-4s ${"variable"}%-46s ${"tipo"}%-10s categorías")
attrs.foreach {
  case a: NominalAttribute =>
    println(f"${a.index.get}%-4d ${a.name.getOrElse("?")}%-46s ${"nominal"}%-10s ${a.getNumValues.getOrElse(a.values.map(_.length).getOrElse(0))}")
  case a =>
    println(f"${a.index.get}%-4d ${a.name.getOrElse("?")}%-46s ${a.attrType.name}%-10s -")
}
println()

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

val depths = Seq(3, 5, 8, 12)
val bins   = Seq(64, 96, 128)

// Resto de parámetros con su valor por defecto
val results = for (d <- depths; b <- bins) yield {
  val r = Try {
    val m = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setMaxDepth(d)
      .setMaxBins(b)
      .fit(train)
    val trainErr = 1.0 - evaluator.evaluate(m.transform(train))
    val validErr = 1.0 - evaluator.evaluate(m.transform(valid))
    (m, trainErr, validErr)
  }
  (d, b, r)
}

println(f"${"maxDepth"}%-9s ${"maxBins"}%-8s ${"trainErr"}%-10s ${"validErr"}%-10s nodes")
results.foreach {
  case (d, b, Success((m, tr, ve))) =>
    println(f"$d%-9d $b%-8d $tr%-10.5f $ve%-10.5f ${m.numNodes}")
  case (d, b, Failure(e)) =>
    println(f"$d%-9d $b%-8d INVALIDA: ${e.getMessage.take(80)}")
}

val ok = results.collect { case (d, b, Success((m, tr, ve))) => (d, b, m, tr, ve) }
val (bd, bb, bm, btr, bte) = ok.minBy(_._5)
println(s"\nMejor combinación (por validación): maxDepth=$bd, maxBins=$bb, errorTrain=$btr, errorValidación=$bte")

// Modelo final: reentrenado con todo el conjunto de entrenamiento
val finalModel = new DecisionTreeClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxDepth(bd)
  .setMaxBins(bb)
  .fit(fullTrain)

val trainPredictions = finalModel.transform(fullTrain)
val finalTrainErr = 1.0 - evaluator.evaluate(trainPredictions)
println(s"Error Train Final = $finalTrainErr")

// Evaluación final sobre test con el modelo final
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

println("\n=========================================")
println("MODELO FINAL\n")
println(s"maxDepth = $bd")
println(s"maxBins  = $bb\n")
println(s"Número de nodos = ${finalModel.numNodes}\n")
println(s"Error Train = $finalTrainErr\n")
println(s"Error Test  = $testErr\n")
println(s"Accuracy = $accuracy")
println(s"Precision (clase 1) = $precPos")
println(s"Recall (clase 1) = $recPos")
println(s"F1 (clase 1) = $f1Pos")
println(s"Precision (clase 0) = $precNeg")
println(s"Recall (clase 0) = $recNeg")
println(s"F1 (clase 0) = $f1Neg")
println(s"[Referencia, ponderadas] Precision = $precision, Recall = $recall, F1 = $f1")
println("=========================================")
println(finalModel.toDebugString)
println(s"Feature importances = ${finalModel.featureImportances}")

// Guardar el modelo
finalModel.write
  .overwrite()
  .save(PATH + "Modelo")
