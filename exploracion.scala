import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.StringIndexerModel
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation

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
val fullCensusDF = spark.read.format("csv").schema(censusSchema).load(PATH + DATA)

// Muestra las primeras 5 columnas y 5 lineas
fullCensusDF.select(fullCensusDF.columns.take(5).map(col): _*).show(5)

// Lista de atributos
val classCol = "income"
val contCols = Seq("age", "wage_per_hour", "capital_gains",
  "capital_losses", "dividends_from_stocks", "instance_weight")
val discCols = Seq("num_persons_worked_for_employer", "weeks_worked_in_year")
val numCols = contCols ++ discCols
// these are codes, not numbers
val codeCols = Seq("detailed_industry_recode", "detailed_occupation_recode",
  "own_business_or_self_employed", "veterans_benefits", "year")

/*      Partición en entrenamiento (66%) y prueba (34%)              */

/*
La partición se hace justo después de cargar los datos y ANTES de cualquier análisis.
Toda la exploración, el análisis de asociación (Pearson, Spearman, Cramér's V) y las
decisiones de limpieza y selección de atributos se hacen únicamente con train. El test
no se usa para construir el modelo ni para tomar decisiones: solo recibe después las
transformaciones aprendidas en train.
La partición es estratificada: con solo un 6% de instancias positivas, un
randomSplit simple podría alterar la proporción de clases entre las particiones.
sampleBy toma el 66% de cada clase por separado, manteniendo el desbalance
original en ambos conjuntos. El test es el complementario (left_anti sobre el id).
*/
val baseDF = (fullCensusDF
  .withColumn("label", when(trim(col("income")) === "50000+.", 1.0).otherwise(0.0))
  .withColumn("id", monotonically_increasing_id())
  .cache())

val trainRawDF = baseDF.stat.sampleBy("label", Map(0.0 -> 0.66, 1.0 -> 0.66), 0L).cache()
val testRawDF = baseDF.join(trainRawDF.select("id"), Seq("id"), "left_anti").cache()

// Comprobamos que la proporción de clases se mantiene en ambas particiones
Seq(("train", trainRawDF), ("test", testRawDF), ("total", baseDF)).foreach { case (nombre, df) =>
  val n = df.count()
  val pos = df.filter(col("label") === 1.0).count()
  println(f"$nombre%-6s n=$n%8d  positivos=$pos%7d (${pos * 100.0 / n}%.2f%%)")
}

// Datos de train con las columnas originales: base de toda la exploración
val trainExpDF = trainRawDF.drop("id", "label")
val nTrain = trainExpDF.count()

// Cantidad de classes positivas y negativas (train)
(trainExpDF
  .groupBy(trim(col(classCol)).as(classCol))
  .count()
  .withColumn("label", when(col(classCol) === "50000+.", "positive").otherwise("negative"))
  .withColumn("percentage", round(col("count") * 100.0 / nTrain, 2))
  .orderBy(desc("count"))
  .show())

/*         
Los valores "NA" en hispanic_origin non son nulos.
Los verdaderos nulos son los "?" que hay en las columnas mencionadas abajo (qCols).
*/

// Cantidad de NA's en hispanic_origin
val naHispanic: Long = (trainExpDF
  .filter(col("hispanic_origin").isNull || trim(col("hispanic_origin")) === "NA")
  .count())

println(s"NA's en hispanic_origin: $naHispanic de $nTrain (${math.round(naHispanic * 10000.0 / nTrain) / 100.0}%)")

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

val qCounts = trainExpDF.select(qCols.map(c => sum(when(trim(col(c)) === "?", 1).otherwise(0)).as(c)): _*).first()

println("%-35s %11s".format("Nombre de Columna", "Incidencias"))
println("-" * 47)
qCols.foreach(c => println(f"$c%-35s ${qCounts.getAs[Long](c)}%11d"))

// La distribución de los atributos: 6 continuos, 2 discretos, 0 booleanos y 33 nominales
val nNominales = trainExpDF.columns.length - 1 - numCols.size // sin la clase ni las numéricas

(Seq(("continuos", contCols.size), ("discretos", discCols.size), ("booleanos", 0), ("nominales", nNominales))
  .toDF("tipo", "count")
  .show())

// Instancias vacías, inválidas y duplicadas
// vacía: todas las columnas son null (el CSV lee los campos vacíos como null)
val nEmpty = nTrain - trainExpDF.na.drop("all").count()
// inválida: la clase no es "- 50000." ni "50000+."
val nInvalid = trainExpDF.filter(!trim(col(classCol)).isin("- 50000.", "50000+.")).count()
// duplicada: fila repetida (total - distintas)
val nDup = nTrain - trainExpDF.distinct().count()

/*
Aunque haya varias columnas duplicadas, hemos decidido no eliminar, porque:
- Los datos de censo, dos personas diferentes pueden tener exactamente los 
mismos valores en todas las columnas.
*/

(Seq(("vacías", nEmpty), ("inválidas", nInvalid), ("duplicadas", nDup))
  .toDF("instancias", "count")
  .withColumn("percentage", round(col("count") * 100.0 / nTrain, 2))
  .show())

// Estadísticas de los atributos Numéricos (mín., máx., media, desviación, distribución, valores atípicos)
/*
Además de count, mean, std, min, quartis y max, se añaden:
- skew     = asimetría (cerca de 0 es simétrica, arriba de 1 muy inclinada a la derecha)
- %zeros   = porcentaje de valores iguales a 0
- n_unicos = cuántos valores distintos hay
*/
val numDF = trainExpDF.select(numCols.map(c => trim(col(c)).cast("double").as(c)): _*)

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
val nomCols = trainExpDF.columns.filterNot(c => numCols.contains(c) || c == classCol).toSeq

val nomStats = nomCols.map { c =>
  trainExpDF
    .groupBy(trim(col(c)).as("valor"))
    .count()
    .agg(lit(c).as("atributo"), count("*").as("valores_posibles"), max(struct(col("count"), col("valor"))).as("m"))
}.reduce(_ union _)

(nomStats
  .select(col("atributo"), col("valores_posibles"), col("m.valor").as("moda"), col("m.count").as("frecuencia"))
  .show(nomCols.size, false))

// Ausentes (?) y “Not in universe”
val ausentes = nomCols.map { c =>
  trainExpDF.agg(
    lit(c).as("atributo"),
    sum(when(trim(col(c)) === "Not in universe", 1).otherwise(0)).as("not_in_universe"),
    sum(when(trim(col(c)) === "?", 1).otherwise(0)).as("interrogacion"))
}.reduce(_ union _)
/*
"?" es un valor ausente. "Not in universe" es una categoría válida y no cuenta como ausente
*/
(ausentes
  .filter(col("not_in_universe") > 0 || col("interrogacion") > 0)
  .withColumn("not_in_universe_%", round(col("not_in_universe") * 100.0 / nTrain, 2))
  .withColumn("interrogacion_%", round(col("interrogacion") * 100.0 / nTrain, 2))
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
  .withColumn("atipicos_%", round(col("n_atipicos") * 100.0 / nTrain, 2))
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

val corrInputDF = (trainExpDF
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

// Train y test sin las columnas eliminadas (la misma lista, decidida con train, para ambos)
val trainDF = trainRawDF.drop(dropCols: _*)
val testDF = testRawDF.drop(dropCols: _*)

/*
Cramér's V: asociación de cada nominal con la clase.
Pearson y Spearman no sirven para los atributos nominales. Cramér's V normaliza el
estadístico chi-cuadrado al intervalo [0, 1] y permite comparar atributos con distinto
número de categorías:
  0.00-0.05  sin asociación      0.05-0.10  débil
  0.10-0.20  moderada            > 0.20     fuerte
Se aplica la corrección de Yates cuando los grados de libertad son 1 (tablas 2x2).
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

/*
En el resto de columnas los "?" son pocos (~2-3%), así que no se eliminan las filas:
se sustituyen por la moda (el valor más frecuente de cada columna, sin contar los "?").
Borrar esas filas descartaría instancias que son válidas en todos los demás atributos.
*/
val imputeCols = qCols.diff(dropCols)

/*
Los atípicos no son errores de captura, sino valores reales y poco frecuentes.
No se eliminan las filas porque se concentran en la clase positiva, que ya es
minoritaria (~6%): descartarlas quitaría justamente las instancias más informativas.
En su lugar se capan al límite superior (q3 + 3*IQR), lo que conserva la instancia
y su orden de magnitud, pero neutraliza el efecto de la cola sobre media y varianza.
Los cuantiles se calculan sobre los valores > 0, porque son columnas cero-infladas
y de lo contrario q1 y q3 serían ambos 0.
age no se incluye: no presenta ningún valor atípico.
*/
val capCols = Seq("capital_gains", "capital_losses", "dividends_from_stocks").diff(dropCols)

// Redundancia entre pares de nominales sospechosos (V alto = informacion duplicada)
val paresSospechosos = Seq(
  ("detailed_industry_recode", "major_industry_code"),
  ("detailed_occupation_recode", "major_occupation_code"),
  ("detailed_household_and_family_stat", "detailed_household_summary_in_household"),
  ("detailed_household_and_family_stat", "marital_stat"),
  ("race", "hispanic_origin"),
  ("country_of_birth_self", "country_of_birth_father"),
  ("country_of_birth_father", "country_of_birth_mother"),
  ("country_of_birth_self", "citizenship"),
  ("state_of_previous_residence", "region_of_previous_residence"))

// Selección final de atributos
/*
De los 42 atributos originales se conservan 17 + la clase. Los criterios han sido:
- numéricos: correlación de Pearson/Spearman con la clase
- nominales: Cramér's V con la clase (se conservan los de asociación >= 0.14)
- redundancia: de cada par con V muy alto se conserva un solo atributo

La lista es fija y se decidió con el análisis de train (Pearson, Spearman y Cramér's V
calculados solo sobre train). Se aplica tal cual a test; el Cramér's V se vuelve a mostrar
sobre train tras la limpieza para comprobar que la selección se sostiene.

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

/*
Los 11 atributos nominales se pasan a índices numéricos con StringIndexer.
Los 6 numéricos NO se indexan: ya son double y pasan directos al VectorAssembler.
Indexarlos convertiría cada valor distinto en una categoría (90 edades -> 90
categorías) y destruiría la relación de orden entre ellos.
*/
val attributeColumns = selNomCols.toArray
val outputColumns = attributeColumns.map(_ + "-num")

/*
Todo el preprocesado se ajusta ÚNICAMENTE con train: de él se obtienen las modas de
imputación, los límites de capado y el StringIndexer (además del análisis de Cramér's V).
Test no calcula ningún estadístico propio: recibe exactamente las mismas transformaciones
aprendidas en train, evitando fuga de información y garantizando que un mismo valor
tenga el mismo índice en ambos conjuntos.
*/
var nValoresCapados = 0L

// Aprendido en train: moda de cada columna a imputar (ignorando los propios "?") y
// límite superior de capado de cada columna
def ajustarParametros(trainDF: DataFrame): (Map[String, String], Map[String, Double]) = {
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

  println("%-35s %s".format("Columna", "Moda imputada (train)"))
  println("-" * 60)
  imputeCols.foreach(c => println("%-35s %s".format(c, modas(c))))

  val limites: Map[String, Double] = capCols.map { c =>
    val q = (trainDF
      .select(trim(col(c)).cast("double").as(c))
      .filter(col(c) > 0)
      .stat.approxQuantile(c, Array(0.25, 0.75), 0.001))
    c -> (q(1) + 3 * (q(1) - q(0)))
  }.toMap

  println("\nLímites de capado (winsorización, train):")
  println("%-25s %12s".format("Columna", "Límite sup."))
  println("-" * 38)
  capCols.foreach(c => println("%-25s %12.1f".format(c, limites(c))))

  (modas, limites)
}

// Imputación y capado con los parámetros de train (igual para train y para test)
def limpiar(nombre: String, rawDF: DataFrame, modas: Map[String, String],
    limites: Map[String, Double]): DataFrame = {
  println(s"\n================ Limpieza: $nombre ================")

  val cleanDF = imputeCols.foldLeft(rawDF) { (df, c) =>
    df.withColumn(c, when(trim(col(c)) === "?", modas(c)).otherwise(trim(col(c))))
  }

  println(s"\nColumnas tras eliminar las irrelevantes: ${cleanDF.columns.length - 2} (sin label ni id)")

  println("\nComprobación de columnas imputadas. Valor '?' no presente")
  (cleanDF
    .select(imputeCols.map(c => sum(when(col(c) === "?", 1).otherwise(0)).as(c)): _*)
    .show(false))

  // Las columnas capadas pasan a ser numéricas (double)
  val cappedDF = capCols.foldLeft(cleanDF) { (df, c) =>
    val x = trim(col(c)).cast("double")
    df.withColumn(c, when(x > limites(c), lit(limites(c))).otherwise(x))
  }

  println("\nValores atípicos capados por columna:")
  val capados = (cappedDF
    .select(capCols.map(c => sum(when(col(c) === limites(c), 1).otherwise(0)).as(c)): _*))
  capados.show(false)
  nValoresCapados += capCols.map(c => capados.first().getAs[Long](c)).sum

  cappedDF
}

// Análisis de asociación (Cramér's V): solo sobre train
def analizarCramer(cappedDF: DataFrame): Unit = {
  // Cramér's V con la clase
  val nomColsClean = (cappedDF.columns
    .filterNot(c => numCols.contains(c) || Seq(classCol, "label", "id").contains(c))
    .toSeq)

  val cardinalidad = (cappedDF
    .select(nomColsClean.map(c => countDistinct(trim(col(c))).as(c)): _*)
    .first())

  val ranking = nomColsClean.map { c =>
    (c, cardinalidad.getAs[Long](c), cramersV(cappedDF, c, classCol))
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

  val paresRedundantes = paresSospechosos
    .filter { case (a, b) => nomColsClean.contains(a) && nomColsClean.contains(b) }

  (paresRedundantes
    .map { case (a, b) => (a, b, cramersV(cappedDF, a, b)) }
    .toDF("atributo_1", "atributo_2", "cramers_v")
    .withColumn("cramers_v", round(col("cramers_v"), 3))
    .orderBy(desc("cramers_v"))
    .show(false))
}

// Selección de atributos (misma lista fija para train y test)
def seleccionar(cappedDF: DataFrame): DataFrame = {
  val finalDF = cappedDF.select(
    selNomCols.map(c => trim(col(c)).as(c)) ++
    selNumCols.map(c => trim(col(c)).cast("double").as(c)) :+
    col("label"): _*)

  println(s"Atributos: ${fullCensusDF.columns.length} -> ${finalDF.columns.length} (incluida la clase)")
  finalDF.printSchema()
  finalDF.show(5, false)
  finalDF
}

/*
setStringOrderType("alphabetDesc"): el índice depende solo del valor y no de su
frecuencia. setHandleInvalid("keep"): las categorías no vistas reciben un índice
reservado en lugar de provocar un error.
El indexador se ajusta SOLO con train; el mismo modelo se aplica a test, así un valor
tiene el mismo índice en ambos conjuntos y una categoría que solo aparezca en test
recibe el índice reservado.
*/
def ajustarIndexer(trainFinalDF: DataFrame): StringIndexerModel = {
  val indexerModel = (new StringIndexer()
    .setInputCols(attributeColumns)
    .setOutputCols(outputColumns)
    .setStringOrderType("alphabetDesc")
    .setHandleInvalid("keep")
    .fit(trainFinalDF))

  // Mapa valor original -> índice (posición en labelsArray; el índice = nº de etiquetas es "desconocido")
  val mapaIndices = attributeColumns.zip(outputColumns).zip(indexerModel.labelsArray).flatMap {
    case ((orig, num), labels) =>
      labels.zipWithIndex.map { case (valor, i) => (orig, num, i, valor) }
  }.toSeq
  val mapaDF = mapaIndices.toDF("atributo", "columna_num", "indice", "valor_original")
  mapaDF.orderBy("atributo", "indice").show(mapaIndices.size, false)
  mapaDF.coalesce(1).write.mode("overwrite").option("header", "true").csv(PATH + "census_indices")

  indexerModel
}

def codificar(finalDF: DataFrame, indexerModel: StringIndexerModel): DataFrame = {
  val indexed = indexerModel.transform(finalDF).drop(attributeColumns: _*)

  indexed.show(5, false)

  /*
  Versión para árboles de decisión: los índices entran directamente en features.
  Los árboles de Spark tratan de forma nativa los atributos categóricos indexados y
  particionan por subconjuntos de categorías, lo que es más expresivo que una
  codificación 1 de K y evita multiplicar la dimensionalidad.
  */
  val treesDF = (new VectorAssembler()
    .setInputCols(outputColumns ++ selNumCols.toArray)
    .setOutputCol("features")
    .transform(indexed)
    .select("features", "label"))

  treesDF.show(5, false)

  treesDF
}

// 1) Ajuste con train
println("\n================ Ajuste con train ================")
val (modasTrain, limitesTrain) = ajustarParametros(trainDF)
val trainCappedDF = limpiar("train", trainDF, modasTrain, limitesTrain)
analizarCramer(trainCappedDF)
val trainFinalDF = seleccionar(trainCappedDF)
val indexerTrain = ajustarIndexer(trainFinalDF)
val trainCensusDF = codificar(trainFinalDF, indexerTrain)

// 2) Test: mismas transformaciones, sin recalcular ningún estadístico ni métrica de asociación
val testCappedDF = limpiar("test", testDF, modasTrain, limitesTrain)
val testCensusDF = codificar(seleccionar(testCappedDF), indexerTrain)

// NTAIE: instancias originales que no llegan a los conjuntos finales (train + test)
val totalRows = fullCensusDF.count()
val instanciasFinales = trainCensusDF.count() + testCensusDF.count()
val NTAIE = totalRows - instanciasFinales
println(s"\nInstancias originales: $totalRows | finales (train + test): $instanciasFinales")
println(s"NTAIE (Numero total de ausentes/inconsistentes eliminados) = $NTAIE")
println(s"  (ausentes '?' imputados con la moda, no eliminados; en train -> vacías: $nEmpty, inválidas: $nInvalid, duplicadas conservadas: $nDup)")

// NTOE: los atípicos no se eliminan, se capan; no se pierde ninguna instancia por este motivo
val NTOE = 0L
println(s"NTOE (Numero total de outliers eliminados) = $NTOE")
println(s"  (valores atípicos capados al límite superior, no eliminados: $nValoresCapados)")

/*
Al final de todo el flujo se guardan los conjuntos ya transformados para reutilizarlos
en el entregable individual sin repetir todo el preprocesado.
*/
trainCensusDF.write.mode("overwrite").parquet(PATH + "census_train.parquet")
testCensusDF.write.mode("overwrite").parquet(PATH + "census_test.parquet")

/*
Archivo con el schema de los parquet: columnas reales (features, label) y el nombre
de cada componente del vector features, tomado de los metadatos del VectorAssembler.
*/
import org.apache.spark.ml.attribute.AttributeGroup
import java.nio.file.{Files, Paths}

def describirSchema(archivo: String, df: DataFrame): String = {
  val nombres = (AttributeGroup.fromStructField(df.schema("features")).attributes
    .map(_.map(_.name.getOrElse("?")).toSeq)
    .getOrElse(Seq.empty))
  val columnas = df.schema.fields.map(f => s"  ${f.name}: ${f.dataType.simpleString}").mkString("\n")
  val componentes = nombres.zipWithIndex.map { case (n, i) => f"  $i%3d  $n" }.mkString("\n")
  s"""$archivo
Columnas:
$columnas
Componentes de features (${nombres.size}):
$componentes
"""
}

val schemaTxt = (Seq(
  ("census_train.parquet", trainCensusDF),
  ("census_test.parquet", testCensusDF))
  .map { case (f, df) => describirSchema(f, df) }
  .mkString("\n"))

Files.write(Paths.get(PATH + "census_schema.txt"), schemaTxt.getBytes("UTF-8"))
println(schemaTxt)
