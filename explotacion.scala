import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import org.apache.spark.ml.feature.VectorAssembler
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
rawCensusDF
  .groupBy(trim(col(classCol)).as(classCol))
  .count()
  .withColumn("label", when(col(classCol) === "50000+.", "positive").otherwise("negative"))
  .withColumn("percentage", round(col("count") * 100.0 / totalRows, 2))
  .orderBy(desc("count"))
  .show()

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

Seq(("continuos", contCols.size), ("discretos", discCols.size), ("booleanos", 0), ("nominales", nNominales))
  .toDF("tipo", "count")
  .show()

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

Seq(("vacías", nEmpty), ("inválidas", nInvalid), ("duplicadas", nDup))
  .toDF("instancias", "count")
  .withColumn("percentage", round(col("count") * 100.0 / totalRows, 2))
  .show()

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

statsDF
  .select(col("atributo"), col("count"), round(col("mean"), 2).as("mean"), round(col("std"), 2).as("std"),
    col("min"), col("q")(0).as("25%"), col("q")(1).as("50%"), col("q")(2).as("75%"), col("max"),
    round(col("skew"), 2).as("skew"), round(col("%zeros"), 2).as("%zeros"), col("n_unicos"))
  .show(false)

// Estadísticas de los atributos nominales (valores posibles, moda, frecuencia)
val nomCols = rawCensusDF.columns.filterNot(c => numCols.contains(c) || c == classCol).toSeq

val nomStats = nomCols.map { c =>
  rawCensusDF
    .groupBy(trim(col(c)).as("valor"))
    .count()
    .agg(lit(c).as("atributo"), count("*").as("valores_posibles"), max(struct(col("count"), col("valor"))).as("m"))
}.reduce(_ union _)

nomStats
  .select(col("atributo"), col("valores_posibles"), col("m.valor").as("moda"), col("m.count").as("frecuencia"))
  .show(nomCols.size, false)

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
ausentes
  .filter(col("not_in_universe") > 0 || col("interrogacion") > 0)
  .withColumn("not_in_universe_%", round(col("not_in_universe") * 100.0 / totalRows, 2))
  .withColumn("interrogacion_%", round(col("interrogacion") * 100.0 / totalRows, 2))
  .select("atributo", "not_in_universe", "not_in_universe_%", "interrogacion", "interrogacion_%")
  .orderBy(desc("interrogacion"))
  .show(nomCols.size, false)

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

outliers
  .withColumn("atipicos_%", round(col("n_atipicos") * 100.0 / totalRows, 2))
  .show(false)


// Matriz de correlación (Pearson y Spearman) entre los atributos numéricos y la clase
/*
- Pearson  = relación lineal entre dos atributos
- Spearman = relación monótona (sobre los rangos), más robusta a atípicos y a las
             distribuciones muy inclinadas (skew) de este conjunto de datos
La clase se añade como "income_bin" (1 = "50000+.", 0 = "- 50000.") para ver la
correlación de cada atributo numérico con la variable objetivo.
*/
val corrCols = numCols :+ "income_bin"

val corrInputDF = rawCensusDF
  .select(numCols.map(c => trim(col(c)).cast("double").as(c)) :+
    when(trim(col(classCol)) === "50000+.", 1.0).otherwise(0.0).as("income_bin"): _*)
  .na.drop(corrCols)

val corrVecDF = new VectorAssembler()
  .setInputCols(corrCols.toArray)
  .setOutputCol("features")
  .transform(corrInputDF)
  .select("features")

// La matriz (atributo x atributo) como DataFrame, redondeada a 2 decimales
def corrMatrixDF(m: Matrix, cols: Seq[String]): DataFrame =
  (for (i <- cols.indices; j <- cols.indices) yield (i, cols(i), cols(j), m(i, j)))
    .toDF("orden", "atributo", "columna", "valor")
    .groupBy("orden", "atributo")
    .pivot("columna", cols)
    .agg(round(first("valor"), 2))
    .orderBy("orden")
    .drop("orden")

val pearsonM = Correlation.corr(corrVecDF, "features", "pearson").head.getAs[Matrix](0)
val spearmanM = Correlation.corr(corrVecDF, "features", "spearman").head.getAs[Matrix](0)

println("Matriz de correlación de Pearson")
corrMatrixDF(pearsonM, corrCols)
  .show(false)

println("Matriz de correlación de Spearman")
corrMatrixDF(spearmanM, corrCols)
  .show(false)

// Pares de atributos ordenados por correlación absoluta (sin la diagonal ni repetidos)
val paresCorr = for {
  i <- corrCols.indices
  j <- (i + 1) until corrCols.size
} yield (corrCols(i), corrCols(j), pearsonM(i, j), spearmanM(i, j))

paresCorr
  .toDF("atributo_1", "atributo_2", "pearson", "spearman")
  .withColumn("pearson", round(col("pearson"), 3))
  .withColumn("spearman", round(col("spearman"), 3))
  .orderBy(abs(col("spearman")).desc)
  .show(paresCorr.size, false)
