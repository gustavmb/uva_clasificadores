import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType}

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

