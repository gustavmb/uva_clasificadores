// Run with: spark-shell -i explore.scala   (or paste into spark-shell)
import org.apache.spark.sql.{DataFrame, Column}
import org.apache.spark.sql.functions._

val PATH = "/home/usuario/census/"
val FILE_DATA = "census-income.data"

val columns = Seq(
  "age",
  "class_of_worker",
  "detailed_industry_recode",
  "detailed_occupation_recode",
  "education",
  "wage_per_hour",
  "enroll_in_edu_inst_last_wk",
  "marital_stat",
  "major_industry_code",
  "major_occupation_code",
  "race",
  "hispanic_origin",
  "sex",
  "member_of_a_labor_union",
  "reason_for_unemployment",
  "full_or_part_time_employment_stat",
  "capital_gains",
  "capital_losses",
  "dividends_from_stocks",
  "tax_filer_stat",
  "region_of_previous_residence",
  "state_of_previous_residence",
  "detailed_household_and_family_stat",
  "detailed_household_summary_in_household",
  "instance_weight",
  "migration_code_change_in_msa",
  "migration_code_change_in_reg",
  "migration_code_move_within_reg",
  "live_in_this_house_1_year_ago",
  "migration_prev_res_in_sunbelt",
  "num_persons_worked_for_employer",
  "family_members_under_18",
  "country_of_birth_father",
  "country_of_birth_mother",
  "country_of_birth_self",
  "citizenship",
  "own_business_or_self_employed",
  "fill_inc_questionnaire_for_veterans_admin",
  "veterans_benefits",
  "weeks_worked_in_year",
  "year",
  "income" // class label: '- 50000.' or '50000+.'
)

// ignoreLeadingWhiteSpace removes the space after each comma
// (Spark cannot read http URLs directly; use the local copy of the file)
def load(keepDefaultNa: Boolean): DataFrame = {
  val reader = spark.read
    .option("header", "false")
    .option("inferSchema", "true")
    .option("ignoreLeadingWhiteSpace", "true")
    .option("ignoreTrailingWhiteSpace", "true")
  // pandas keep_default_na=False: empty strings are kept as values, not nulls
  val configured =
    if (keepDefaultNa) reader.option("nullValue", "")
    else reader.option("nullValue", "\u0000__NO_NULLS__").option("emptyValue", "")
  configured.csv(PATH + FILE_DATA).toDF(columns: _*)
}

var df = load(keepDefaultNa = true)

df.printSchema()

// Cramér's V between two columns (nulls in x are treated as the category "NULO")
def cramersV(data: DataFrame, x: String, y: String): Double = {
  val cells = data
    .select(coalesce(col(x).cast("string"), lit("NULO")).as("x"), col(y).cast("string").as("y"))
    .where(col("y").isNotNull)
    .groupBy("x", "y").count()
    .collect()
    .map(r => ((r.getString(0), r.getString(1)), r.getLong(2).toDouble))
    .toMap

  val rowTotals = cells.groupBy(_._1._1).map { case (k, v) => k -> v.values.sum }
  val colTotals = cells.groupBy(_._1._2).map { case (k, v) => k -> v.values.sum }
  val n = cells.values.sum
  val dof = (rowTotals.size - 1) * (colTotals.size - 1)

  // chi-square over every cell of the contingency table (missing cells have observed = 0)
  var chi2 = 0.0
  for ((rk, rt) <- rowTotals; (ck, ct) <- colTotals) {
    val expected = rt * ct / n
    var diff = cells.getOrElse((rk, ck), 0.0) - expected
    // Yates' continuity correction for 2x2 tables, as scipy does
    if (dof == 1) diff = math.signum(diff) * math.max(0.0, math.abs(diff) - 0.5)
    chi2 += diff * diff / expected
  }

  val k = math.min(rowTotals.size, colTotals.size) - 1
  math.sqrt(chi2 / (n * k))
}

// --- Target class distribution ---
val totalRows = df.count()
val positiveCount = df.filter(col("income") === "50000+.").count()
val negativeCount = totalRows - positiveCount

val positivePct = positiveCount.toDouble / totalRows * 100
val negativePct = negativeCount.toDouble / totalRows * 100
val razon = negativeCount.toDouble / positiveCount

println("Distribución de la clase objetivo (income)")
println("-" * 40)
println(s"Total de registros:        $totalRows")
println(f"Ingresos altos (50000+):   $positiveCount ($positivePct%.2f%%)")
println(f"Ingresos bajos (- 50000):  $negativeCount ($negativePct%.2f%%)")

// --- Reload keeping the default NA strings as plain values ---
df = load(keepDefaultNa = false)
df.groupBy("hispanic_origin").count().orderBy(desc("count")).show(50, truncate = false)

val classCol = "income"
val contCols = Seq("age", "wage_per_hour", "capital_gains",
  "capital_losses", "dividends_from_stocks", "instance_weight")
val discCols = Seq("num_persons_worked_for_employer", "weeks_worked_in_year")
val numCols = contCols ++ discCols
// these are codes, not numbers
val codeCols = Seq("detailed_industry_recode", "detailed_occupation_recode",
  "own_business_or_self_employed", "veterans_benefits", "year")
// 33 columns (includes codeCols)
val nomCols = columns.filterNot(c => (numCols :+ classCol).contains(c))

// copy for exploration; df stays intact
val dfExp = codeCols.foldLeft(df)((d, c) => d.withColumn(c, col(c).cast("string")))

// --- Data quality checks ---
val emptyRows = df.filter(df.columns.map(c => col(c).isNull).reduce(_ && _)).count()
val invalidClass = df.filter(!col("income").isin("- 50000.", "50000+.")).count()
// pandas duplicated(): rows that repeat an earlier identical row
val duplicates = totalRows - df.distinct().count()

println(s"filas vacías: $emptyRows")
println(s"clase inválida: $invalidClass")
println(s"duplicadas: $duplicates")
