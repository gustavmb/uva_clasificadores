// Run with: spark-shell -i final_scala_script.scala   (or paste into spark-shell)
// Scala/Spark translation of final_script.py.
// Spark cannot read http URLs directly, so it uses the local copy of census-income.data.
// Plots are replaced by the tables/matrices that back them (printed, or written to CSV).
import org.apache.spark.sql.{DataFrame, Row}
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

// Lista de atributos
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

// ---------------------------------------------------------------------------
// Loading (equivalent of pd.read_csv with skipinitialspace=True)
// ---------------------------------------------------------------------------

// Strings pandas treats as NA by default
val naStrings = Seq("", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
  "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None", "n/a", "nan", "null")

def load(keepDefaultNa: Boolean): DataFrame = {
  val raw = spark.read
    .option("header", "false")
    .option("inferSchema", "false")
    .option("nullValue", "\u0000__NO_NULLS__")
    .option("ignoreLeadingWhiteSpace", "true")
    .option("ignoreTrailingWhiteSpace", "true")
    .csv(PATH + FILE_DATA)
    .toDF(columns: _*)

  val cleaned = columns.foldLeft(raw) { (d, c) =>
    val v = trim(col(c))
    if (keepDefaultNa) d.withColumn(c, when(v.isNull || v.isin(naStrings: _*), lit(null).cast("string")).otherwise(v))
    else d.withColumn(c, coalesce(v, lit("")))
  }
  numCols.foldLeft(cleaned)((d, c) => d.withColumn(c, col(c).cast("double")))
}

var df = load(keepDefaultNa = true)
df.cache()

// df.info()
println(s"<DataFrame> ${df.count()} entries, ${columns.size} columns")
df.printSchema()
val nonNull = df.select(columns.map(c => count(col(c)).as(c)): _*).first()
columns.zipWithIndex.foreach { case (c, i) => println(f"$i%3d $c%-45s ${nonNull.getLong(i)}%8d non-null") }

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

// Cramér's V between two categorical columns (nulls in x are the category "NULO").
// Chi-square over the full contingency table, with Yates' correction when dof == 1 (as scipy).
def cramersVCells(cells: Map[(String, String), Double]): Double = {
  val rowTotals = cells.groupBy(_._1._1).map { case (k, v) => k -> v.values.sum }
  val colTotals = cells.groupBy(_._1._2).map { case (k, v) => k -> v.values.sum }
  val n = cells.values.sum
  val dof = (rowTotals.size - 1) * (colTotals.size - 1)
  var chi2 = 0.0
  for ((rk, rt) <- rowTotals; (ck, ct) <- colTotals) {
    val expected = rt * ct / n
    var diff = cells.getOrElse((rk, ck), 0.0) - expected
    if (dof == 1) diff = math.signum(diff) * math.max(0.0, math.abs(diff) - 0.5)
    chi2 += diff * diff / expected
  }
  val k = math.min(rowTotals.size, colTotals.size) - 1
  math.sqrt(chi2 / (n * k))
}

def cramersV(xs: Array[String], ys: Array[String]): Double = {
  val cells = scala.collection.mutable.HashMap.empty[(String, String), Double].withDefaultValue(0.0)
  var i = 0
  while (i < xs.length) {
    cells((if (xs(i) == null) "NULO" else xs(i), ys(i))) += 1.0
    i += 1
  }
  cramersVCells(cells.toMap)
}

// Same statistic, but the contingency table is aggregated inside Spark (only the small table is collected)
def cramersVSpark(data: DataFrame, x: String, y: String): Double = {
  val cells = data
    .select(coalesce(col(x), lit("NULO")).as("x"), col(y).as("y"))
    .groupBy("x", "y").count()
    .collect()
    .map(r => ((r.getString(0), r.getString(1)), r.getLong(2).toDouble))
    .toMap
  cramersVCells(cells)
}

// Value counts of a column computed in Spark (nulls excluded), sorted by frequency desc, then value
def valueCountsSpark(data: DataFrame, c: String): Seq[(String, Int)] =
  data.where(col(c).isNotNull).groupBy(c).count().collect()
    .map(r => (r.getString(0), r.getLong(1).toInt)).toSeq.sortBy { case (k, n) => (-n, k) }

// Quantile with linear interpolation (pandas default)
def quantile(sorted: Array[Double], q: Double): Double = {
  val pos = (sorted.length - 1) * q
  val lo = math.floor(pos).toInt
  val hi = math.ceil(pos).toInt
  sorted(lo) + (sorted(hi) - sorted(lo)) * (pos - lo)
}

def mean(a: Array[Double]): Double = a.sum / a.length

def std(a: Array[Double]): Double = {
  val m = mean(a)
  math.sqrt(a.map(v => (v - m) * (v - m)).sum / (a.length - 1))
}

// Unbiased sample skewness (pandas .skew())
def skew(a: Array[Double]): Double = {
  val n = a.length.toDouble
  val m = mean(a)
  val m2 = a.map(v => math.pow(v - m, 2)).sum / n
  val m3 = a.map(v => math.pow(v - m, 3)).sum / n
  math.sqrt(n * (n - 1)) / (n - 2) * m3 / math.pow(m2, 1.5)
}

def pearson(a: Array[Double], b: Array[Double]): Double = {
  val ma = mean(a); val mb = mean(b)
  val cov = a.zip(b).map { case (x, y) => (x - ma) * (y - mb) }.sum
  val va = a.map(x => (x - ma) * (x - ma)).sum
  val vb = b.map(y => (y - mb) * (y - mb)).sum
  cov / math.sqrt(va * vb)
}

// Average ranks for ties
def rank(a: Array[Double]): Array[Double] = {
  val idx = a.indices.sortBy(a(_))
  val ranks = new Array[Double](a.length)
  var i = 0
  while (i < idx.length) {
    var j = i
    while (j + 1 < idx.length && a(idx(j + 1)) == a(idx(i))) j += 1
    val r = (i + j) / 2.0 + 1
    (i to j).foreach(k => ranks(idx(k)) = r)
    i = j + 1
  }
  ranks
}

def spearman(a: Array[Double], b: Array[Double]): Double = pearson(rank(a), rank(b))

// Value counts sorted by frequency (descending), ties by value
def valueCounts(a: Array[String]): Seq[(String, Int)] =
  a.groupBy(identity).map { case (k, v) => (k, v.length) }.toSeq.sortBy { case (k, n) => (-n, k) }

def printMatrix(title: String, names: Seq[String], m: Array[Array[Double]]): Unit = {
  println(title)
  println(" " * 36 + names.map(n => f"${n.take(10)}%10s").mkString(" "))
  names.indices.foreach { i =>
    println(f"${names(i)}%-35s " + m(i).map(v => f"$v%10.2f").mkString(" "))
  }
}

// ---------------------------------------------------------------------------
// Distribución de la clase objetivo
// ---------------------------------------------------------------------------

val totalRows = df.count()
val positiveCount = df.filter(col(classCol) === "50000+.").count()
val negativeCount = totalRows - positiveCount

val positivePct = positiveCount.toDouble / totalRows * 100
val negativePct = negativeCount.toDouble / totalRows * 100
val razon = negativeCount.toDouble / positiveCount

println("Distribución de la clase objetivo (income)")
println("-" * 40)
println(s"Total de registros:        $totalRows")
println(f"Ingresos altos (50000+):   $positiveCount ($positivePct%.2f%%)")
println(f"Ingresos bajos (- 50000):  $negativeCount ($negativePct%.2f%%)")

// ---------------------------------------------------------------------------
// Posibles valores en hispanic_origin (reload with keep_default_na=False)
// ---------------------------------------------------------------------------

df.unpersist()
df = load(keepDefaultNa = false)
df.cache()
df.groupBy("hispanic_origin").count().orderBy(desc("count"), col("hispanic_origin")).show(50, truncate = false)

// Replacement for the missing-values heatmap: "?" counts per column on a 2000-row sample
println("Valores '?' por columna (muestra de 2000 filas)")
val sample2000 = df.orderBy(rand(42)).limit(2000).cache()
sample2000.select(columns.map(c => sum(when(col(c).cast("string") === "?", 1).otherwise(0)).as(c)): _*)
  .first()
  .toSeq.zip(columns)
  .foreach { case (n, c) => println(f"$c%-45s $n") }
sample2000.unpersist()

// df_exp: copy for exploration (code columns as strings); df stays intact
val dfExp = codeCols.foldLeft(df)((d, c) => d.withColumn(c, col(c).cast("string")))

// ---------------------------------------------------------------------------
// Instancias vacías, inválidas y duplicadas
// ---------------------------------------------------------------------------

val emptyRows = df.filter(columns.map(c => col(c).isNull).reduce(_ && _)).count()
val invalidClass = df.filter(!col(classCol).isin("- 50000.", "50000+.")).count()
val duplicates = totalRows - df.distinct().count() // like pandas duplicated(): repeats of an earlier row

println(s"filas vacías: $emptyRows")
println(s"clase inválida: $invalidClass")
println(s"duplicadas: $duplicates")

// ---------------------------------------------------------------------------
// Numéricos (min, max, media, desviación, distribución, valores atípicos)
// ---------------------------------------------------------------------------

val numRows = df.select(numCols.map(col) :+ col(classCol): _*).collect()
val incomeBinNum: Array[Double] = numRows.map(r => if (r.getString(numCols.size) == "50000+.") 1.0 else 0.0)
val numData: Map[String, Array[Double]] = numCols.zipWithIndex.map { case (c, i) =>
  c -> numRows.map(r => if (r.isNullAt(i)) Double.NaN else r.getDouble(i))
}.toMap
val numClean: Map[String, Array[Double]] = numData.map { case (c, a) => c -> a.filterNot(_.isNaN) }

println(f"${"atributo"}%-34s ${"count"}%8s ${"mean"}%12s ${"std"}%12s ${"min"}%10s ${"25%"}%10s ${"50%"}%10s ${"75%"}%10s ${"max"}%12s ${"skew"}%8s ${"%zeros"}%8s ${"n_unicos"}%9s")
numCols.foreach { c =>
  val a = numClean(c)
  val s = a.sorted
  val zeros = a.count(_ == 0.0).toDouble / a.length * 100
  println(f"$c%-34s ${a.length}%8d ${mean(a)}%12.4f ${std(a)}%12.4f ${s.head}%10.2f ${quantile(s, .25)}%10.2f ${quantile(s, .5)}%10.2f ${quantile(s, .75)}%10.2f ${s.last}%12.2f ${skew(a)}%8.2f $zeros%8.2f ${a.distinct.length}%9d")
}

val rangos = Seq(
  "age" -> (0.0, 90.0), "wage_per_hour" -> (0.0, 9999.0), "capital_gains" -> (0.0, 99999.0),
  "capital_losses" -> (0.0, 99999.0), "dividends_from_stocks" -> (0.0, 99999.0),
  "instance_weight" -> (0.0, Double.PositiveInfinity),
  "num_persons_worked_for_employer" -> (0.0, 6.0), "weeks_worked_in_year" -> (0.0, 52.0))
rangos.foreach { case (c, (lo, hi)) =>
  println(s"$c fuera de rango: ${numClean(c).count(v => v < lo || v > hi)}")
}

// Replacement for the log-scale histograms: 40 equal-width bins per attribute
numCols.foreach { c =>
  val a = numClean(c)
  val lo = a.min; val hi = a.max
  val width = (hi - lo) / 40
  val counts = new Array[Int](40)
  a.foreach { v => counts(if (width == 0) 0 else math.min(39, ((v - lo) / width).toInt)) += 1 }
  println(s"Histograma $c (40 bins, min=$lo, max=$hi)")
  counts.zipWithIndex.foreach { case (n, i) =>
    println(f"  [${lo + i * width}%14.2f, ${lo + (i + 1) * width}%14.2f) $n%8d")
  }
}

// ---------------------------------------------------------------------------
// Cómputo de nominales
// ---------------------------------------------------------------------------

// Cached so the many per-column aggregations don't re-parse the CSV
val dfNom = dfExp.select(nomCols.map(col): _*).cache()

val nomCounts: Map[String, Seq[(String, Int)]] = nomCols.map(c => c -> valueCountsSpark(dfNom, c)).toMap

println(f"${"atributo"}%-45s ${"n_categorias"}%12s ${"moda"}%-45s ${"freq_moda_%"}%12s")
nomCols
  .map { c =>
    val vc = nomCounts(c)
    (c, vc.size, vc.head._1, vc.head._2.toDouble / vc.map(_._2).sum * 100)
  }
  .sortBy { case (_, n, _, _) => -n }
  .foreach { case (c, n, mode, pct) => println(f"$c%-45s $n%12d ${mode.take(45)}%-45s $pct%12.2f") }

// Replacement for the bar charts
Seq("education", "major_occupation_code", "marital_stat").foreach { c =>
  println(s"Frecuencias de $c")
  nomCounts(c).foreach { case (k, n) => println(f"  $k%-50s $n%8d") }
}

// ---------------------------------------------------------------------------
// Ausentes (?). "Not in universe" es una categoría válida y no cuenta como ausente
// ---------------------------------------------------------------------------

val miss = nomCols.map { c =>
  val vc = nomCounts(c)
  val q = vc.filter(_._1 == "?").map(_._2).sum
  (c, q, vc.filter(_._1.startsWith("Not in universe")).map(_._2).sum, math.round(q.toDouble / totalRows * 10000) / 100.0)
}
println(f"${"atributo"}%-45s ${"interrogacion"}%14s ${"not_in_universe"}%16s ${"interrogacion_%"}%16s")
miss.filter(_._2 > 0).sortBy(-_._4).foreach { case (c, q, niu, pct) =>
  println(f"$c%-45s $q%14d $niu%16d $pct%16.2f")
}

// ---------------------------------------------------------------------------
// Valores atípicos
// ---------------------------------------------------------------------------

println(f"${"atributo"}%-25s ${"n>0"}%8s ${"%>0"}%8s ${"max"}%12s ${"limite_extremo"}%16s ${"n_extremos"}%11s")
Seq("age", "wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks").foreach { c =>
  // only values > 0, because the global IQR is 0
  val nz = numClean(c).filter(_ > 0).sorted
  val q1 = quantile(nz, .25)
  val q3 = quantile(nz, .75)
  val lim = q3 + 3 * (q3 - q1)
  val pct = math.round(nz.length.toDouble / totalRows * 10000) / 100.0
  println(f"$c%-25s ${nz.length}%8d $pct%8.2f ${numClean(c).max}%12.2f $lim%16.2f ${nz.count(_ > lim)}%11d")
}

println(df.filter(col("capital_gains") === 99999).count())
df.filter(col("capital_gains") === 99999).groupBy(classCol).count().orderBy(desc("count")).show(truncate = false)

// ---------------------------------------------------------------------------
// Correlaciones (Pearson y Spearman)
// ---------------------------------------------------------------------------

// Pairwise-complete observations
def pairs(a: String, b: String): (Array[Double], Array[Double]) = {
  val keep = numData(a).indices.filter(i => !numData(a)(i).isNaN && !numData(b)(i).isNaN)
  (keep.map(numData(a)(_)).toArray, keep.map(numData(b)(_)).toArray)
}

Seq("pearson" -> pearson _, "spearman" -> spearman _).foreach { case (name, f) =>
  val m = numCols.map(a => numCols.map { b => val (x, y) = pairs(a, b); f(x, y) }.toArray).toArray
  printMatrix(name, numCols, m)
}

// ---------------------------------------------------------------------------
// Nominal por nominal (Cramér's V sobre una muestra de 50000 filas)
// ---------------------------------------------------------------------------

val muestraRows = dfExp.select(nomCols.map(col): _*).orderBy(rand(42)).limit(50000).collect()
val muestra: Map[String, Array[String]] = nomCols.zipWithIndex.map { case (c, i) =>
  c -> muestraRows.map(r => r.getString(i))
}.toMap

val nNom = nomCols.size
val V = Array.fill(nNom, nNom)(1.0)
for (i <- 0 until nNom; j <- i + 1 until nNom) {
  val v = cramersV(muestra(nomCols(i)), muestra(nomCols(j)))
  V(i)(j) = v
  V(j)(i) = v
}

// Replacement for the heatmap: matrix written to CSV
val matrixFile = new java.io.PrintWriter(PATH + "cramers_v_matrix.csv")
matrixFile.println(("" +: nomCols).mkString(","))
nomCols.indices.foreach(i => matrixFile.println((nomCols(i) +: V(i).map(_.toString).toSeq).mkString(",")))
matrixFile.close()
println(s"Matriz de Cramér's V escrita en ${PATH}cramers_v_matrix.csv")

// Candidatos a redundantes
val pares = (for (i <- 0 until nNom; j <- i + 1 until nNom) yield (nomCols(i), nomCols(j), V(i)(j)))
  .sortBy(-_._3)
pares.filter(_._3 >= 0.9).foreach { case (a, b, v) => println(f"$a%-45s $b%-45s $v%.6f") }

// ---------------------------------------------------------------------------
// Atributo x clase (relevancia)
// ---------------------------------------------------------------------------

val dfBin = dfExp.withColumn("income_bin", when(col(classCol) === "50000+.", "1").otherwise("0")).cache()

val vClase = nomCols.map(c => c -> cramersVSpark(dfBin, c, "income_bin")).sortBy(-_._2)
println("Cramér's V (nominal vs income)")
vClase.foreach { case (c, v) => println(f"$c%-45s $v%.6f") }

println("|Correlación| (numérico vs income)")
numCols
  .map { c =>
    val keep = numData(c).indices.filter(i => !numData(c)(i).isNaN)
    c -> math.abs(pearson(keep.map(numData(c)(_)).toArray, keep.map(incomeBinNum(_)).toArray))
  }
  .sortBy(-_._2)
  .foreach { case (c, v) => println(f"$c%-45s $v%.6f") }
