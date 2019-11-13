package test.naivebayes
import org.apache.spark.sql.{SQLContext, SparkSession}
import org.apache.log4j._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

object KMeansSpark {
  Logger.getLogger("org").setLevel(Level.ERROR)
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
      .setAppName("Bayes Test")
      .setMaster("local")
      .set("spark.executor.memory","1g")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    // Import Kmeans clustering Algorithm
    import org.apache.spark.ml.clustering.KMeans
    //    val sc = SparkContext.getOrCreate(conf)
    //    val sqlContext = SQLContext.getOrCreate(sc)
    //    val dataRDD = sqlContext
    //      .read
    //      .format("com.databricks.spark.csv")
    //      .option("header","true")
    //      .load("Wholesale customers data.csv")
    //    dataRDD.show(5)

    // Load the Wholesale Customers Data
    val data = spark.read.option("header","true").option("inferschema","true").format("csv").load("Wholesale customers data.csv")
    data.printSchema()
//
    // Select the following columns for the training set:
    // Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
    // Cal this new subset feature_data
    val feature_data = (data.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"))


    // Create a new VectorAssembler object called assembler for the feature
    // columns as the input Set the output column to be called features
    // Remember there is no Label column
    val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")

    // Use the assembler object to transform the feature_data
    // Call this new data training_data
    val training_data = assembler.transform(feature_data).select("features")
    //show data created
    training_data.show(5)
    // Create a Kmeans Model with K=3
    val kmeans = new KMeans().setK(3).setSeed(12345)
    //
    //  // Fit that model to the training_data
    val model = kmeans.fit(training_data)
    //
    //  // Evaluate clustering by computing Within Set Sum of Squared Errors.
    val WSSSE = model.computeCost(training_data)
    //
    // Shows the result.
    println(s"Withing set sum of squared errors = $WSSSE")
    println("Cluster Centers: ")
    model.clusterCenters.foreach(println)
  }
}
