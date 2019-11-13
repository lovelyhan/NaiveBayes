package test.naivebayes

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.{Vector => OldVector, Vectors => OldVectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.slf4j.{Logger, LoggerFactory}

object KMeansModel {
  private val logger: Logger = LoggerFactory.getLogger(KMeansModel.getClass)

}

class KMeansModel(val clusterCenters: Array[Vector]) {
  @transient protected val logger: Logger = KMeansModel.logger
  private val clusterCentersWithNorm =
    if (clusterCenters == null) null else clusterCenters.map(new VectorWithNorm(_))
  final val dimensions = 999990
  //set iteration's time
  val iterations = 5
  //set K
  val k = 3
  //set random
  val randomItem = 12345
  //set epsilon
  val epsilon = 1e-4

  //load data
  def loadData(sc: SparkContext,path: String): Dataset[_]= {
    val conf = new SparkConf()
      .setAppName("KMeans Test")
//      .setMaster("local")
//      .set("spark.executor.memory","1g")
    val spark = SparkSession.builder().config(conf).getOrCreate()

    // Import Kmeans clustering Algorithm
    // Load the Wholesale Customers Data
    val startTime = System.currentTimeMillis()
    val data = spark
      .read
      .option("header","true")
      .option("inferschema","true")
      .format("csv")
      .load("Wholesale customers data.csv")

    val totalLength = data.count()
    val feature_data = (data.select("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"))
    val assembler = new VectorAssembler().setInputCols(Array("Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen")).setOutputCol("features")
    val training_data = assembler.transform(feature_data).select("features")
    logger.info(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
    println(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
    training_data
  }

  def initRandom(data: RDD[VectorWithNorm]): Array[VectorWithNorm] = {
    // Select without replacement; may still produce duplicates if the data has < k distinct
    // points, so deduplicate the centroids to match the behavior of k-means|| in the same situation
    data.takeSample(false, k, new XORShiftRandom(this.randomItem).nextInt())
      .map(_.vector).distinct.map(new VectorWithNorm(_))
  }


  def run(
         data: Dataset[_]
         ): KMeansModel ={
    //Change data form from dataframe[_] to rdd[vector]
    val instances: RDD[OldVector] = data.select("features").rdd.map {
      case Row(point: Vector) => OldVectors.fromML(point)
    }
    if (instances.getStorageLevel == StorageLevel.NONE){
      logger.info(s"The input data is not directly cached,which may hurt performance if its" + "parent RDDs are also uncached.")
    }
    //compute squared norms and cache them
    val norms = instances.map(OldVectors.norm(_, 2.0))
    norms.persist()
    val zippedData = instances.zip(norms).map { case (v, norm) =>
      new VectorWithNorm(v, norm)
    }





  }

}

//Used to calculate distance
class VectorWithNorm(val vector: Vector, val norm: Double) extends Serializable {
  def this(vector: Vector,norm:Double) = this(vector,norm)

  def this(vector: Vector) = this(vector, Vectors.norm(vector, 2.0))

  def this(array: Array[Double]) = this(Vectors.dense(array))

  /** Converts the vector to a dense vector. */
  def toDense: VectorWithNorm = new VectorWithNorm(Vectors.dense(vector.toArray), norm)
}


