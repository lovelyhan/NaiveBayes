package test.naivebayes

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.ArrayBuffer

object NativeBayesModel {
  private val logger: Logger = LoggerFactory.getLogger(NativeBayesModel.getClass)

  object Model {
    //Gaussian Bayes model
    var meanValues: Vector = _
    var variance: Vector = _
  }
}

import test.naivebayes.NativeBayesModel.Model._
class NativeBayesModel() extends Serializable{
  @transient protected val logger: Logger = NativeBayesModel.logger
  final val dimensions = 999990+1
  val epoch = 1
  val batchNum = 20

    def loadData(sc: SparkContext,path: String):RDD[LabeledPoint] = {
      val startTime = System.currentTimeMillis()
      val dataRdd = MLUtils.loadLibSVMFile(sc,path)
        .coalesce(sc.defaultMinPartitions)
        .persist(StorageLevel.MEMORY_AND_DISK)
      val totalLength = dataRdd.count()
      logger.info(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
      dataRdd
    }

    protected def initParams(): Unit ={
      val means = new Array[Double](dimensions)
      meanValues = Vectors.dense(means)
      val variances = new Array[Double](dimensions)
      variance = Vectors.dense(variances)
    }

  def calculateMeansAndVariance(miniBatch: Iterator[LabeledPoint],totalMeans: Vector,totalVariances: Vector): Iterator[Tuple2[Array[Double],Array[Double]]] ={
    val ans = new ArrayBuffer[Tuple2[Array[Double],Array[Double]]]
    val totalMeans = new Array[Double](dimensions)
    val totalVar = new Array[Double](dimensions)
    val data = miniBatch.toList
    val dataLength = data.length

    for(m <- 0 until data.length) {
      val labeledPoint = data.apply(m)
      val features = labeledPoint.features.asInstanceOf[SparseVector]
      val keys = features.indices
      val values = features.values
      //Sum up all to calculate mean value
      for (i <- 0 until keys.length) {
        val num = keys.apply(i)
        totalMeans(num) += values.apply(i)
      }
    }
    for(m <- 0 until dimensions){
      totalMeans(m) /= dataLength
    }
    for(n <- 0 until data.length){
      val labeledPoint = data.apply(n)
      val features = labeledPoint.features.asInstanceOf[SparseVector]
      val keys = features.indices
      val values = features.values
      for(k <- 0 until keys.length){
        val num = keys.apply(k)
        val diff = values.apply(k) - totalMeans(num)
        totalVar(num) += diff * diff
      }
    }
    ans.+=:(totalMeans,totalVar)
   ans.iterator
  }

  def axpy(a: Double, x: Array[Double], y: Array[Double]): Unit = {
    for(i <- 0 to y.length){
      y(i) += a * x(i)
    }
  }

  def run(path: String, sc: SparkContext): Unit = {
    val partitionNum = sc.defaultMinPartitions
    initParams()
    val rdd = loadData(sc, path)
    val rddTotalNum = rdd.count()
    for (epochTime <- 0 until epoch) {
      logger.info(s"Epoch[$epoch] start training")
      for (batchNum <- 0 until batchNum) {
        logger.info(s"Iteration[$batchNum] starts")
        val startBatchTime = System.currentTimeMillis()
        val oneIterationRDD = rdd.sample(false, 0.05, 0L)
        //广播全局参数
        val broadcastMeans = sc.broadcast(meanValues)
        val broadcastVariance = sc.broadcast(variance)
        val total = oneIterationRDD.mapPartitions(miniList => calculateMeansAndVariance(miniList, broadcastMeans.value, broadcastVariance.value))
        logger.debug(s"Mean value and variance are calculated,took${System.currentTimeMillis() - startBatchTime} ms")
        val sumAll = total.treeReduce((x, y) => {
          axpy(1.0,x._1,y._1)
          axpy(1.0,x._2,y._2)
          y
        })
        val partitionMean = sumAll._1
        for(j <- 0 to partitionMean.length){
          partitionMean(j) = partitionMean(j)/partitionNum
        }
        val partitionVar = sumAll._2
        for(k <- 0 to partitionVar.length){
          partitionVar(k) = partitionVar(k) / partitionNum
        }
        meanValues = Vectors.dense(partitionMean)
        variance = Vectors.dense(partitionVar)
        logger.debug(s"This term's calculation finished,totally took ${System.currentTimeMillis() - startBatchTime} ms")
      }
    }
  }
}
