package test.naivebayes

import java.util

import org.apache.spark.api.java.function.FlatMapFunction
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{SparkConf, SparkContext, SparkEnv}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

object NativeBayesModel {
  private val logger: Logger = LoggerFactory.getLogger(NativeBayesModel.getClass)

  object Model {
    //Gaussian Bayes model
    var meanValues: Vector = _
    var variance: Vector = _
  }
}

import NativeBayesModel.Model._
abstract class NativeBayesModel() extends Serializable{
  @transient protected val logger: Logger = NativeBayesModel.logger
  final val dimensions = 999990
  val epoch = 1
  val batchNum = 20

    def getSparkContext(): SparkContext = {
      val conf = new SparkConf()
        .setAppName("Bayes Test")
        .setMaster("local")
        .set("spark.executor.memory","1g");
      val sc = SparkContext(conf);
      sc
    }

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

  def calculateMeansAndVariance(miniBatch: Iterator[LabeledPoint],totalMeans: Vector,totalVariances: Vector): Iterator[Tuple2[Vector,Vector]] ={
    val ans = new ArrayBuffer[Tuple2[Vector,Vector]]
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
        totalMeans.update(num, totalMeans.apply(num) + values.apply(num))
      }
    }
    for(m <- 0 until dimensions){
      totalMeans.update(m, totalMeans.apply(m)/dataLength)
    }
    for(n <- 0 until data.length){
      val labeledPoint = data.apply(n)
      val features = labeledPoint.features.asInstanceOf[SparseVector]
      val keys = features.indices
      val values = features.values
      for(k <- 0 until keys.length){
        val num = keys.apply(k)
        val diff = values.apply(num) - totalMeans.apply(num)
        totalVar.update(num,totalVar.apply(num)+diff * diff)
      }
    }
    ans.+=:(totalMeans.asInstanceOf[Vector],totalVar.asInstanceOf[Vector])
   ans.iterator
  }

  def axpy(a: Int,x: SparseVector,y:SparseVector):Unit ={
    val xIndexs = x.indices
    val xlength = xIndexs.length
    val xValues = x.values
    val yValues = y.values
    var k = 0
    while(k < xlength){
      yValues(xIndexs(k)) += a * xValues(k)
      k += 1
    }
  }

  def main(args: Array[String]): Unit = {
    val sc = getSparkContext()
    val partitionNum = sc.defaultMinPartitions
    initParams()
    val path = "/Users/apple/Desktop/avazu-app"
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
        //结果向量转矩阵
        val sumAll = total.treeReduce((x, y) => {
          val x1Sparse = x._1.asInstanceOf[SparseVector]
          val x2Sparse = x._2.asInstanceOf[SparseVector]
          val y1Sparse = y._1.asInstanceOf[SparseVector]
          val y2Sparse = y._2.asInstanceOf[SparseVector]
          axpy(1,x1Sparse,y1Sparse)
          axpy(1,x2Sparse,y2Sparse)
          (y1Sparse,y2Sparse)
        })
        val partitionMean = sumAll._1.toArray
        for(j <- 0 to partitionMean.length){
          partitionMean(j) = partitionMean(j)/partitionNum
        }
        val partitionVar = sumAll._2.toArray
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
