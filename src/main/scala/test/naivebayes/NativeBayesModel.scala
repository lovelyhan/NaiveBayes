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
    var meanValues0: Vector = _
    var meanValue1: Vector = _
    var variance0: Vector = _
    var variance1: Vector = _
  }
}

import test.naivebayes.NativeBayesModel.Model._
class NativeBayesModel() extends Serializable{
  @transient protected val logger: Logger = NativeBayesModel.logger
  final val dimensions = 999990
  val epoch = 1
  val batchNum = 20

    def loadData(sc: SparkContext,path: String):RDD[LabeledPoint] = {
      val startTime = System.currentTimeMillis()
      val dataRdd = MLUtils.loadLibSVMFile(sc,path)
        .coalesce(sc.defaultMinPartitions)
        .persist(StorageLevel.MEMORY_AND_DISK)
      val totalLength = dataRdd.count()
      logger.info(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
      println(s"Load data cost ${System.currentTimeMillis() - startTime} ms")
      dataRdd
    }

    protected def initParams(): Unit ={
      val means1,means2 = new Array[Double](dimensions)
      meanValues0 = Vectors.dense(means1)
      meanValue1 = Vectors.dense(means2)
      val var1,var2 = new Array[Double](dimensions)
      variance0 = Vectors.dense(var1)
      variance1 = Vectors.dense(var2)
    }

  def calculateMeansAndVariance(miniBatch: Iterator[LabeledPoint],totalMeans0: Vector,totalMeans1: Vector,totalVariance0: Vector,totalVariance1: Vector): Iterator[Tuple4[Array[Double],Array[Double],Array[Double],Array[Double]]] ={
    val ans = new ArrayBuffer[Tuple4[Array[Double],Array[Double],Array[Double],Array[Double]]]
    val totalMeans0 = new Array[Double](dimensions)
    val totalMeans1 = new Array[Double](dimensions)
    val totalVar0 = new Array[Double](dimensions)
    val totalVar1 = new Array[Double](dimensions)
    var totalCount0 = 0
    var totalCount1 = 0
    val data = miniBatch.toList
    val dataLength = data.length

    for(m <- 0 until data.length) {
      val labeledPoint = data.apply(m)
      val label = labeledPoint.label
      val features = labeledPoint.features.asInstanceOf[SparseVector]
      val keys = features.indices
      val values = features.values
      //Sum up all to calculate mean value
      for (i <- 0 until keys.length) {
        val num = keys.apply(i)
        if(label == 0.0){
          totalMeans0(num) += values.apply(i)
          totalCount0 += 1
        }
        else {
          totalMeans1(num) += values.apply(i)
          totalCount1 += 1
        }
      }
    }
    for(m <- 0 until dimensions){
      totalMeans0(m) /= totalCount0
      totalMeans1(m) /= totalCount1
    }
    for(n <- 0 until data.length){
      val labeledPoint = data.apply(n)
      val features = labeledPoint.features.asInstanceOf[SparseVector]
      val label = labeledPoint.label
      val keys = features.indices
      val values = features.values
      for(k <- 0 until keys.length){
        val num = keys.apply(k)
        val diff = values.apply(k) - totalMeans0(num)
        if(label == 0.0){
          totalVar0(num) += diff * diff
        }
        else{
          totalVar1(num) += diff * diff
        }
      }
    }
    ans.+=:(totalMeans0,totalMeans1,totalVar0,totalVar1)
   ans.iterator
  }

  def axpy(a: Double, x: Array[Double], y: Array[Double]): Unit = {
    for(i <- 0 to y.length - 1){
      y(i) += a * x(i)
    }
  }

  def train(path: String, sc: SparkContext): Unit = {
    val partitionNum = sc.defaultMinPartitions
    initParams()
    val rdd = loadData(sc, path)
    val rddTotalNum = rdd.count()
    for (epochTime <- 0 until epoch) {
      logger.info(s"Epoch[$epoch] start training")
      println(s"Epoch[$epoch] start training")
      for (batchNum <- 0 until batchNum) {
        logger.info(s"Iteration[$batchNum] starts")
        println(s"Iteration[$batchNum] starts")
        val startBatchTime = System.currentTimeMillis()
        val oneIterationRDD = rdd.sample(false, 0.05, 0L)
        //广播全局参数
        val broadcastMeans0 = sc.broadcast(meanValues0)
        val broadcastVariance0 = sc.broadcast(variance0)
        val broadcastMeans1 = sc.broadcast(meanValue1)
        val broadcastVariance1 = sc.broadcast(variance1)
        val total = oneIterationRDD.mapPartitions(miniList => calculateMeansAndVariance(miniList, broadcastMeans0.value, broadcastMeans1.value,broadcastVariance0.value,broadcastVariance1.value))
        logger.info(s"Mean value and variance are calculated,took${System.currentTimeMillis() - startBatchTime} ms")
        println(s"Mean value and variance are calculated,took${System.currentTimeMillis() - startBatchTime} ms")
        val sumAll = total.treeReduce((x,y) => {
          axpy(1.0,x._1,y._1)
          axpy(1.0,x._2,y._2)
          axpy(1.0,x._3,y._3)
          axpy(1.0,x._4,y._4)
          y
        })
        val partitionMean0 = sumAll._1
        for(j <- 0 to partitionMean0.length - 1){
          partitionMean0(j) = partitionMean0(j)/partitionNum
        }
        val partitionMean1 = sumAll._2
        for(k <- 0 to partitionMean1.length - 1){
          partitionMean1(k) = partitionMean1(k) / partitionNum
        }
        val partitionVar0 = sumAll._3
        for(j <- 0 to partitionVar0.length - 1){
          partitionVar0(j) = partitionVar0(j)/partitionNum
        }
        val partitionVar1 = sumAll._4
        for(j <- 0 to partitionVar1.length - 1){
          partitionVar1(j) = partitionVar1(j)/partitionNum
        }
        meanValues0 = Vectors.dense(partitionMean0)
        meanValue1 = Vectors.dense(partitionMean1)
        variance0 = Vectors.dense(partitionVar0)
        variance1 = Vectors.dense(partitionVar1)
        logger.info(s"This term's calculation finished,totally took ${System.currentTimeMillis() - startBatchTime} ms")
        println(s"This term's calculation finished,totally took ${System.currentTimeMillis() - startBatchTime} ms")
      }
    }
  }

  //预测
  def predict(path: String, sc: SparkContext,meanValues0: Vector,meanValue1:Vector,variance0:Vector,variance1:Vector):Unit = {
    val rdd = loadData(sc, path)
    val predictRdd = rdd.sample(false,0.01,0)
    val rddTotalNum = predictRdd.count()
    var right = 0
    val ans = predictRdd.map(x =>{
      //whetherTrue is used to compare whethre the prediction of our model
      //is equivalent to the original label
      val whetherTrue = 0.0
      val label = x.label
      var pre0 = 1.0
      var pre1 = 1.0
      val features = x.features.asInstanceOf[SparseVector]
      val keys = features.indices
      val values = features.values
      //Sum up all to calculate mean value
      for (i <- 0 until keys.length) {
        //get the number of the feature
        val num = keys.apply(i)
        val xValue = values.apply(i)
        //prepare to predict
        val mean_0 = meanValues0.apply(num)
        val mean_1 = meanValue1.apply(num)
        val var_0 = variance0.apply(num)
        val var_1 = variance1.apply(num)
        var predict0 = (-1.0) * (xValue - mean_0) * (xValue - mean_0) / (2 * var_0)
        predict0 = math.exp(predict0)
        predict0 = (1.0) / math.sqrt(2 * math.Pi * var_0) * predict0
        var predict1 = (-1.0) * (xValue - mean_1) * (xValue - mean_1) / (2 * var_1)
        predict1 = math.exp(predict1)
        predict1 = (1.0) / math.sqrt(2 * math.Pi * var_1) * predict1
        pre1 *= predict0
      }
      //compare the value of prediction
      var finalSelect = 0.0
      if(pre1 >= pre0){
        finalSelect = 1.0
      }
      if (finalSelect == label){
        finalSelect = 1.0
      }
      finalSelect
    })
    val finalTotal = ans.reduce((x,y) => {
     x+y
    })
    val rate = 1.0 * finalTotal / rddTotalNum
    logger.info(s"The accuracy of model is ${rate}")
    println(s"The accuracy of model is ${rate}")
  }
}
