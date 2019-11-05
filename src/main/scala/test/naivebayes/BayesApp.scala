package main.scala.test.naivebayes

import main.scala.test.naivebayes.NativeBayesModel
import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import test.naivebayes.NativeBayesModel.Model.{meanValues, variance}

object BayesApp {
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
