package main.scala.test.naivebayes

import org.apache.spark.mllib.linalg.{SparseVector, Vectors}
import test.naivebayes.NativeBayesModel

object BayesApp {
  def main(args: Array[String]): Unit = {
    val nativeBayesModel = new NativeBayesModel()
    val sc = nativeBayesModel.getSparkContext()
    val partitionNum = sc.defaultMinPartitions
    nativeBayesModel.initParams()
    val path = "/user/hadoop/data/avazu/avazu-app"
    val rdd = nativeBayesModel.loadData(sc, path)
    val rddTotalNum = rdd.count()
    for (epochTime <- 0 until nativeBayesModel.epoch) {
      nativeBayesModel.logger.info(s"Epoch[${nativeBayesModel.epoch}] start training")
      for (batchNum <- 0 until nativeBayesModel.batchNum) {
        nativeBayesModel.logger.info(s"Iteration[$batchNum] starts")
        val startBatchTime = System.currentTimeMillis()
        val oneIterationRDD = rdd.sample(false, 0.05, 0L)
        //广播全局参数
        val broadcastMeans = sc.broadcast(nativeBayesModel.meanValues)
        val broadcastVariance = sc.broadcast(nativeBayesModel.variance)
        val total = oneIterationRDD.mapPartitions(miniList => nativeBayesModel.calculateMeansAndVariance(miniList, broadcastMeans.value, broadcastVariance.value))
        nativeBayesModel.logger.debug(s"Mean value and variance are calculated,took${System.currentTimeMillis() - startBatchTime} ms")
        //结果向量转矩阵
        val sumAll = total.treeReduce((x, y) => {
          val x1Sparse = x._1.asInstanceOf[SparseVector]
          val x2Sparse = x._2.asInstanceOf[SparseVector]
          val y1Sparse = y._1.asInstanceOf[SparseVector]
          val y2Sparse = y._2.asInstanceOf[SparseVector]
          nativeBayesModel.axpy(1,x1Sparse,y1Sparse)
          nativeBayesModel.axpy(1,x2Sparse,y2Sparse)
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
        nativeBayesModel.meanValues = Vectors.dense(partitionMean)
        nativeBayesModel.variance = Vectors.dense(partitionVar)
        nativeBayesModel.logger.debug(s"This term's calculation finished,totally took ${System.currentTimeMillis() - startBatchTime} ms")
      }
    }
  }

}
