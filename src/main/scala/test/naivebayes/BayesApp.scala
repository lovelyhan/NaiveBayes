package test.naivebayes

import org.apache.spark.{SparkConf, SparkContext}

object BayesApp {
  def main(args: Array[String]): Unit = {
    // spark 的初始化放到外面来, 方便修改配置
    val conf = new SparkConf()
      .setAppName("Bayes Test")
//      .setMaster("local")
//      .set("spark.executor.memory","1g")
    val sc = SparkContext.getOrCreate(conf)

    // path 也抽出来, 方便使用的时候配置
    val path = "/user/hadoop/data/avazu/avazu-small"
    val model = new NativeBayesModel()
    // 其它的可以修改的配置项也应该这样抽取出来作为参数传递进去
    val ansList =  model.train(path, sc)
    model.predict(path,sc,ansList)
  }
}
