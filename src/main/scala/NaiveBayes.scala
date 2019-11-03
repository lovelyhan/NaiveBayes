package main.scala

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.BLAS
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector}

object NaiveBayes {
  //多项式模型类别
  private[classification] val Multinomial:String = "multinomial"
  //伯努利模型类别
  private[classification] val Bernoulli:String = "bernoulli"

  //设置模型支持类别
  private[classification] val supportedModelTypes = Set(Multinomial,Bernoulli)

  //模型训练样本数据格式为RDD(label,features)
  def train(input: RDD[LabeledPoint]): NaiveBayesModel = {
    new NaiveBayes().run(input)
  }

  /**
    * 训练一个朴素贝叶斯模型，样本数据格式为RDD(label，features)
    * @param input RDD 样本RDD,格式为RDD(label，features)
    * @param lambda 平滑参数
    *
    * @param modelType 模型类别：多项式或伯努利
    */
  def train(input:RDD[LabeledPoint],lambda:Double,modelType:String):NaiveBayesModel = {
    require(supportedModelTypes.contains(modelType),s"NaiveBayes was created with an unknown modelType:$modelType.")
    new NaiveBayes(lambda,modelType).run(input)
  }
}

class NaiveBayes private(
                          private var lambda:Double,
                          private var modelType:String
                        )extends Serializable with Logging {
  import NaiveBayes.{Bernoulli,Multinomial}

  def this(lambda: Double) = this(lambda,NaiveBayes.Multinomial)

  def this() = this(1.0,NaiveBayes.Multinomial)

  //设置平滑参数
  def setLambda(lambda: Double): NaiveBayes = {
    this.lambda = lambda
    this
  }
  //平滑参数
  def getLambda: Double = lambda

  //设置模型类别
  def setModelType(modelType: String): NaiveBayes = {
    require(NaiveBayes.supportedModelTypes.contains(modelType),
      s"NaiveBayes was created with an unknown modelType:$modelType.")
    this.modelType = modelType
    this
  }

  //模型类别
  def getModelType: String = this.modelType

  //根据参数以及输入样本数据运行算法
  /**
    * @param data 样本RD，格式为RDD[[LabeledPoint]]
    */
  def run(data: RDD[LabeledPoint]): NaiveBayesModel = {
    val requireNonnegativeValues: Vector => Unit = (v: Vector) => {
      val values = v match {
        case sv: SparseVector => sv.values
        case dv: DenseVector => dv.values
      }
      if(!values.forall(_ >=0.0)){
        throw new SparkException(s"Naive Bayes requires nonnegative feature values but found $v.")
      }
    }

    val requireZeroOneBernoulliValues: Vector => Unit = (v: Vector) => {
      val values = v match {
        case sv : SparseVector => sv.values
        case dv : DenseVector => dv.values
      }
      if(!values.forall(v => v == 0.0 || v == 1.0)){
        throw new SparkException(
          s"Bernoulli naive Bayes requires 0 or 1 feature values but found $v.")
      }
    }
    //对每个标签进行聚合操作计算，求得每个标签对应特征的频数
    //aggreageted：以label为key，聚合同一个label的features
    //aggregated返回格式：（label，（计数，features之和））
    val aggregated = data.map(p => (p.label,p.features)).combineByKey[(Long,DenseVector)] (
      createCombiner = (v:Vector) => {
        if (modelType == Bernoulli){
          requireZeroOneBernoulliValues(v)
        } else{
          requireNonnegativeValues(v)
        }
        (1L,v.copy.toDense)
      },
      //mergeValue:将样本中的value合并到C类型的数据中(c:(Long,SparseVector),v:Vector)->(c:(Long,SparseVector)
      mergeValue = (c:(Long,DenseVector),v:Vector) => {
        requireNonnegativeValues(v)
        BLAS.axpy(1.0,v,c._2)
        (c._1 + 1L, c._2)
      },
      //mergeCombiners:根据每个key对应的多个进行合并，（c1:(Long,DenseVector),c2:(Long,DenseVector)->c:(Long,DenseVector)）
      mergeCombiners = (c1:(Long,DenseVector),c2:(Long,DenseVector)) => {
        BLAS.axpy(1.0,c2._2,c1._2)
      }
    ).collect()
    }

  }

}









