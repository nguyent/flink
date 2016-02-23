package org.apache.flink.ml.pca

import org.apache.flink.api.scala._
import org.apache.flink.ml.math._
import org.apache.flink.test.util.FlinkTestBase

import org.scalatest.{Matchers, FlatSpec}

class PCASuite
  extends FlatSpec
    with Matchers
    with FlinkTestBase {

  behavior of "The Principal Component Analysis (PCA) implementation"

    it should "use the dimensions attribute" in {
      val env = ExecutionEnvironment.getExecutionEnvironment

      val maxVectorSize = 10
      val numericArray = (1 to maxVectorSize).toArray
      val basicVector = DenseVector(numericArray)
      val maxVectors = 5

      val vData = List.fill(maxVectors)(basicVector)
      val inputDS = env.fromCollection(vData)

      for (dimension <- 1 until maxVectors) {
        val pca = PrincipalComponentAnalysis.pca(dimension, inputDS, env)
        pca.count() should equal(dimension)
      }
    }

    it should "properly compute the Principal Components" in {
      val env = ExecutionEnvironment.getExecutionEnvironment
      val testX = DenseVector(2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1)
      val testY = DenseVector(2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9)
      val vData = List(testX, testY)
      val testDS = env.fromCollection(vData)

      val pca: DataSet[(Double, DenseVector)] = PrincipalComponentAnalysis.pca(1, testDS, env)
      val principalComponent: DenseVector = pca.collect()(0)._2

      principalComponent should equal(DenseVector(0.6778733985280118, 0.735178655544408))
    }
}
