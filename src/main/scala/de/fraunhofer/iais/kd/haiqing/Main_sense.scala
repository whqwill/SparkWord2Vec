package de.fraunhofer.iais.kd.haiqing

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable

/**
  * Created by hwang on 09.02.16.
  * recommended VM-Options : -Xmx4000m -Dspark.executor.memory=2g
  */
object Main_sense {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN) // omit all iteration logging
    Logger.getLogger("akka").setLevel(Level.WARN) // omit all iteration logging
    val conf = new SparkConf().setAppName("Sense2Vec")
    if (!conf.contains("spark.master")) {
      conf.setMaster("local[*]")
      conf.setMaster("local[4]")
    }
    val sc = new SparkContext(conf)
    println("sc.defaultParallelism=" + sc.defaultParallelism + "   " + sc.master)
    val argNam = Array("inpFile", "numRdd", "numEpoch", "minCount", "numNegative", "windowSize", "vectorSize",
      "count2numSenses", "learnRate", "stepSize", "local", "1-senseModelPath", "multi-senseModelPath")
    for (i <- 0 until args.length)
      println("arg(" + i + ") " + argNam(i) + " = " + args(i))

    // arg0 = input file  /home/gpaass/news.2013.en.shuffled.001fraction sentences
    // /data/Dokumente/research/masterWang/news.2013.en.shuffled.001fraction 10 10 10 20 5 50 20_30 0.01 10000 true
    // /data/Dokumente/research/masterWang/resultsData/r1sense /data/Dokumente/research/masterWang/resultsData/rmsense
    // input file should contain lines of text, each line containing a sentence
    val input = sc.textFile(args(0), sc.defaultParallelism)
    val lines = input.take(2)
    println("read " + input.count() + " lines of text from file " + args(0))
    for (i <- 0 until lines.length)
      println(lines(i))
    val ENCODE = 100
    val oneSense = (args.length == 12)
    //val oneSense = true
    val softMax = false // softMax or sigmoid
    val modelPathMultiSense = if (args.length < 13) "" else args(12)
    val modelSaveIter = if (oneSense) args(1).toInt*args(2).toInt else 20 // save the model after this number of iterations
    println(modelSaveIter)
    val validationRation = 0.1f // max. fraction of data to use for validation
    val modelValidateIter = 2 //  validate the model after this number of iterations
    val validationRatio = 0.1f // maximum fraction of data to use for validation
    val maxValidationSize: Int = 10000 // maximum number of sentences for validation
    val validationIsSubset = true // select validationset as subset of trainingssets
    val maxEmbNorm = 5.0f
    val senseProbThresh = 0.02f // re-initialize embedding if senseProbability is lower

    val senseModel = new SenseAssignment(
      args(0), // inputfile
      args(1).toInt, // numRdds
      args(2).toInt, // numEpoch = iterations through whole dataset
      args(3).toInt, // minCount = minimal count of words to include
      args(7), // thresholds for count -> number of senses val count2numSenses:Array[Int],
      42l, // seed
      args(10).toBoolean, // true not use cluster
      validationRatio,
      maxValidationSize, // maximum number of sentences for validation
      validationIsSubset // true if validation set is subset of trainingset
    )
    senseModel.setModelConst(
      args(4).toInt, // 5 number of negative samples
      args(5).toInt, // 5 window size
      args(6).toInt, // 50 embedding size
      args(8).toFloat, // 0.025 beginning learning rate
      ENCODE, // multiplier for word number, mu
      args(9).toInt, // 10000 how many word words are processed before reducing learning rate
      oneSense, // indicator for using only 1 sense
      softMax, // indicator for sftMax or sigmoid activation
      args(11), //synPath path with stored model with 1 sense
      modelPathMultiSense, //outputPath path to write multisense model
      modelSaveIter, // save the model after this number of iterations
      modelValidateIter, // validate the model after this number of iterations
      maxEmbNorm, // maximum square length of embedding. If larger the vectors are scaled down
      senseProbThresh) //re-initialize embedding if senseProbability is lower

    senseModel.trainWrapper(
      input //outputPath path to write multisense model
    )

  }
}