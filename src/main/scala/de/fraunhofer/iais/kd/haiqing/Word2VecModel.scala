package de.fraunhofer.iais.kd.haiqing

import java.io.{File, PrintWriter}

import com.github.fommil.netlib.BLAS._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import com.github.fommil.netlib.BLAS.{getInstance => blas}

/**
  * Created by hwang on 26.04.16.
  */

case class VocabWord(
                      var word: String,
                      var cn: Int
                    )

class Word2VecModel(
                     private val wordIndex: Map[String, Int],
                     private val wordVectors: Array[Float]) {


  private val numWords = wordIndex.size
  private val vectorSize = wordVectors.length / numWords

  private val wordList: Array[String] = {
    val (wl, _) = wordIndex.toSeq.sortBy(_._2).unzip
    wl.toArray
  }

  private val wordVecNorms: Array[Double] = {
    val wordVecNorms = new Array[Double](numWords)
    var i = 0
    while (i < numWords) {
      val vec = wordVectors.slice(i * vectorSize, i * vectorSize + vectorSize)
      wordVecNorms(i) = blas.snrm2(vectorSize, vec, 1)
      i += 1
    }
    wordVecNorms
  }

  private def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / norm1 / norm2
  }

  def transform(word: String): Vector = {
    wordIndex.get(word) match {
      case Some(ind) =>
        val vec = wordVectors.slice(ind * vectorSize, ind * vectorSize + vectorSize)
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    val cosineVec = Array.fill[Float](numWords)(0)
    val alpha: Float = 1
    val beta: Float = 0

    blas.sgemv(
      "T", vectorSize, numWords, alpha, wordVectors, vectorSize, fVector, 1, beta, cosineVec, 1)

    // Need not divide with the norm of the given vector since it is constant.
    val cosVec = cosineVec.map(_.toDouble)
    var ind = 0
    while (ind < numWords) {
      cosVec(ind) /= wordVecNorms(ind)
      ind += 1
    }

    wordList.zip(cosVec)
      .toSeq
      .sortBy(-_._2)
      .take(num + 1)
      .tail
      .toArray
  }

  def getVectors: Map[String, Array[Float]] = {
    wordIndex.map { case (word, ind) =>
      (word, wordVectors.slice(vectorSize * ind, vectorSize * ind + vectorSize))
    }
  }

  def save(path: String): Unit = {
    val file1 = new PrintWriter(new File(path + "/wordIndex.txt"))
    val file2 = new PrintWriter(new File(path + "/wordVectors.txt"))
    val iter = wordIndex.toIterator
    while (iter.hasNext) {
      val tmp = iter.next()
      file1.write(tmp._1 + " " + tmp._2 + "\n")
    }
    for (i <- 0 to wordVectors.size - 2)
      file2.write(wordVectors(i) + " ")
    file2.write(wordVectors(wordVectors.size - 1) + "\n")
    file1.close()
    file2.close()
  }
}