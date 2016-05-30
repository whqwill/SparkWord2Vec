import _root_.sbtassembly.AssemblyPlugin.autoImport._
import _root_.sbtassembly.PathList

mainClass in assembly := Some("de.fraunhofer.iais.kd.haiqing.Main_sense")

assemblyJarName in assembly := "SparkWord2Vec.jar" // name of generated jar

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

test in assembly := {} // {} means: no test

val meta = """META.INF(.)*""".r

assemblyMergeStrategy in assembly := {
  case PathList("javax", "servlet", xs@_*) => MergeStrategy.last
  case PathList(ps@_*) if ps.last endsWith ".html" => MergeStrategy.first
  case n if n.startsWith("reference.conf") => MergeStrategy.concat
  case n if n.endsWith(".conf") => MergeStrategy.concat
  case meta(_) => MergeStrategy.discard
  case x => MergeStrategy.last
}


name := "SparkWord2Vec"

version := "1.0"

scalaVersion := "2.11.7"

organization := "Fraunhofer IAIS"

description := "Spark Word2Vec implementation with word senses"

    