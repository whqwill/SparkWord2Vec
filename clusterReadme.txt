0) check Program versions on local machine and cluster nodes, check .bashrc, check build.sbt
    java: jdk1.7.0
        java -version
    scala: scala-2.11.7
    spark: 1.6.0
        try to run /opt/spark-1.6.0/bin/spark-shell

1) Intellij: Open an Sbt console
2) Intellij: Start it (click on the green arrow). The compilation is performed
3) Enter the command "assembly"
   Create an jar with all dependent libraries, which can be executed on the cluster
   parameters in build.sbt:
      mainClass in assembly := Some("de.fraunhofer.iais.kd.annSeq.ContrastSenseModelRunnerCluster")
      assemblyJarName in assembly := "annSeq_v2.jar"

   The resulting jar is in .../target/scala-2.11/
   success: correct run
4) On a cluster node (e.g. dpl01.kdlan) create directory SparkDeepLearn with user rights a+x
   Copy jar to the cluster node (with scp in a terminal)
   scp /home/gpaass/Intellij/SparkDeepGradAndDataCluster/theseus-ctc/SparkDeepGradAndData/target/scala-2.11/annSeq_v2.jar gpaass@str01.kdlan:/home/IAIS/gpaass/SparkDeepLearn/

5) Login on dpl01.kdlan (or other node with implemented Spark)
   spark-submit options are explained in Learning Spark p.122,
   online on http://spark.apache.org/docs/latest/submitting-applications.html
   spark-submit can accept any Spark property using the --conf flag
   Running ./bin/spark-submit --help will show the entire list of these options.
   options  --spark.master              : IP of master + port   The cluster manager to connect to
            --spark.executor.memory 2g  Amount of memory to use per executor process (e.g. 2g, 8g).
            --spark.driver.memory 5g    Amount of memory to use for the driver process, where SparkContext is
                                        initialized.
            --local-dir /data/user/gpaass  Directory to use for "scratch" space in Spark, including map output files
                                        and RDDs that get stored on disk.
            --spark.cores.max           the maximum amount of CPU cores to request for the application from across the
                                        cluster (not from each machine)



   /opt/spark-1.6.0/bin/spark-submit --master spark://10.116.44.51:7077 --deploy-mode cluster --cores.max=10 \
       --executor-memory 2g --driver-memory 15g --conf "spark.default.parallelism=351" \
       --class de.fraunhofer.iais.kd.annSeq.ContrastSenseModelRunnerCluster ./SparkDeepLearn/annSeq_v2.jar \
       inpFileName /home/IAIS/gpaass/WikiTextsLines/docs.txt numHidden 100 noutDim 50 maxNumToken 100000 \
       numIter 5 maxLag 5 inpSuffix 1 outdir /home/IAIS/gpaass/tmp/ nrdd 150 \
        setIbatchCumDataSaveFreq 200

   /opt/spark-1.6.0/bin/spark-submit --master spark://10.116.44.51:6066 --deploy-mode cluster --total-executor-cores 10 --executor-memory 2g --driver-memory 15g --conf "spark.default.parallelism=351" --class de.fraunhofer.iais.kd.annSeq.ContrastSenseModelRunnerCluster ./SparkDeepLearn/annSeq_v2.jar inpFileName /home/IAIS/gpaass/WikiTextsLines/docs.txt numHidden 100 noutDim 50 maxNumToken 100000 numIter 5 maxLag 5 inpSuffix 1 outdir /home/IAIS/gpaass/tmp/ nrdd 150 setIbatchCumDataSaveFreq 200

6)  kill job: kill drivers in the GUI

7)  diagnosis of running program: open in web browser ip-address of master:8080
    http://10.116.44.51:8080/
    http://<driver>:4040 lists Spark properties in the “Environment” tab.


  old call
 /opt/spark-1.6.0/bin/spark-submit --master spark://10.116.44.51:7077 --deploy-mode cluster --executor-memory 80g \
   --driver-memory 15g --conf "spark.default.parallelism=191" ./SparkDeepLearn/annSeq_v2.jar

 old call
 /opt/spark-1.6.0/bin/spark-submit "-Dspark.master=m" "-Dakka.loglevel=WARNING" "-Dspark.executor.memory=80g" \
 "-Dspark.driver.supervise=false" "-Dspark.app.name=de.fraunhofer.iais.kd.annSeq.ContrastSenseModelRunnerCluster" \
 "-Dspark.submit.deployMode=cluster" "-Dspark.rpc.askTimeout=10" "-Dspark.jars=file:/home/IAIS/gbernard/annSeq_v2.jar"\
 "-Dspark.driver.memory=15g" "-Dspark.default.parallelism=191" "org.apache.spark.deploy.worker.DriverWrapper" \
 "spark://Worker@10.116.44.54:45907" "/opt/spark/work/driver-20160317151424-0074/annSeq_v2.jar" \
 "de.fraunhofer.iais.kd.annSeq.ContrastSenseModelRunnerCluster" \
 "inpFileName" "/home/IAIS/gpaass/WikiTextsLines/docs.txt" "numHidden" "100" "noutDim" "50" "maxNumToken" "10000" \
 "numIter" "5" "maxLag" "5" "inpSuffix" "1" "outdir" "/home/IAIS/gbernard/tmp/" "nrdd" "300" \
 "setIbatchCumDataSaveFreq" "200"