import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.classification.{RandomForestClassifier,LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
import org.apache.spark.sql.SparkSession

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer ,StopWordsRemover}
import org.apache.spark.sql.functions._

object spammers {
  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("please input five parameters")
    }

    val sc = new SparkContext(new SparkConf().setAppName("spamDetect"))

    val spark = SparkSession
      .builder()
      .appName("SpamDetect")
      .getOrCreate()

    import spark.implicits._

    //Importing preprocesed dataset into a DataFrame
    val data=spark.read.option("header","true").option("inferSchema","true").csv(args(0))

    //Splitting the helpfulness column in two columns, one for the numerator and one for the denominator
    val df = data.withColumn("productId", $"productId").
      withColumn("userId", $"userId").
      withColumn("profileName", $"profileName").
      withColumn("helpfulness", $"helpfulness").
      withColumn("help_num", split($"helpfulness", "/").getItem(0)).
      withColumn("help_denom", split($"helpfulness", "/").getItem(1)).
      drop("helpfulness").
      withColumn("review_score", $"review_score").
      withColumn("review_time", $"review_time").
      withColumn("review_summary", $"review_summary").
      withColumn("text", $"text")

    //Removing rows with null values
    val pre_data = df.select("productId", "userId", "profileName", "help_num", "help_denom", "review_score", "review_time", "review_summary", "text")
    pre_data.na.drop().createOrReplaceTempView("view1") // create a temporary view of the table on which you can use SQL queries

    val reviews = spark.sql("SELECT * FROM view1")

    //Loading the AFINN file which conatins a sentiment score for individual words
    val sentiment_score = sc.textFile(args(1)).map(x => x.split("\t")).map(x => (x(0).toString, x(1).toInt))

    //create a map of the word and the sentiment score and broadcast it to all nodes
    val sentiment_score_new = sentiment_score.collectAsMap.toMap
    val broadcast_variable = sc.broadcast(sentiment_score_new)

    //Generate the sentiment score for each review
    val sentiment_score_review = reviews.map(a => {
      val sentiment_review_words = a(8).toString.split(" ").map(b => {
        val sentiment: Int = broadcast_variable.value.getOrElse(b.toLowerCase(), 0)
        sentiment;
      });
      val final_sentiment = sentiment_review_words.sum
      (a(0).toString, a(1).toString, a(2).toString, a(3).toString, a(4).toString, a(5).toString, a(6).toString, a(7).toString, a(8).toString, final_sentiment)
    })

    sentiment_score_review.createOrReplaceTempView("view2")


    val table = spark.sql("select _1 AS productID ,_2 AS userID ,_3 AS profileName, _4 AS Helpfulnr,_5 AS Helpfuldr, CAST(_6 AS INT) AS Score, CAST(_7 AS DOUBLE) AS time,_8 AS Summary,_9 AS Text,CAST(_10 AS INT) AS SentiScore from view2")


    table.createOrReplaceTempView("view3")

    //Average sentiment score of each user
    val score_user = spark.sql("select userID, avg(SentiScore) from view3 group by userID having avg(SentiScore)>20")

    //Compute Overall average sentiment score
    var avg_score = spark.sql("select avg(SentiScore) as score from view3 ")

    //Label the reviews based on the threshold sentiment score which is calculated above
    import org.apache.spark.sql.functions._
    val results = table.withColumn("SentiScore", when($"SentiScore" >= -4.0, 1).otherwise(0))
    results.show()
    results.createOrReplaceTempView("view4")

    pre_data.createOrReplaceTempView("view5")


    val mldatabase = spark.sql("SELECT productID,userID,Text,SentiScore as label FROM view4")

    //Create train and test dataset
    val Array(training, test) = mldatabase.randomSplit(Array(0.8, 0.2), seed = 12345)

    //Create a pipeline
    //Extract words from the reviews
    //Remove Stop words
    //Extract features from the words
    //Create Machine Learning models and feed the data
    val tknzer = new Tokenizer().setInputCol("Text").setOutputCol("words")
    val rmvr = new StopWordsRemover().setInputCol("words").setOutputCol("filtered").setCaseSensitive(false)
    val hshnTF = new HashingTF().setNumFeatures(1000).setInputCol("filtered").setOutputCol("rawFeatures")
    val i_d_f = new IDF().setInputCol("rawFeatures").setOutputCol("features").setMinDocFreq(0)
    val logistic = new LogisticRegression().setRegParam(0.01).setThreshold(0.5)
    val pipe_line = new Pipeline().setStages(Array(tknzer, rmvr, hshnTF, i_d_f, logistic))

    //Training the Logistic Regression model
    val lr_model = pipe_line.fit(training)

    //Predicting for the unseen reviews
    var predictions_logistic = lr_model.transform(test)

    //Evaluate the models
    val binary_classi = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
    val value = binary_classi.evaluate(predictions_logistic)
    var statistics="Logistic Regression's ROC curve area covered= " + binary_classi.evaluate(predictions_logistic)

    //Constructing RandomForest Model
    val random_forest = new RandomForestClassifier()
      .setNumTrees(100)
      .setFeatureSubsetStrategy("auto")
    val random = new Pipeline().setStages(Array(tknzer, rmvr, hshnTF, i_d_f, random_forest))

    val rf_model = random.fit(training)

    val 	 = rf_model.transform(test)

    //Evaluating RandomForest classifier
    val binary_classi_1 = new BinaryClassificationEvaluator().setMetricName("areaUnderROC")
    statistics+="\nRandom Forest's ROC curve area covered= " + binary_classi_1.evaluate(random)
    val value1 = binary_classi_1.evaluate(random)
    val parameter_grid = new ParamGridBuilder().
      addGrid(logistic.regParam, Array(0.4, 0.1, 0.2)).
      addGrid(logistic.threshold, Array(0.5, 0.6, 0.7)).
      build()

    //5 fold cross validation
    val cross_validator = new CrossValidator().setEstimator(pipe_line).setEvaluator(binary_classi).setEstimatorParamMaps(parameter_grid).setNumFolds(5)


    val cross_validator_model = cross_validator.fit(training)

    statistics+="\n\nArea under the ROC curve for non-tuned model = " + binary_classi.evaluate(random)
    statistics+="\nArea under the ROC curve for fitted model = " + binary_classi.evaluate(cross_validator_model.transform(test))
    statistics+="\nImprovement = " + "%.2f".format((binary_classi.evaluate(cross_validator_model.transform(test)) - binary_classi.evaluate(random)) * 100 / binary_classi.evaluate(random)) + "%"

    //Saving the results in files
    sc.parallelize(List(statistics)).saveAsTextFile(args(4))

    predictions_logistic.rdd.saveAsTextFile(args(2))
    random.rdd.saveAsTextFile(args(3))

  }
}