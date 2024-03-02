import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.Model;  // Import Model interface
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

// Import the Rating class from the appropriate package
import org.apache.spark.mllib.recommendation.Rating;

public class SVD {
    public static void main(String[] args) {
        // Define your HDFS and Spark cluster paths
        String hdfsPath = "hdfs://10.0.14.16:9000/user/amazon.edges";
        String sparkClusterPath = "spark://master:7077";

        // Set up a Spark configuration and session
        SparkConf sparkConf = new SparkConf().setAppName("SVD").setMaster(sparkClusterPath);
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        SparkSession spark = SparkSession.builder().appName("SVD").getOrCreate();

        // Load your dataset as a JavaRDD of Rating objects
        JavaRDD<Rating> ratings = jsc.textFile(hdfsPath)
                .map(line -> {
                    String[] parts = line.split(",");
                    int userId = Integer.parseInt(parts[0]);
                    int productId = Integer.parseInt(parts[1]);
                    double rating = Double.parseDouble(parts[2]);
                    return new Rating(userId, productId, rating);
                });

        // Convert JavaRDD to DataFrame
        Dataset<Row> ratingsDF = spark.createDataFrame(ratings, Rating.class)
                .toDF("userId", "productId", "rating");  // Use toDF directly

        // Split the data into training and test sets (80% training, 20% testing)
        Dataset<Row>[] splits = ratingsDF.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Set up the ALS (Alternating Least Squares) matrix factorization model
        ALS als = new ALS()
                .setMaxIter(10)
                .setRegParam(0.01)
                .setUserCol("userId")
                .setItemCol("productId")
                .setRatingCol("rating");

        // Define the parameter grid for hyperparameter tuning
        ParamGridBuilder paramGrid = new ParamGridBuilder()
                .addGrid(als.rank(), new int[]{5, 10, 15})
                .addGrid(als.regParam(), new double[]{0.01, 0.1, 1.0});

        // Set up the evaluator
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rating")
                .setPredictionCol("prediction");

        // Set up cross-validation
        CrossValidator crossValidator = new CrossValidator()
                .setEstimator(als)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid.build())
                .setNumFolds(5);

        // Fit the ALS model to the training data
        Model model = crossValidator.fit(trainingData);  // Use Model interface

        // Generate predictions on the test data
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate the model
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        // Print the best hyperparameters
        System.out.println("Best hyperparameters: " + model.bestModel().extractParamMap());

        // Stop the Spark context
        jsc.stop();
    }
}

