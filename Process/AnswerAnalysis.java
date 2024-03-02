import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import scala.Tuple2;

public class AnswerAnalysis {

    public static void main(String[] args) {
        // Configure Spark
        SparkConf conf = new SparkConf()
                .setAppName("AnswerAnalysis")
                .setMaster("spark://192.168.29.24:7077");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Read the dataset from HDFS
        String hdfsPath = "hdfs://192.168.29.24:9000/user/sx-stackoverflow.txt";
        JavaRDD<String> lines = sc.textFile(hdfsPath);

        // Split each line into columns and extract the user ID (SRC)
        JavaRDD<String> userIDs = lines.map(line -> line.split(" ")[0]);

        // Count the number of questions answered by each user
        JavaRDD<Tuple2<String, Integer>> userAnswerCounts = userIDs.mapToPair(userID -> new Tuple2<>(userID, 1))
                .reduceByKey((count1, count2) -> count1 + count2)
                .map(pair -> new Tuple2<>(pair._1, pair._2));

        // Sort the results in ascending order based on the count
        JavaRDD<Tuple2<String, Integer>> sortedUserAnswerCounts = userAnswerCounts
                .sortBy(pair -> pair._2, true, 1);

        // Collect the results
        java.util.List<Tuple2<String, Integer>> result = sortedUserAnswerCounts.collect();

        // Print the results
        for (Tuple2<String, Integer> pair : result) {
            System.out.println("User " + pair._1 + " answered " + pair._2 + " question(s)");
        }

        // Stop the Spark context
        sc.stop();
    }
}

