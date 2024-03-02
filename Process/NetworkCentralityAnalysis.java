import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class NetworkCentralityAnalysis {

    public static void main(String[] args) {
        // Configure Spark
        SparkConf conf = new SparkConf()
                .setAppName("NetworkCentralityAnalysis")
                .setMaster("spark://192.168.29.24:7077");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // Create a SparkSession
        SparkSession spark = SparkSession.builder().getOrCreate();

        // Read the dataset from HDFS
        String hdfsPath = "hdfs://192.168.29.24:9000/user/sx-stackoverflow.txt";
        Dataset<Row> data = spark.read().text(hdfsPath);

        // Split the lines and convert to DataFrame
        Dataset<Row> splitData = data.selectExpr("split(value, ' ') as columns")
                .selectExpr("columns[0] as src", "columns[1] as tgt");

        // Calculate the degree centrality for each user
        Dataset<Row> outDegree = splitData.groupBy("src").count().withColumnRenamed("count", "degreeCentrality");

        // Collect the results
        Row[] centralityRows = (Row[]) outDegree.collect();

        // Print the degree centrality for each user
        for (Row row : centralityRows) {
            String userId = row.getAs("src");
            long degreeCentrality = row.getAs("degreeCentrality");
            System.out.println("The Degree of Centrality of User " + userId + " is " + degreeCentrality);
        }

        // Stop the Spark context
        sc.stop();
    }
}

