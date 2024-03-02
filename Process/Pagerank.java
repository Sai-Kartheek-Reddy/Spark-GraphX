import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Pattern;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;




public class Pagerank {
    private static final Pattern SPACES = Pattern.compile("\\s+");
    public static void main(String[] args) {
        /*if (args.length < 2) {
            System.err.println("Insufficient number of arguments. Please provide the input file and the number of iterations.");
            System.exit(1);
        }*/

        String sparkMaster = "spark://192.168.29.24:7077";
        String hdfsPath = "hdfs://192.168.29.24:9000/user/sx-stackoverflow.txt";

        SparkSession spark = SparkSession.builder()
                .appName("PageRank")
                .master(sparkMaster)
                .getOrCreate();

        // Loads in input file. It should be in the format of:
        // URL         neighbor URL
        // URL         neighbor URL
        // URL         neighbor URL
        // ...
        JavaRDD<String> lines = spark.read().textFile(hdfsPath).javaRDD();

        // Loads all URLs from input file and initialize their neighbors.
        JavaPairRDD<String, Iterable<String>> links = lines.mapToPair(line -> {
            String[] parts = SPACES.split(line);
            return new Tuple2<>(parts[0], parts[1]);
        }).distinct().groupByKey().cache();

        // Loads all URLs with other URL(s) linked to from input file and initialize ranks of them to one.
        JavaPairRDD<String, Double> ranks = links.mapValues(rs -> 1.0);

        int iterations = 10; // Set the number of iterations manually


        // Calculates and updates URL ranks continuously using the PageRank algorithm.
        for (int current = 0; current < iterations; current++) {
            // Calculates URL contributions to the rank of other URLs.
            JavaPairRDD<String, Double> contribs = links.join(ranks).values()
                    .flatMapToPair(s -> {
                        int urlCount = 0;
                        List<Tuple2<String, Double>> results = new ArrayList<>();
                        Iterator<String> iter = s._1().iterator();

                        while (iter.hasNext()) {
                            urlCount++;
                            results.add(new Tuple2<>(iter.next(), s._2() / urlCount));
                        }

                        return results.iterator();
                    });

            // Re-calculates URL ranks based on neighbor contributions.
            ranks = contribs.reduceByKey((a, b) -> a + b)
                    .mapValues(sum -> 0.15 + sum * 0.85);
        }

        // Collects all URL ranks and prints them.
        List<Tuple2<String, Double>> output = ranks.collect();
        for (Tuple2<String, Double> tuple : output) {
            System.out.println(tuple._1() + " has rank: " + tuple._2() + ".");
        }

        spark.stop();
    }
}
