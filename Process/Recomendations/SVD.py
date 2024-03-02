from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# Define your HDFS and Spark cluster paths
hdfs_path = "hdfs://10.0.1.204:9000/user/amazon.edges"
spark_cluster_path = "spark://master:7077"

# Set up a Spark session
spark = SparkSession.builder.appName("SVD").master(spark_cluster_path).getOrCreate()

# Load your dataset as a DataFrame
data = spark.read.text(hdfs_path).rdd \
    .map(lambda line: line.value.split(",")) \
    .map(lambda parts: Row(userId=int(parts[0]), productId=int(parts[1]), rating=float(parts[2])))

ratings = spark.createDataFrame(data)

# Set up the ALS (Alternating Least Squares) matrix factorization model
rank = 10  # Number of latent factors
num_iterations = 10  # Number of iterations
als = ALS(rank=rank, maxIter=num_iterations, regParam=0.01, userCol="userId", itemCol="productId", ratingCol="rating")
model = als.fit(ratings)

# Define a user for whom you want to generate recommendations
user_id = 1
num_recommendations = 10

# Generate recommendations for the user
user_df = spark.createDataFrame([Row(userId=user_id)])
user_recommendations = model.recommendForUserSubset(user_df, num_recommendations)

# Display the top recommendations for the user
print("---------------------------------------------------------------")
print(f"Top {num_recommendations} recommendations for user {user_id}:")
for row in user_recommendations.collect()[0]["recommendations"]:
    product = row["productId"]
    predicted_rating = row["rating"]
    print(f"Product: {product}, Predicted Rating: {predicted_rating}")
print("---------------------------------------------------------------")
# Stop the Spark session
spark.stop()

