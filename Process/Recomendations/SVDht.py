from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

# Define your HDFS and Spark cluster paths
hdfs_path = "hdfs://10.0.3.101:9000/user/amazon.edges"
spark_cluster_path = "spark://master:7077"

# Set up a Spark session
spark = SparkSession.builder.appName("SVDht").master(spark_cluster_path).getOrCreate()

# Load your dataset as a DataFrame
data = spark.read.text(hdfs_path).rdd \
    .map(lambda line: line.value.split(",")) \
    .map(lambda parts: Row(userId=int(parts[0]), productId=int(parts[1]), rating=float(parts[2])))

ratings = spark.createDataFrame(data)

# Set up the ALS (Alternating Least Squares) matrix factorization model
als = ALS(userCol="userId", itemCol="productId", ratingCol="rating")

# Define the parameter grid
param_grid = ParamGridBuilder() \
    .addGrid(als.rank, [10]) \
    .addGrid(als.maxIter, [2]) \
    .addGrid(als.regParam, [0.1]) \
    .build()

# Define the evaluator
evaluator = RegressionEvaluator(metricName="mse", labelCol="rating", predictionCol="prediction")

# Set up the cross-validator
cross_validator = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator)

# Fit the model
cv_model = cross_validator.fit(ratings)

# Get the best parameters from the cross-validation
best_rank = cv_model.bestModel.rank
best_max_iter = cv_model.bestModel._java_obj.parent().getMaxIter()
best_reg_param = cv_model.bestModel._java_obj.parent().getRegParam()

# Fit the ALS model with the best hyperparameters to get the correct MSE
best_model = ALS(rank=best_rank, maxIter=best_max_iter, regParam=best_reg_param,
                 userCol="userId", itemCol="productId", ratingCol="rating").fit(ratings)

# Define a user for whom you want to generate recommendations
user_id = 1
num_recommendations = 10

# Generate recommendations for the user using the best model
user_df = spark.createDataFrame([Row(userId=user_id)])
user_recommendations = best_model.recommendForUserSubset(user_df, num_recommendations)

# Display the top recommendations for the user
print("---------------------------------------------------------------")
print(f"Top {num_recommendations} recommendations for user {user_id}:")
print("---------------------------------------------------------------")
for row in user_recommendations.collect()[0]["recommendations"]:
    product = row["productId"]
    predicted_rating = row["rating"]
    print(f"Product: {product}, Predicted Rating: {predicted_rating}")
print("---------------------------------------------------------------")

# Print the best hyperparameters
print("---------------------------------------------------")
print("Best Hyperparameters:")
print(f"Rank: {best_rank}")
print(f"Number of Iterations: {best_max_iter}")
print(f"Regularization Parameter: {best_reg_param}")

# Get the MSE for the best model
predictions = best_model.transform(ratings)
best_mse = evaluator.evaluate(predictions)
print("------------------------------------------------------------------")

print(f"Mean Squared Error (MSE) with Best Hyperparameters: {best_mse}")
print("------------------------------------------------------------------")

# Stop the Spark session
spark.stop()

