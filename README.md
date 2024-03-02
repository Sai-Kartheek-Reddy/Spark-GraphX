# README

## Abstract

Managing and optimizing hyperparameters for Singular Value Decomposition (SVD) on top of ALS and recommendations on temporal graphs, such as the Amazon Review dataset, can be a challenging task. It becomes even more complex when attempting to perform hyperparameter tuning on a single computer due to the extensive computational requirements.

To address this issue, our solution involves leveraging the power of a Spark cluster on a larger scale. This approach allows us to distribute the computational load across multiple nodes, making the hyperparameter tuning process more efficient and feasible.

## Key Points

1. **Challenge Identification:**
   - Detecting the right parameters for SVD recommendations on temporal graphs, like the Amazon Review dataset, is a difficult task.

2. **Hyperparameter Tuning:**
   - Hyperparameter tuning is essential for optimizing the performance of the SVD model in recommendation systems.

3. **Scaling Difficulties:**
   - Performing hyperparameter tuning on a single computer can be overwhelming and impractical due to the extensive computational requirements.

4. **Spark Cluster Integration:**
   - We overcome the scaling challenges by connecting to a Spark cluster on a larger scale.

5. **Computational Distribution:**
   - Distributing the computational load across multiple nodes in the Spark cluster enhances the efficiency of hyperparameter tuning.

6. **Optimization Results:**
   - The use of a Spark cluster significantly improves the optimization process, making it more manageable and effective.

## Project Structure

The project is organized into three main directories:

1. **Process:**
   - This directory contains the entire process of setting up, configuring, and executing the hyperparameter tuning for SVD recommendations on temporal graphs. Follow the steps outlined here to initiate the project.

2. **Progress:**
   - In this directory, you can track the progress of the project. It includes updates on what has been accomplished and any significant developments during the optimization process.

3. **Data:**
   - The 'Data' directory holds all relevant datasets, including the Amazon Review dataset or any other temporal graph data needed for the hyperparameter tuning.
   - Datasets are very large so only samples are uploaded in the repo thanks.
   - rec-amazon-ratings (https://networkrepository.com/rec-amazon-ratings.php)
   - sx-stackoverflow (https://snap.stanford.edu/data/sx-stackoverflow.html)
