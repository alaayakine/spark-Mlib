# LRApp - Linear Regression Application

The `LRApp` application is a Java program utilizing the Apache Spark MLlib library for linear regression analysis. The application performs the following steps:

1. **Set up Spark Session:**
   - Create a Spark session with the name "LRApp."
   - Set log levels to control console output.

2. **Load and Explore Data:**
   - Read advertising data from a CSV file ("advertising.csv") into a Spark DataFrame.
   - Print the schema and show the first 10 rows of the dataset.

3. **Feature Engineering:**
   - Use VectorAssembler to combine the "TV," "Radio," and "Newspaper" columns into a single feature column named "features."
   - Split the transformed dataset into training (85%) and testing (15%) sets.

4. **Linear Regression Model:**
   - Initialize a Linear Regression model, setting the label column as "Sales" and features column as "features."
   - Fit the model on the training data.
   - Make predictions on the test data.
   - Display the predictions.

     ![image](https://github.com/alaayakine/spark-Mlib/assets/106708512/812d2594-daae-4d58-8bf7-0691a0920902)


# KmeansApp - K-Means Clustering Application

The `KmeansApp` application is a Java program utilizing the Apache Spark MLlib library for K-Means clustering. The application performs the following steps:

1. **Set up Spark Session:**
   - Create a Spark session with the name "KmeansApp."
   - Set log levels to control console output.

2. **Load and Explore Data:**
   - Read customer data from a CSV file ("Mall_Customers.csv") into a Spark DataFrame.
   - Print the schema and show the first 10 rows of the dataset.

3. **Feature Engineering:**
   - Use VectorAssembler to combine "Age," "Annual Income (k$)," and "Spending Score (1-100)" into a single feature column named "Features."
   - Transform the dataset to include the vectorized features.
   - Normalize the features using MinMaxScaler.

4. **K-Means Clustering:**
   - Apply K-Means clustering on the normalized features with three clusters.
   - Fit the K-Means model to the transformed dataset.
   - Make predictions on the transformed dataset.
   - Show the dataset with added cluster predictions.

5. **Evaluation:**
   - Use ClusteringEvaluator to evaluate the clustering predictions.
   - Print the evaluation result.

     ![image](https://github.com/alaayakine/spark-Mlib/assets/106708512/8fc18e72-b10d-47a3-b0fd-e8cd57f2eb20)


Both applications demonstrate essential steps in building machine learning models using Spark MLlib, including data loading, exploration, feature engineering, model training, and result evaluation.
