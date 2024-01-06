package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class KmeansApp {
    public static void main(String[] args) {
        // Set Spark log level to ERROR to reduce unnecessary output
        Logger.getLogger("org.apache.spark").setLevel(Level.ERROR);

        // Create a Spark session
        SparkSession spark = SparkSession.builder()
                .appName("KmeansApp")  // Set a meaningful application name
                .master("local[2]")    // Use "yarn" for deployment on a YARN cluster
                .getOrCreate();

        // Set application log level to INFO for relevant information
        Logger.getLogger("org.example").setLevel(Level.INFO);

        // Read data from a CSV file into a Spark DataFrame
        Dataset<Row> dataset = spark.read().option("inferSchema", true)
                .option("header", true)
                .csv("Mall_Customers.csv");

        // Print the schema of the dataset
        dataset.printSchema();

        // Show the first 10 rows of the dataset
        dataset.show(10);

        // Vectorize the features for clustering using VectorAssembler
        VectorAssembler vectorAssembler = new VectorAssembler().setInputCols(
                        new String[]{"Age", "Annual Income (k$)", "Spending Score (1-100)"})
                .setOutputCol("Features");

        // Transform the dataset to include the vectorized features
        Dataset<Row> transformed = vectorAssembler.transform(dataset);

        // Print the schema of the transformed dataset
        transformed.printSchema();

        // Show the transformed dataset with vectorized features
        transformed.show();
        MinMaxScaler minMaxScaler=new MinMaxScaler().setInputCol("Features").setOutputCol("NormalizedFeatures");
        Dataset<Row> normalizedData = minMaxScaler.fit(transformed).transform(transformed);
        // Apply KMeans clustering on the vectorized features
        KMeans kMeans = new KMeans().setFeaturesCol("NormalizedFeatures")
                .setK(3)  // Set the number of clusters
                .setPredictionCol("clusters").setSeed(123).setPredictionCol("prediction");  // Set the column for predicted clusters

        // Fit the KMeans model to the transformed dataset
        KMeansModel model = kMeans.fit(normalizedData);

        // Make predictions on the transformed dataset
        Dataset<Row> predictions = model.transform(normalizedData);

        // Show the dataset with added cluster predictions
        predictions.show(200);
        ClusteringEvaluator evaluator=new ClusteringEvaluator().setPredictionCol("prediction").setFeaturesCol("NormalizedFeatures");
        double evaluate = evaluator.evaluate(predictions);
        System.out.println(evaluate);
    }
}
