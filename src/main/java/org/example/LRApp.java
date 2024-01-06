package org.example;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class LRApp {
    public static void main(String[] args) {
        Logger.getLogger("org.apache.spark").setLevel(Level.ERROR);

        // Create a Spark session
        SparkSession spark = SparkSession.builder()
                .appName("LRApp")  // Set a meaningful application name
                .master("local[*]")    // Use "yarn" for deployment on a YARN cluster
                .getOrCreate();

        // Set application log level to INFO for relevant information
        Logger.getLogger("org.example").setLevel(Level.INFO);

        // Read data from a CSV file into a Spark DataFrame
        Dataset<Row> dataset = spark.read().option("inferSchema", true)
                .option("header", true)
                .csv("advertising.csv");

        // Print the schema of the dataset
        dataset.printSchema();

        // Show the first 10 rows of the dataset
         dataset.show(10);

        VectorAssembler assembler=new VectorAssembler().setInputCols(new String[]{ "TV","Radio","Newspaper"}).setOutputCol("features");

        Dataset<Row> transformedDS = assembler.transform(dataset);
        Dataset<Row>[] datasets = transformedDS.randomSplit(new double[]{0.85,0.15},27);

        Dataset<Row> train=datasets[0] ;
        train.show();
        Dataset<Row> test=datasets[0];
        test.show();


        transformedDS.printSchema();
        transformedDS.show();

        LinearRegression linearRegression=new LinearRegression().setLabelCol("Sales").setFeaturesCol("features");
        LinearRegressionModel fitted = linearRegression.fit(train);
        Dataset<Row> predictions = fitted.transform(test);
        predictions.show();

    }

}