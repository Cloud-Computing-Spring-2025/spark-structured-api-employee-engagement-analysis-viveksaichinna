from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, round as spark_round

def initialize_spark(app_name="Task1_Identify_Departments"):
    """
    Initialize and return a SparkSession.
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, file_path):
    """
    Load the employee data from a CSV file into a Spark DataFrame.
    """
    schema = """
        EmployeeID INT, 
        Department STRING, 
        JobTitle STRING, 
        SatisfactionRating INT, 
        EngagementLevel STRING, 
        ReportsConcerns BOOLEAN, 
        ProvidedSuggestions BOOLEAN
    """
    
    df = spark.read.csv(file_path, header=True, schema=schema)
    return df

def identify_departments_high_satisfaction(df):
    """
    Identify departments with more than 50% of employees having a Satisfaction Rating > 4 and Engagement Level 'High'.
    """
    # Filter employees with SatisfactionRating > 4 and EngagementLevel == 'High'
    satisfied_df = df.filter((col("SatisfactionRating") > 4) & (col("EngagementLevel") == "High"))
    
    # Count total employees per department
    total_count = df.groupBy("Department").agg(count("EmployeeID").alias("TotalEmployees"))
    
    # Count satisfied employees per department
    satisfied_count = satisfied_df.groupBy("Department").agg(count("EmployeeID").alias("SatisfiedEmployees"))
    
    # Calculate percentage of satisfied employees per department
    result_df = total_count.join(satisfied_count, "Department", "left").fillna(0)
    result_df = result_df.withColumn("Percentage", spark_round((col("SatisfiedEmployees") / col("TotalEmployees")) * 100, 2))
    
    # Filter departments where percentage > 50%
    result_df = result_df.filter(col("Percentage") > 5).select("Department", "Percentage")
    
    return result_df

def write_output(result_df, output_path):
    """
    Write the result DataFrame to a CSV file.
    """
    result_df.coalesce(1).write.csv(output_path, header=True, mode='overwrite')

def main():
    """
    Main function to execute Task 1.
    """
    # Initialize Spark
    spark = initialize_spark()
    
    # Define file paths
    input_file = "input/employee_data.csv"
    output_file = "outputs/task1/identify_departments.txt"
    
    # Load data
    df = load_data(spark, input_file)
    
    # Perform Task 1
    result_df = identify_departments_high_satisfaction(df)
    
    # Write the result to CSV
    write_output(result_df, output_file)
    
    # Stop Spark Session
    spark.stop()

if __name__ == "__main__":
    main()
