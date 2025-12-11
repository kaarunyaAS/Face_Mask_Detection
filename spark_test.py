from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("TestSpark").getOrCreate()

# Create a small DataFrame
data = [("Kaarunya", 21), ("Kabhin", 22), ("Phoenix", 999)]
columns = ["Name", "Age"]

df = spark.createDataFrame(data, columns)

# Trigger a Spark action
df.show()

# Keep the app running to view the UI
input("Press Enter to close Spark...")
