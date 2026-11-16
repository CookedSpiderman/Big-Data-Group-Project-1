from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CS4371-GP1").getOrCreate()

print("Spark version:", spark.version)

df = spark.range(5)
df.show()

spark.stop()
