import polars as pl
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

# 1. Build your data in Polars as usual
pdf = pl.DataFrame({
    "block_height": [1, 2, 3],
    "tx_count": [1200, 980, 1450],
    "btc_price": [61234.5, 61890.2, 62001.0],
}).to_pandas()  # Spark needs pandas or Arrow, not a Polars object directly

# 2. Connect to serverless compute remotely
spark = DatabricksSession.builder.serverless().profile("DEFAULT").getOrCreate()


spark = DatabricksSession.builder.remote(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN")
).serverless(True).getOrCreate()


# 3. Convert to Spark DataFrame and write as plain Parquet (no Delta, no table registration)
sdf = spark.createDataFrame(pdf)
sdf.write.mode("overwrite").parquet("/Volumes/workspace/default/my_volume/test_data.parquet")


# list the existing volumes
volumes = spark.sql("SHOW VOLUMES").collect()
volumes

# list the existing catalogues
catalogs = spark.sql("SHOW CATALOGS").collect()
catalogs



spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.my_volume")



# databricks functions 

# --------- // write to volume

def write_to_databricks(file_name:str):
    pass
    return None 


# -----------/// read from volume


def read_from_databricks(file_name:str):
    pass
    return None


# --------/// list files that exist in volume

def list_files_in_volume():
    pass
    return None