from typing import Optional, Any, List
import pandas as pd
from abc import ABC, abstractmethod
from lotus.models.iceberg_rm import IcebergRM
import trino
from lotus.models.lm import LM
from lotus.models.litellm_rm import LiteLLMRM
import lotus
from lotus.models import CrossEncoderReranker
from pyspark.sql.functions import col


class DataConnector(ABC):
    @abstractmethod
    def get_table(self, table_name: str) -> Any:
        """Returns a table object that supports common DataFrame operations"""
        pass

class SparkConnector(DataConnector):
    def __init__(self, spark_session):
        self._spark = spark_session

    def get_table(self, table_name: str) -> 'LazySparkTable':
        return LazySparkTable(table_name, self._spark)

class LazySparkTable:
    from pyspark.sql.functions import col
    def __init__(self, table_name: str, spark):
        self.index_cols = None
        self._spark = spark
        self._df = spark.table(table_name)
        self._table_name = table_name

    def _decorate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper function to decorate DataFrame with index information, read method should call this before returning"""
        df.attrs["index_dirs"] = {}

        assert self.index_cols is not None, "index_cols must be set before calling read API, check write API"
        if not hasattr(self, 'index_dirs'):
            tblprops = self._spark.sql(f"SHOW TBLPROPERTIES {self._table_name}").toPandas()
            for col in self.index_cols:
                df.attrs["index_dirs"][col] = tblprops.loc[tblprops['key'] == f'index_dir_{col}', 'value'].values[0]
            self.index_dirs = df.attrs["index_dirs"]
        else:
            df.attrs["index_dirs"][col] = self.index_dirs[col]
        return df

    # add load method, essetially read the table into a pandas dataframe
    def load(self) -> pd.DataFrame:
        df = self._df.toPandas()
        return self._decorate_dataframe(df)
    
    def head(self, n: int = 5) -> pd.DataFrame:
        df = self._df.limit(n).toPandas()
        return self._decorate_dataframe(df)

    def tail(self, n: int = 5) -> pd.DataFrame:
        # Note: tail in Spark is expensive, might want to warn users
        return self._df.tail(n)
    
    def describe(self) -> pd.DataFrame:
        return self._df.describe().toPandas()

    def __getattr__(self, name: str) -> Any:
        if hasattr(self._df, name):
            return getattr(self._df, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def write(self, df: pd.DataFrame,  index_cols: Optional[List[str]] = None, mode: str = "append") -> pd.DataFrame:
        """
        Write a pandas DataFrame to a Spark table.
        
        Args:
            df: pandas DataFrame to write
            table_name: name of the target table
            index_cols: list of column names to be used for indexing
            mode: write mode ('overwrite', 'append', 'ignore', or 'error')
        """
        spark_df = self._spark.createDataFrame(df)
        spark_df.write.mode(mode).insertInto(self._table_name)

        if index_cols:
            self.index_cols = index_cols
            val = self._spark.sql(f"""CALL system.compute_table_embeddings(
                             table => '{self._table_name}', 
                             model_name => 'ollama/llama3.1', 
                             model_inputs => map('x', 'y'), 
                             columns => array({','.join([f"'{col}'" for col in index_cols])}))""")            
            for col in index_cols:
                df.sem_index(f"{col}", f"{self._table_name}.{col}", static_rm=IcebergRM(spark=self._spark))
                self._spark.sql(f"ALTER TABLE {self._table_name} SET TBLPROPERTIES ('index_dir_{col}' = '{self._table_name}.{col}')")

class OpenHouse:
    def __init__(self, connector: DataConnector):
        self._connector = connector
        lotus.settings.configure(lm = LM(model="ollama/llama3.1"), rm = LiteLLMRM(model="ollama/llama3.1"), reranker=CrossEncoderReranker())

    def table(self, table_name: str) -> Any:
        return self._connector.get_table(table_name)
