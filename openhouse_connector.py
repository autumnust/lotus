from typing import Optional, Any, List
import pandas as pd
from abc import ABC, abstractmethod
from lotus.models.iceberg_rm import IcebergRM
import trino
from lotus.models.lm import LM
from lotus.models.litellm_rm import LiteLLMRM
import lotus
from lotus.models import CrossEncoderReranker


class DataConnector(ABC):
    @abstractmethod
    def get_table(self, table_name: str) -> Any:
        """Returns a table object that supports common DataFrame operations"""
        pass

class TrinoConnector(DataConnector):
    def __init__(self, connection_params: dict):
        self._connection_params = connection_params
        self._connection = None

    def _ensure_connection(self):
        if self._connection is None:
            self._connection = trino.dbapi.connect(**self._connection_params)

    def get_table(self, table_name: str) -> 'LazyTrinoTable':
        return LazyTrinoTable(table_name, self)

    def execute_query(self, query: str) -> pd.DataFrame:
        self._ensure_connection()
        cur = self._connection.cursor()
        cur.execute(query)
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)

class SparkConnector(DataConnector):
    def __init__(self, spark_session):
        self._spark = spark_session

    def get_table(self, table_name: str) -> 'LazySparkTable':
        return LazySparkTable(table_name, self._spark)

class LazyTrinoTable:
    def __init__(self, table_name: str, connector: TrinoConnector):
        self._table_name = table_name
        self._connector = connector
        self._query = f"SELECT * FROM {table_name}"
        self._df: Optional[pd.DataFrame] = None

    def _ensure_data(self):
        if self._df is None:
            self._df = self._connector.execute_query(self._query)
        return self._df

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._ensure_data().head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        return self._ensure_data().tail(n)
    
    def count(self) -> int:
        return self._ensure_data().shape[0]
    
    def describe(self) -> pd.DataFrame:
        return self._ensure_data().describe()

    def __getattr__(self, name: str) -> Any:
        def method(*args, **kwargs):
            df = self._ensure_data()
            if hasattr(df, name):
                return getattr(df, name)(*args, **kwargs)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return method

class LazySparkTable:
    def __init__(self, table_name: str, spark):
        self.index_col = ""
        self._spark = spark
        self._df = spark.table(table_name)
        # reconstruct the location for index dir
        self._table_name = table_name

    def head(self, n: int = 5) -> pd.DataFrame:
        return self._df.limit(n).toPandas()

    def tail(self, n: int = 5) -> pd.DataFrame:
        # Note: tail in Spark is expensive, might want to warn users
        return self._df.tail(n)

    # def read(self) -> pd.DataFrame:
    #     self._df.toPandas()
    
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
            index_cols: optional list of column names to be used for indexing
            mode: write mode ('overwrite', 'append', 'ignore', or 'error')
        """
        spark_df = self._spark.createDataFrame(df)
        spark_df.write.mode(mode).insertInto(self._table_name)

        # Only attempt to update index if index_cols are provided
        if index_cols:
            self.index_col = index_cols[0]
            val = self._spark.sql(f"""CALL system.compute_table_embeddings(
                             table => '{self._table_name}', 
                             model_name => 'ollama/llama3.1', 
                             model_inputs => map('x', 'y'), columns => array('{ ','.join(index_cols)}'))""")
            df = df.sem_index(f"{index_cols[0]}", f"{self._table_name}.{index_cols[0]}", static_rm=IcebergRM(spark=self._spark))
            return df

class OpenHouse:
    def __init__(self, connector: DataConnector):
        self._connector = connector
        lotus.settings.configure(lm = LM(model="ollama/llama3.1"), rm = LiteLLMRM(model="ollama/llama3.1"), reranker=CrossEncoderReranker())

    def table(self, table_name: str) -> Any:
        return self._connector.get_table(table_name)
