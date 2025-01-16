from openhouse_connector import OpenHouse
from openhouse_connector import TrinoConnector
from openhouse_config import connection_params

trino_connector = TrinoConnector(connection_params)
openhouse = OpenHouse(trino_connector)
df = openhouse.table("openhouse.u_openhouse.lotus_test")
print(df.head())  # Now connects and fetches data
