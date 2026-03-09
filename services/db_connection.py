import psycopg2
import pandas as pd

def get_connection():
    return psycopg2.connect(
        dbname="clinical_dosing_db",
        user="postgres",
        password="123456",
        host="localhost",
        port="5432"
    )


conn = get_connection()
from sqlalchemy import create_engine

# إنشاء الاتصال
engine = create_engine(
    "postgresql://postgres:123456@localhost:5432/clinical_dosing_db"
)

query = "SELECT * FROM drug_reference"

# قراءة الداتا
df = pd.read_sql(query, engine)

# حفظ CSV
df.to_csv("drugs.csv", index=False)

print("CSV file created successfully")