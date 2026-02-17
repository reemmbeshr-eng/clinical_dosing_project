import psycopg2

def get_connection():
    return psycopg2.connect(
        dbname="clinical_dosing_db",
        user="postgres",
        password="123456",
        host="localhost",
        port="5432"
    )
