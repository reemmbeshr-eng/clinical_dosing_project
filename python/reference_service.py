from db_connection import get_connection

def get_drug_reference(drug_name, indication):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT
        dosage,
        renal_adjustment,
        renal_adjustment_dose,
        administration,
        preparation_for_administration
    FROM drug_reference
    WHERE LOWER(generic_name) = LOWER(%s)
    AND LOWER(indication) LIKE LOWER(%s);

    """


    cursor.execute(
    query,
    (drug_name, f"%{indication}%")
)

    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if not result:
        return None

    return {
        "dosage": result[0],
        "renal_adjustment": result[1],
        "renal_adjustment_dose": result[2],
        "administration": result[3],
        "preparation": result[4],
    }
