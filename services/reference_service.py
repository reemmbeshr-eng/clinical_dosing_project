from .db_connection import get_connection

## Get drug reference information from the database :
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

    rows = cursor.fetchall()   

    cursor.close()
    conn.close()

    if not rows:
        return None

    
    renal_doses = [
        row[2] for row in rows
        if row[2] is not None and row[2].strip() != ""
    ]

    return {
        "dosage": rows[0][0],
        "renal_adjustment": rows[0][1],
        "renal_adjustment_dose": "\n".join(renal_doses),  # ✅ كلهم
        "administration": rows[0][3],
        "preparation": rows[0][4],
    }