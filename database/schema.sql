-- Database schema for Clinical Drug Dosing Decision Support System

CREATE TABLE drug_reference (
    generic_name TEXT,
    brand_name TEXT,
    drug_class TEXT,
    indication TEXT,
    dosage TEXT,
    renal_adjustment TEXT,
    renal_adjustment_dose TEXT,
    administration TEXT,
    preparation_for_administration TEXT,
    storage_stability TEXT
);
