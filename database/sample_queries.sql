-- Sample queries for Clinical Drug Dosing Decision Support System

-- 1. Retrieve all drugs
SELECT DISTINCT generic_name
FROM drug_reference;

-- 2. Retrieve indications for a specific drug
SELECT indication
FROM drug_reference
WHERE generic_name = 'Ampicillin';

-- 3. Retrieve dosage reference for a drug and indication
SELECT dosage, renal_adjustment, renal_adjustment_dose
FROM drug_reference
WHERE generic_name = 'Ampicillin'
  AND indication = 'Endocarditis';
