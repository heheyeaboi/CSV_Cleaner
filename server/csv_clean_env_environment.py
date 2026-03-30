from openenv.core import Environment, State
import pandas as pd
import numpy as np
import io
from models import CsvCleanAction, CsvCleanObservation


class CsvCleanEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    MAX_STEPS: int = 20

    # ── Dataset 1: Products (20 rows) ────────────────────────────────────
    DIRTY_1 = [
        {"id": 1, "product": "  Widget A ", "price": "12.99", "quantity": 100},
        {"id": 2, "product": "Gadget B  ", "price": "24.50", "quantity": 200},
        {"id": 3, "product": " Gizmo C", "price": "N/A", "quantity": 150},
        {"id": 4, "product": "Doohickey D ", "price": "8.75", "quantity": None},
        {"id": 5, "product": "  Thingamajig E", "price": "15.00", "quantity": 80},
        {"id": 6, "product": "Widget F  ", "price": "33.20", "quantity": 60},
        {"id": 7, "product": " Gadget G ", "price": None, "quantity": 45},
        {"id": 8, "product": "Gizmo H", "price": "19.99", "quantity": 300},
        {"id": 9, "product": "  Doohickey I ", "price": "7.50", "quantity": 90},
        {"id": 10, "product": "Thingamajig J", "price": "42.00", "quantity": None},
        {"id": 11, "product": " Widget K", "price": "11.25", "quantity": 110},
        {"id": 12, "product": "Gadget L  ", "price": "N/A", "quantity": 70},
        {"id": 13, "product": "  Gizmo M ", "price": "29.99", "quantity": 55},
        {"id": 14, "product": "Doohickey N", "price": "16.80", "quantity": 140},
        {"id": 15, "product": " Thingamajig O ", "price": "5.99", "quantity": 200},
        {"id": 16, "product": "Widget P", "price": "22.10", "quantity": 85},
        {"id": 17, "product": "  Gadget Q", "price": "N/A", "quantity": None},
        {"id": 18, "product": "Gizmo R ", "price": "38.50", "quantity": 175},
        {"id": 19, "product": " Doohickey S ", "price": "14.00", "quantity": 65},
        {"id": 20, "product": "Thingamajig T  ", "price": "9.25", "quantity": 120},
    ]

    CLEAN_1 = [
        {"id": 1, "product": "Widget A", "price": 12.99, "quantity": 100},
        {"id": 2, "product": "Gadget B", "price": 24.50, "quantity": 200},
        {"id": 4, "product": "Doohickey D", "price": 8.75, "quantity": 0},
        {"id": 5, "product": "Thingamajig E", "price": 15.00, "quantity": 80},
        {"id": 6, "product": "Widget F", "price": 33.20, "quantity": 60},
        {"id": 8, "product": "Gizmo H", "price": 19.99, "quantity": 300},
        {"id": 9, "product": "Doohickey I", "price": 7.50, "quantity": 90},
        {"id": 10, "product": "Thingamajig J", "price": 42.00, "quantity": 0},
        {"id": 11, "product": "Widget K", "price": 11.25, "quantity": 110},
        {"id": 13, "product": "Gizmo M", "price": 29.99, "quantity": 55},
        {"id": 14, "product": "Doohickey N", "price": 16.80, "quantity": 140},
        {"id": 15, "product": "Thingamajig O", "price": 5.99, "quantity": 200},
        {"id": 16, "product": "Widget P", "price": 22.10, "quantity": 85},
        {"id": 18, "product": "Gizmo R", "price": 38.50, "quantity": 175},
        {"id": 19, "product": "Doohickey S", "price": 14.00, "quantity": 65},
        {"id": 20, "product": "Thingamajig T", "price": 9.25, "quantity": 120},
    ]

    # ── Dataset 2: Employees (30 rows) ───────────────────────────────────
    DIRTY_2 = [
        {"emp_id": 101, "name": "Alice Johnson", "Department": "Engineering", "salary": 95000, "join_date": "2021-03-15"},
        {"emp_id": 102, "name": "Bob Smith", "Department": "ENGINEERING", "salary": 88000, "join_date": "2020-07-01"},
        {"emp_id": 103, "name": "Carol White", "Department": "engineering", "salary": None, "join_date": "2022-01-10"},
        {"emp_id": 104, "name": "David Brown", "Department": "Marketing", "salary": 72000, "join_date": "2019-11-22"},
        {"emp_id": 105, "name": "Eve Davis", "Department": "MARKETING", "salary": 68000, "join_date": "2021-06-30"},
        {"emp_id": 106, "name": "Frank Miller", "Department": "Engineering", "salary": 102000, "join_date": "2018-04-05"},
        {"emp_id": 107, "name": "Grace Wilson", "Department": "HR", "salary": None, "join_date": "2023-02-14"},
        {"emp_id": 108, "name": "Hank Moore", "Department": "hr", "salary": 61000, "join_date": "2020-09-18"},
        {"emp_id": 109, "name": "Ivy Taylor", "Department": "Engineering", "salary": 97000, "join_date": "2021-08-25"},
        {"emp_id": 110, "name": "Jack Anderson", "Department": "MARKETING", "salary": 75000, "join_date": "2022-05-12"},
        {"emp_id": 111, "name": "Karen Thomas", "Department": "engineering", "salary": 91000, "join_date": "2019-12-01"},
        {"emp_id": 112, "name": "Leo Jackson", "Department": "HR", "salary": 58000, "join_date": "2023-07-20"},
        {"emp_id": 113, "name": "Mona Harris", "Department": "Marketing", "salary": None, "join_date": "2020-03-09"},
        {"emp_id": 114, "name": "Nate Clark", "Department": "ENGINEERING", "salary": 105000, "join_date": "2017-10-30"},
        {"emp_id": 115, "name": "Olivia Lewis", "Department": "hr", "salary": 63000, "join_date": "2022-11-15"},
        {"emp_id": 116, "name": "Paul Robinson", "Department": "Engineering", "salary": 99000, "join_date": "2021-01-07"},
        {"emp_id": 117, "name": "Quinn Walker", "Department": "Marketing", "salary": 71000, "join_date": "2020-06-14"},
        {"emp_id": 118, "name": "Rosa Hall", "Department": "ENGINEERING", "salary": None, "join_date": "2023-04-21"},
        {"emp_id": 119, "name": "Sam Allen", "Department": "HR", "salary": 59000, "join_date": "2019-08-03"},
        {"emp_id": 120, "name": "Tina Young", "Department": "engineering", "salary": 93000, "join_date": "2022-02-28"},
        {"emp_id": 121, "name": "Uma King", "Department": "Marketing", "salary": 74000, "join_date": "2021-09-10"},
        {"emp_id": 122, "name": "Vince Wright", "Department": "HR", "salary": 62000, "join_date": "2020-12-05"},
        {"emp_id": 123, "name": "Wendy Scott", "Department": "ENGINEERING", "salary": 98000, "join_date": "2018-07-19"},
        {"emp_id": 124, "name": "Xavier Green", "Department": "Marketing", "salary": 70000, "join_date": "2023-01-25"},
        {"emp_id": 125, "name": "Yara Adams", "Department": "hr", "salary": 60000, "join_date": "2021-05-16"},
        {"emp_id": 126, "name": "Zane Baker", "Department": "Engineering", "salary": 100000, "join_date": "2019-03-08"},
        # Duplicate rows (4 exact duplicates of earlier rows)
        {"emp_id": 101, "name": "Alice Johnson", "Department": "Engineering", "salary": 95000, "join_date": "2021-03-15"},
        {"emp_id": 105, "name": "Eve Davis", "Department": "MARKETING", "salary": 68000, "join_date": "2021-06-30"},
        {"emp_id": 109, "name": "Ivy Taylor", "Department": "Engineering", "salary": 97000, "join_date": "2021-08-25"},
        {"emp_id": 114, "name": "Nate Clark", "Department": "ENGINEERING", "salary": 105000, "join_date": "2017-10-30"},
    ]

    CLEAN_2 = [
        {"emp_id": 101, "name": "Alice Johnson", "department": "engineering", "salary": 95000, "join_date": pd.Timestamp("2021-03-15")},
        {"emp_id": 102, "name": "Bob Smith", "department": "engineering", "salary": 88000, "join_date": pd.Timestamp("2020-07-01")},
        {"emp_id": 103, "name": "Carol White", "department": "engineering", "salary": 0, "join_date": pd.Timestamp("2022-01-10")},
        {"emp_id": 104, "name": "David Brown", "department": "marketing", "salary": 72000, "join_date": pd.Timestamp("2019-11-22")},
        {"emp_id": 105, "name": "Eve Davis", "department": "marketing", "salary": 68000, "join_date": pd.Timestamp("2021-06-30")},
        {"emp_id": 106, "name": "Frank Miller", "department": "engineering", "salary": 102000, "join_date": pd.Timestamp("2018-04-05")},
        {"emp_id": 107, "name": "Grace Wilson", "department": "hr", "salary": 0, "join_date": pd.Timestamp("2023-02-14")},
        {"emp_id": 108, "name": "Hank Moore", "department": "hr", "salary": 61000, "join_date": pd.Timestamp("2020-09-18")},
        {"emp_id": 109, "name": "Ivy Taylor", "department": "engineering", "salary": 97000, "join_date": pd.Timestamp("2021-08-25")},
        {"emp_id": 110, "name": "Jack Anderson", "department": "marketing", "salary": 75000, "join_date": pd.Timestamp("2022-05-12")},
        {"emp_id": 111, "name": "Karen Thomas", "department": "engineering", "salary": 91000, "join_date": pd.Timestamp("2019-12-01")},
        {"emp_id": 112, "name": "Leo Jackson", "department": "hr", "salary": 58000, "join_date": pd.Timestamp("2023-07-20")},
        {"emp_id": 113, "name": "Mona Harris", "department": "marketing", "salary": 0, "join_date": pd.Timestamp("2020-03-09")},
        {"emp_id": 114, "name": "Nate Clark", "department": "engineering", "salary": 105000, "join_date": pd.Timestamp("2017-10-30")},
        {"emp_id": 115, "name": "Olivia Lewis", "department": "hr", "salary": 63000, "join_date": pd.Timestamp("2022-11-15")},
        {"emp_id": 116, "name": "Paul Robinson", "department": "engineering", "salary": 99000, "join_date": pd.Timestamp("2021-01-07")},
        {"emp_id": 117, "name": "Quinn Walker", "department": "marketing", "salary": 71000, "join_date": pd.Timestamp("2020-06-14")},
        {"emp_id": 118, "name": "Rosa Hall", "department": "engineering", "salary": 0, "join_date": pd.Timestamp("2023-04-21")},
        {"emp_id": 119, "name": "Sam Allen", "department": "hr", "salary": 59000, "join_date": pd.Timestamp("2019-08-03")},
        {"emp_id": 120, "name": "Tina Young", "department": "engineering", "salary": 93000, "join_date": pd.Timestamp("2022-02-28")},
        {"emp_id": 121, "name": "Uma King", "department": "marketing", "salary": 74000, "join_date": pd.Timestamp("2021-09-10")},
        {"emp_id": 122, "name": "Vince Wright", "department": "hr", "salary": 62000, "join_date": pd.Timestamp("2020-12-05")},
        {"emp_id": 123, "name": "Wendy Scott", "department": "engineering", "salary": 98000, "join_date": pd.Timestamp("2018-07-19")},
        {"emp_id": 124, "name": "Xavier Green", "department": "marketing", "salary": 70000, "join_date": pd.Timestamp("2023-01-25")},
        {"emp_id": 125, "name": "Yara Adams", "department": "hr", "salary": 60000, "join_date": pd.Timestamp("2021-05-16")},
        {"emp_id": 126, "name": "Zane Baker", "department": "engineering", "salary": 100000, "join_date": pd.Timestamp("2019-03-08")},
    ]

    # ── Dataset 3: Patients (40 rows) ────────────────────────────────────
    DIRTY_3 = [
        {"patient_id": 1001, "name": " John Doe ", "age": "45", "gender": "Male", "blood_pressure": "120/80 ", "diagnosis": "Hypertension", "notes": None, "admission_date": "2023-01-05", "discharge_date": "2023-01-12", "insurance_code": "INS-001"},
        {"patient_id": 1002, "name": "Jane Smith", "age": "34", "gender": "female", "blood_pressure": " 110/70", "diagnosis": "Diabetes", "notes": None, "admission_date": "2023-02-10", "discharge_date": "2023-02-18", "insurance_code": None},
        {"patient_id": 1003, "name": "  Bob Williams ", "age": "unknown", "gender": "MALE", "blood_pressure": "130/85", "diagnosis": "Asthma", "notes": None, "admission_date": "2023-01-20", "discharge_date": "2023-01-25", "insurance_code": "INS-003"},
        {"patient_id": 1004, "name": "Alice Brown", "age": "58", "gender": "Female", "blood_pressure": "140/90 ", "diagnosis": "Arthritis", "notes": None, "admission_date": "2023-03-01", "discharge_date": "2023-03-10", "insurance_code": "INS-004"},
        {"patient_id": 1005, "name": " Charlie Davis ", "age": "29", "gender": "male", "blood_pressure": " 118/76 ", "diagnosis": "Migraine", "notes": None, "admission_date": "2023-02-15", "discharge_date": "2023-02-17", "insurance_code": None},
        {"patient_id": 1006, "name": "Diana Evans", "age": "67", "gender": "FEMALE", "blood_pressure": "150/95", "diagnosis": "Heart Disease", "notes": None, "admission_date": "2023-04-02", "discharge_date": "2023-04-15", "insurance_code": "INS-006"},
        {"patient_id": 1007, "name": "  Edward Foster ", "age": "unknown", "gender": "Male", "blood_pressure": "125/82 ", "diagnosis": "Bronchitis", "notes": None, "admission_date": "2023-03-18", "discharge_date": "2023-03-22", "insurance_code": "INS-007"},
        {"patient_id": 1008, "name": "Fiona Garcia", "age": "41", "gender": "female", "blood_pressure": " 115/72", "diagnosis": "Anemia", "notes": None, "admission_date": "2023-05-10", "discharge_date": "2023-05-14", "insurance_code": None},
        {"patient_id": 1009, "name": " George Harris ", "age": "53", "gender": "MALE", "blood_pressure": "135/88", "diagnosis": "Diabetes", "notes": None, "admission_date": "2023-04-25", "discharge_date": "2023-05-02", "insurance_code": "INS-009"},
        {"patient_id": 1010, "name": "Helen Irving  ", "age": "36", "gender": "Female", "blood_pressure": " 112/68 ", "diagnosis": "Thyroid", "notes": None, "admission_date": "2023-06-01", "discharge_date": "2023-06-05", "insurance_code": "INS-010"},
        {"patient_id": 1011, "name": "Ian Jackson", "age": "72", "gender": "male", "blood_pressure": "155/100", "diagnosis": "Hypertension", "notes": None, "admission_date": "2023-05-20", "discharge_date": "2023-06-01", "insurance_code": None},
        {"patient_id": 1012, "name": "  Julia Kim ", "age": "48", "gender": "FEMALE", "blood_pressure": "128/84 ", "diagnosis": "Asthma", "notes": None, "admission_date": "2023-07-08", "discharge_date": "2023-07-12", "insurance_code": "INS-012"},
        {"patient_id": 1013, "name": " Kevin Lee", "age": "unknown", "gender": "Male", "blood_pressure": " 122/78", "diagnosis": "Back Pain", "notes": None, "admission_date": "2023-06-15", "discharge_date": "2023-06-18", "insurance_code": "INS-013"},
        {"patient_id": 1014, "name": "Laura Martin ", "age": "55", "gender": "female", "blood_pressure": "138/86", "diagnosis": "Arthritis", "notes": None, "admission_date": "2023-08-01", "discharge_date": "2023-08-09", "insurance_code": None},
        {"patient_id": 1015, "name": "  Mike Nelson ", "age": "31", "gender": "MALE", "blood_pressure": " 116/74 ", "diagnosis": "Allergies", "notes": None, "admission_date": "2023-07-22", "discharge_date": "2023-07-24", "insurance_code": "INS-015"},
        {"patient_id": 1016, "name": "Nina Ortiz", "age": "63", "gender": "Female", "blood_pressure": "145/92", "diagnosis": "Heart Disease", "notes": None, "admission_date": "2023-09-05", "discharge_date": "2023-09-18", "insurance_code": "INS-016"},
        {"patient_id": 1017, "name": " Oscar Perez ", "age": "39", "gender": "male", "blood_pressure": "120/80 ", "diagnosis": "Migraine", "notes": None, "admission_date": "2023-08-12", "discharge_date": "2023-08-14", "insurance_code": None},
        {"patient_id": 1018, "name": "Paula Quinn  ", "age": "50", "gender": "FEMALE", "blood_pressure": " 132/86", "diagnosis": "Diabetes", "notes": None, "admission_date": "2023-10-01", "discharge_date": "2023-10-08", "insurance_code": "INS-018"},
        {"patient_id": 1019, "name": "  Ralph Reed", "age": "44", "gender": "Male", "blood_pressure": "126/82", "diagnosis": "Bronchitis", "notes": None, "admission_date": "2023-09-20", "discharge_date": "2023-09-24", "insurance_code": "INS-019"},
        {"patient_id": 1020, "name": "Sara Stone ", "age": "unknown", "gender": "female", "blood_pressure": " 108/66 ", "diagnosis": "Anemia", "notes": None, "admission_date": "2023-11-10", "discharge_date": "2023-11-14", "insurance_code": None},
        {"patient_id": 1021, "name": " Tom Underwood ", "age": "60", "gender": "MALE", "blood_pressure": "148/94", "diagnosis": "Hypertension", "notes": None, "admission_date": "2023-10-15", "discharge_date": "2023-10-28", "insurance_code": "INS-021"},
        {"patient_id": 1022, "name": "Uma Vargas", "age": "33", "gender": "Female", "blood_pressure": "114/72 ", "diagnosis": "Thyroid", "notes": None, "admission_date": "2023-12-01", "discharge_date": "2023-12-04", "insurance_code": "INS-022"},
        {"patient_id": 1023, "name": "  Victor Walsh ", "age": "57", "gender": "male", "blood_pressure": " 136/88", "diagnosis": "Arthritis", "notes": None, "admission_date": "2023-11-20", "discharge_date": "2023-11-28", "insurance_code": "INS-023"},
        {"patient_id": 1024, "name": " Wanda Xu", "age": "42", "gender": "FEMALE", "blood_pressure": "124/80", "diagnosis": "Allergies", "notes": None, "admission_date": "2024-01-05", "discharge_date": "2024-01-07", "insurance_code": None},
        {"patient_id": 1025, "name": "Xander Young  ", "age": "unknown", "gender": "Male", "blood_pressure": " 118/76 ", "diagnosis": "Back Pain", "notes": None, "admission_date": "2023-12-18", "discharge_date": "2023-12-20", "insurance_code": "INS-025"},
        {"patient_id": 1026, "name": " Yolanda Zimmerman ", "age": "70", "gender": "female", "blood_pressure": "152/96", "diagnosis": "Heart Disease", "notes": None, "admission_date": "2024-01-15", "discharge_date": "2024-01-30", "insurance_code": "INS-026"},
        {"patient_id": 1027, "name": "Aaron Bennett", "age": "38", "gender": "MALE", "blood_pressure": "122/78 ", "diagnosis": "Migraine", "notes": None, "admission_date": "2024-02-01", "discharge_date": "2024-02-03", "insurance_code": "INS-027"},
        {"patient_id": 1028, "name": "  Brenda Cole ", "age": "52", "gender": "Female", "blood_pressure": " 134/86", "diagnosis": "Diabetes", "notes": None, "admission_date": "2024-02-10", "discharge_date": "2024-02-17", "insurance_code": "INS-028"},
        {"patient_id": 1029, "name": " Carl Dixon", "age": "46", "gender": "male", "blood_pressure": "128/82", "diagnosis": "Bronchitis", "notes": None, "admission_date": "2024-03-01", "discharge_date": "2024-03-05", "insurance_code": "INS-029"},
        {"patient_id": 1030, "name": "Debra Ellis  ", "age": "61", "gender": "FEMALE", "blood_pressure": " 142/90 ", "diagnosis": "Hypertension", "notes": None, "admission_date": "2024-03-10", "discharge_date": "2024-03-22", "insurance_code": "INS-030"},
        {"patient_id": 1031, "name": " Eric Flores ", "age": "27", "gender": "Male", "blood_pressure": "110/70", "diagnosis": "Allergies", "notes": None, "admission_date": "2024-04-01", "discharge_date": "2024-04-02", "insurance_code": "INS-031"},
        {"patient_id": 1032, "name": "Faye Grant", "age": "49", "gender": "female", "blood_pressure": "130/84 ", "diagnosis": "Anemia", "notes": None, "admission_date": "2024-04-15", "discharge_date": "2024-04-19", "insurance_code": "INS-032"},
        {"patient_id": 1033, "name": "  Greg Hayes ", "age": "56", "gender": "MALE", "blood_pressure": " 140/88", "diagnosis": "Arthritis", "notes": None, "admission_date": "2024-05-01", "discharge_date": "2024-05-08", "insurance_code": "INS-033"},
        {"patient_id": 1034, "name": " Holly Ingram", "age": "35", "gender": "Female", "blood_pressure": "116/74", "diagnosis": "Thyroid", "notes": None, "admission_date": "2024-05-20", "discharge_date": "2024-05-23", "insurance_code": "INS-034"},
        {"patient_id": 1035, "name": "Ivan Jensen  ", "age": "68", "gender": "male", "blood_pressure": " 150/96 ", "diagnosis": "Heart Disease", "notes": None, "admission_date": "2024-06-01", "discharge_date": "2024-06-14", "insurance_code": "INS-035"},
        # Duplicate rows (5 exact duplicates of earlier rows)
        {"patient_id": 1001, "name": " John Doe ", "age": "45", "gender": "Male", "blood_pressure": "120/80 ", "diagnosis": "Hypertension", "notes": None, "admission_date": "2023-01-05", "discharge_date": "2023-01-12", "insurance_code": "INS-001"},
        {"patient_id": 1006, "name": "Diana Evans", "age": "67", "gender": "FEMALE", "blood_pressure": "150/95", "diagnosis": "Heart Disease", "notes": None, "admission_date": "2023-04-02", "discharge_date": "2023-04-15", "insurance_code": "INS-006"},
        {"patient_id": 1011, "name": "Ian Jackson", "age": "72", "gender": "male", "blood_pressure": "155/100", "diagnosis": "Hypertension", "notes": None, "admission_date": "2023-05-20", "discharge_date": "2023-06-01", "insurance_code": None},
        {"patient_id": 1021, "name": " Tom Underwood ", "age": "60", "gender": "MALE", "blood_pressure": "148/94", "diagnosis": "Hypertension", "notes": None, "admission_date": "2023-10-15", "discharge_date": "2023-10-28", "insurance_code": "INS-021"},
        {"patient_id": 1028, "name": "  Brenda Cole ", "age": "52", "gender": "Female", "blood_pressure": " 134/86", "diagnosis": "Diabetes", "notes": None, "admission_date": "2024-02-10", "discharge_date": "2024-02-17", "insurance_code": "INS-028"},
    ]

    CLEAN_3 = [
        {"patient_id": 1001, "name": "John Doe", "age": 45.0, "gender": "male", "blood_pressure": "120/80", "diagnosis": "Hypertension", "admission_date": pd.Timestamp("2023-01-05"), "discharge_date": pd.Timestamp("2023-01-12"), "insurance_code": "INS-001"},
        {"patient_id": 1002, "name": "Jane Smith", "age": 34.0, "gender": "female", "blood_pressure": "110/70", "diagnosis": "Diabetes", "admission_date": pd.Timestamp("2023-02-10"), "discharge_date": pd.Timestamp("2023-02-18"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1003, "name": "Bob Williams", "age": None, "gender": "male", "blood_pressure": "130/85", "diagnosis": "Asthma", "admission_date": pd.Timestamp("2023-01-20"), "discharge_date": pd.Timestamp("2023-01-25"), "insurance_code": "INS-003"},
        {"patient_id": 1004, "name": "Alice Brown", "age": 58.0, "gender": "female", "blood_pressure": "140/90", "diagnosis": "Arthritis", "admission_date": pd.Timestamp("2023-03-01"), "discharge_date": pd.Timestamp("2023-03-10"), "insurance_code": "INS-004"},
        {"patient_id": 1005, "name": "Charlie Davis", "age": 29.0, "gender": "male", "blood_pressure": "118/76", "diagnosis": "Migraine", "admission_date": pd.Timestamp("2023-02-15"), "discharge_date": pd.Timestamp("2023-02-17"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1006, "name": "Diana Evans", "age": 67.0, "gender": "female", "blood_pressure": "150/95", "diagnosis": "Heart Disease", "admission_date": pd.Timestamp("2023-04-02"), "discharge_date": pd.Timestamp("2023-04-15"), "insurance_code": "INS-006"},
        {"patient_id": 1007, "name": "Edward Foster", "age": None, "gender": "male", "blood_pressure": "125/82", "diagnosis": "Bronchitis", "admission_date": pd.Timestamp("2023-03-18"), "discharge_date": pd.Timestamp("2023-03-22"), "insurance_code": "INS-007"},
        {"patient_id": 1008, "name": "Fiona Garcia", "age": 41.0, "gender": "female", "blood_pressure": "115/72", "diagnosis": "Anemia", "admission_date": pd.Timestamp("2023-05-10"), "discharge_date": pd.Timestamp("2023-05-14"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1009, "name": "George Harris", "age": 53.0, "gender": "male", "blood_pressure": "135/88", "diagnosis": "Diabetes", "admission_date": pd.Timestamp("2023-04-25"), "discharge_date": pd.Timestamp("2023-05-02"), "insurance_code": "INS-009"},
        {"patient_id": 1010, "name": "Helen Irving", "age": 36.0, "gender": "female", "blood_pressure": "112/68", "diagnosis": "Thyroid", "admission_date": pd.Timestamp("2023-06-01"), "discharge_date": pd.Timestamp("2023-06-05"), "insurance_code": "INS-010"},
        {"patient_id": 1011, "name": "Ian Jackson", "age": 72.0, "gender": "male", "blood_pressure": "155/100", "diagnosis": "Hypertension", "admission_date": pd.Timestamp("2023-05-20"), "discharge_date": pd.Timestamp("2023-06-01"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1012, "name": "Julia Kim", "age": 48.0, "gender": "female", "blood_pressure": "128/84", "diagnosis": "Asthma", "admission_date": pd.Timestamp("2023-07-08"), "discharge_date": pd.Timestamp("2023-07-12"), "insurance_code": "INS-012"},
        {"patient_id": 1013, "name": "Kevin Lee", "age": None, "gender": "male", "blood_pressure": "122/78", "diagnosis": "Back Pain", "admission_date": pd.Timestamp("2023-06-15"), "discharge_date": pd.Timestamp("2023-06-18"), "insurance_code": "INS-013"},
        {"patient_id": 1014, "name": "Laura Martin", "age": 55.0, "gender": "female", "blood_pressure": "138/86", "diagnosis": "Arthritis", "admission_date": pd.Timestamp("2023-08-01"), "discharge_date": pd.Timestamp("2023-08-09"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1015, "name": "Mike Nelson", "age": 31.0, "gender": "male", "blood_pressure": "116/74", "diagnosis": "Allergies", "admission_date": pd.Timestamp("2023-07-22"), "discharge_date": pd.Timestamp("2023-07-24"), "insurance_code": "INS-015"},
        {"patient_id": 1016, "name": "Nina Ortiz", "age": 63.0, "gender": "female", "blood_pressure": "145/92", "diagnosis": "Heart Disease", "admission_date": pd.Timestamp("2023-09-05"), "discharge_date": pd.Timestamp("2023-09-18"), "insurance_code": "INS-016"},
        {"patient_id": 1017, "name": "Oscar Perez", "age": 39.0, "gender": "male", "blood_pressure": "120/80", "diagnosis": "Migraine", "admission_date": pd.Timestamp("2023-08-12"), "discharge_date": pd.Timestamp("2023-08-14"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1018, "name": "Paula Quinn", "age": 50.0, "gender": "female", "blood_pressure": "132/86", "diagnosis": "Diabetes", "admission_date": pd.Timestamp("2023-10-01"), "discharge_date": pd.Timestamp("2023-10-08"), "insurance_code": "INS-018"},
        {"patient_id": 1019, "name": "Ralph Reed", "age": 44.0, "gender": "male", "blood_pressure": "126/82", "diagnosis": "Bronchitis", "admission_date": pd.Timestamp("2023-09-20"), "discharge_date": pd.Timestamp("2023-09-24"), "insurance_code": "INS-019"},
        {"patient_id": 1020, "name": "Sara Stone", "age": None, "gender": "female", "blood_pressure": "108/66", "diagnosis": "Anemia", "admission_date": pd.Timestamp("2023-11-10"), "discharge_date": pd.Timestamp("2023-11-14"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1021, "name": "Tom Underwood", "age": 60.0, "gender": "male", "blood_pressure": "148/94", "diagnosis": "Hypertension", "admission_date": pd.Timestamp("2023-10-15"), "discharge_date": pd.Timestamp("2023-10-28"), "insurance_code": "INS-021"},
        {"patient_id": 1022, "name": "Uma Vargas", "age": 33.0, "gender": "female", "blood_pressure": "114/72", "diagnosis": "Thyroid", "admission_date": pd.Timestamp("2023-12-01"), "discharge_date": pd.Timestamp("2023-12-04"), "insurance_code": "INS-022"},
        {"patient_id": 1023, "name": "Victor Walsh", "age": 57.0, "gender": "male", "blood_pressure": "136/88", "diagnosis": "Arthritis", "admission_date": pd.Timestamp("2023-11-20"), "discharge_date": pd.Timestamp("2023-11-28"), "insurance_code": "INS-023"},
        {"patient_id": 1024, "name": "Wanda Xu", "age": 42.0, "gender": "female", "blood_pressure": "124/80", "diagnosis": "Allergies", "admission_date": pd.Timestamp("2024-01-05"), "discharge_date": pd.Timestamp("2024-01-07"), "insurance_code": "UNKNOWN"},
        {"patient_id": 1025, "name": "Xander Young", "age": None, "gender": "male", "blood_pressure": "118/76", "diagnosis": "Back Pain", "admission_date": pd.Timestamp("2023-12-18"), "discharge_date": pd.Timestamp("2023-12-20"), "insurance_code": "INS-025"},
        {"patient_id": 1026, "name": "Yolanda Zimmerman", "age": 70.0, "gender": "female", "blood_pressure": "152/96", "diagnosis": "Heart Disease", "admission_date": pd.Timestamp("2024-01-15"), "discharge_date": pd.Timestamp("2024-01-30"), "insurance_code": "INS-026"},
        {"patient_id": 1027, "name": "Aaron Bennett", "age": 38.0, "gender": "male", "blood_pressure": "122/78", "diagnosis": "Migraine", "admission_date": pd.Timestamp("2024-02-01"), "discharge_date": pd.Timestamp("2024-02-03"), "insurance_code": "INS-027"},
        {"patient_id": 1028, "name": "Brenda Cole", "age": 52.0, "gender": "female", "blood_pressure": "134/86", "diagnosis": "Diabetes", "admission_date": pd.Timestamp("2024-02-10"), "discharge_date": pd.Timestamp("2024-02-17"), "insurance_code": "INS-028"},
        {"patient_id": 1029, "name": "Carl Dixon", "age": 46.0, "gender": "male", "blood_pressure": "128/82", "diagnosis": "Bronchitis", "admission_date": pd.Timestamp("2024-03-01"), "discharge_date": pd.Timestamp("2024-03-05"), "insurance_code": "INS-029"},
        {"patient_id": 1030, "name": "Debra Ellis", "age": 61.0, "gender": "female", "blood_pressure": "142/90", "diagnosis": "Hypertension", "admission_date": pd.Timestamp("2024-03-10"), "discharge_date": pd.Timestamp("2024-03-22"), "insurance_code": "INS-030"},
        {"patient_id": 1031, "name": "Eric Flores", "age": 27.0, "gender": "male", "blood_pressure": "110/70", "diagnosis": "Allergies", "admission_date": pd.Timestamp("2024-04-01"), "discharge_date": pd.Timestamp("2024-04-02"), "insurance_code": "INS-031"},
        {"patient_id": 1032, "name": "Faye Grant", "age": 49.0, "gender": "female", "blood_pressure": "130/84", "diagnosis": "Anemia", "admission_date": pd.Timestamp("2024-04-15"), "discharge_date": pd.Timestamp("2024-04-19"), "insurance_code": "INS-032"},
        {"patient_id": 1033, "name": "Greg Hayes", "age": 56.0, "gender": "male", "blood_pressure": "140/88", "diagnosis": "Arthritis", "admission_date": pd.Timestamp("2024-05-01"), "discharge_date": pd.Timestamp("2024-05-08"), "insurance_code": "INS-033"},
        {"patient_id": 1034, "name": "Holly Ingram", "age": 35.0, "gender": "female", "blood_pressure": "116/74", "diagnosis": "Thyroid", "admission_date": pd.Timestamp("2024-05-20"), "discharge_date": pd.Timestamp("2024-05-23"), "insurance_code": "INS-034"},
        {"patient_id": 1035, "name": "Ivan Jensen", "age": 68.0, "gender": "male", "blood_pressure": "150/96", "diagnosis": "Heart Disease", "admission_date": pd.Timestamp("2024-06-01"), "discharge_date": pd.Timestamp("2024-06-14"), "insurance_code": "INS-035"},
    ]

    _sessions: dict[str, dict] = {}

    def __init__(self):
        pass

    def reset(self, seed=None, episode_id=None, **kwargs) -> CsvCleanObservation:
        session_id = kwargs.get("session_id", episode_id or "default")
        task = kwargs.get("task", "easy")

        if task == "easy":
            dirty_df = pd.DataFrame(self.DIRTY_1)
            truth_df = pd.DataFrame(self.CLEAN_1)
            task_description = (
                "Clean a sales dataset: strip whitespace from product names, "
                "fix price column type, fill missing quantities with 0."
            )
        elif task == "medium":
            dirty_df = pd.DataFrame(self.DIRTY_2)
            truth_df = pd.DataFrame(self.CLEAN_2)
            task_description = (
                "Clean an employee dataset: remove duplicates, standardize "
                "department casing, rename Department to department, fix "
                "join_date type, fill missing salaries with 0."
            )
        elif task == "hard":
            dirty_df = pd.DataFrame(self.DIRTY_3)
            truth_df = pd.DataFrame(self.CLEAN_3)
            task_description = (
                "Clean this medical records dataset for downstream analysis. "
                "The data has quality issues that need to be resolved before "
                "it can be used. Identify and fix all problems you find."
            )
        else:
            raise ValueError(f"Unknown task: {task}")

        self._sessions[session_id] = {
            "current_df": dirty_df.copy(),
            "ground_truth_df": truth_df.copy(),
            "current_task": task,
            "task_description": task_description,
            "steps_taken": 0,
            "done": False,
        }

        return CsvCleanObservation(
            current_csv=dirty_df.head(20).to_csv(index=False),
            num_rows=len(dirty_df),
            num_cols=len(dirty_df.columns),
            null_counts={col: int(v) for col, v in dirty_df.isnull().sum().items()},
            dtypes={col: str(dtype) for col, dtype in dirty_df.dtypes.items()},
            last_operation_result="Environment reset.",
            errors=[],
            task_name=task,
            task_description=task_description,
            steps_taken=0,
        )

    def _execute_operation(self, session_id: str, action: CsvCleanAction) -> tuple[str, list[str]]:
        try:
            session = self._sessions[session_id]
            df = session["current_df"]

            if action.operation == "drop_nulls":
                if not action.column:
                    return ("", ["Column name required for drop_nulls"])
                if action.column not in df.columns:
                    return ("", [f"Column '{action.column}' not found"])
                before = len(df)
                df = df.dropna(subset=[action.column]).reset_index(drop=True)
                session["current_df"] = df
                return (f"Dropped {before - len(df)} rows with nulls in '{action.column}'", [])

            elif action.operation == "fill_nulls":
                if not action.column:
                    return ("", ["Column name required for fill_nulls"])
                if not action.value:
                    return ("", ["Value required for fill_nulls"])
                if action.column not in df.columns:
                    return ("", [f"Column '{action.column}' not found"])
                col_dtype = df[action.column].dtype
                if col_dtype in (float, int) or pd.api.types.is_numeric_dtype(col_dtype):
                    fill_value = float(action.value)
                elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                    fill_value = pd.Timestamp(action.value)
                else:
                    fill_value = action.value
                count = int(df[action.column].isnull().sum())
                df[action.column] = df[action.column].fillna(fill_value)
                session["current_df"] = df
                return (f"Filled {count} nulls in '{action.column}' with {fill_value}", [])

            elif action.operation == "fix_type":
                if not action.column:
                    return ("", ["Column name required for fix_type"])
                if not action.value:
                    return ("", ["Target type required for fix_type"])
                if action.column not in df.columns:
                    return ("", [f"Column '{action.column}' not found"])
                if action.value in ("int", "float"):
                    df[action.column] = pd.to_numeric(df[action.column], errors="coerce")
                elif action.value == "datetime":
                    df[action.column] = pd.to_datetime(df[action.column], errors="coerce")
                elif action.value == "str":
                    df[action.column] = df[action.column].astype(str)
                session["current_df"] = df
                return (f"Cast column '{action.column}' to {action.value}", [])

            elif action.operation == "rename_column":
                if not action.column:
                    return ("", ["Column name required for rename_column"])
                if not action.value:
                    return ("", ["New name required for rename_column"])
                if action.column not in df.columns:
                    return ("", [f"Column '{action.column}' not found"])
                df = df.rename(columns={action.column: action.value})
                session["current_df"] = df
                return (f"Renamed column '{action.column}' to '{action.value}'", [])

            elif action.operation == "drop_column":
                if not action.column:
                    return ("", ["Column name required for drop_column"])
                if action.column not in df.columns:
                    return ("", [f"Column '{action.column}' not found"])
                df = df.drop(columns=[action.column])
                session["current_df"] = df
                return (f"Dropped column '{action.column}'", [])

            elif action.operation == "deduplicate":
                subset = [action.column] if action.column and action.column in df.columns else None
                before = len(df)
                df = df.drop_duplicates(subset=subset).reset_index(drop=True)
                session["current_df"] = df
                return (f"Removed {before - len(df)} duplicate rows", [])

            elif action.operation == "strip_whitespace":
                if not action.column:
                    return ("", ["Column name required for strip_whitespace"])
                if action.column not in df.columns:
                    return ("", [f"Column '{action.column}' not found"])
                df[action.column] = df[action.column].str.strip()
                session["current_df"] = df
                return (f"Stripped whitespace from '{action.column}'", [])

            elif action.operation == "standardize_case":
                if not action.column:
                    return ("", ["Column name required for standardize_case"])
                if action.column not in df.columns:
                    return ("", [f"Column '{action.column}' not found"])
                df[action.column] = df[action.column].str.lower()
                session["current_df"] = df
                return (f"Lowercased all values in '{action.column}'", [])

            else:
                return ("", [f"Unknown operation: '{action.operation}'"])

        except Exception as e:
            return ("", [str(e)])

    def _compute_score(self, session_id: str) -> float:
        session = self._sessions[session_id]
        current = session["current_df"]
        truth = session["ground_truth_df"]

        truth_cols = set(truth.columns)
        current_cols = set(current.columns)
        shared = truth_cols & current_cols

        # 1. dtype_score
        if not shared:
            dtype_score = 0.0
        else:
            matches = sum(
                1 for col in shared if str(current[col].dtype) == str(truth[col].dtype)
            )
            dtype_score = matches / len(truth.columns)

        # 2. null_score
        if not shared:
            null_score = 0.0
        else:
            matches = sum(
                1 for col in shared
                if current[col].isnull().sum() == truth[col].isnull().sum()
            )
            null_score = matches / len(truth.columns)

        # 3. shape_score
        row_score = 1.0 - min(abs(len(current) - len(truth)) / max(len(truth), 1), 1.0)
        col_score = 1.0 - min(abs(len(current.columns) - len(truth.columns)) / max(len(truth.columns), 1), 1.0)
        shape_score = (row_score + col_score) / 2

        # 4. value_score
        if len(current) != len(truth) or len(current.columns) != len(truth.columns):
            value_score = 0.0
        else:
            c = current.reset_index(drop=True)
            t = truth.reset_index(drop=True)
            shared_list = list(shared)
            if not shared_list:
                value_score = 0.0
            elif c[shared_list].equals(t[shared_list]):
                value_score = 1.0
            else:
                matching_cells = (c[shared_list] == t[shared_list]).sum().sum()
                total_cells = truth.shape[0] * len(shared_list)
                value_score = float(np.clip(matching_cells / total_cells, 0.0, 1.0))

        final_score = (dtype_score + null_score + shape_score + value_score) / 4
        return round(float(np.clip(final_score, 0.0, 1.0)), 4)

    def step(self, action: CsvCleanAction, timeout_s=None, **kwargs) -> CsvCleanObservation:
        session_id = kwargs.get("session_id", "default")

        if session_id not in self._sessions:
            return CsvCleanObservation(errors=["Session not found. Call reset() first."])

        session = self._sessions[session_id]

        if session["done"] or session["steps_taken"] >= self.MAX_STEPS:
            return self._build_observation(session_id, "Episode already ended.", ["Episode already ended."])

        session["steps_taken"] += 1

        if action.operation == "done":
            score = self._compute_score(session_id)
            session["done"] = True
            obs = self._build_observation(session_id, f"Episode complete. Final score: {score:.3f}", [])
            obs.reward = score
            obs.done = True
            return obs

        result, errors = self._execute_operation(session_id, action)
        return self._build_observation(session_id, result, errors)

    def _build_observation(self, session_id: str, result: str, errors: list[str]) -> CsvCleanObservation:
        session = self._sessions[session_id]
        df = session["current_df"]

        return CsvCleanObservation(
            current_csv=df.head(20).to_csv(index=False),
            num_rows=len(df),
            num_cols=len(df.columns),
            null_counts={col: int(count) for col, count in df.isnull().sum().items()},
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            last_operation_result=result,
            errors=errors,
            task_name=session["current_task"],
            task_description=session["task_description"],
            steps_taken=session["steps_taken"],
        )

    @property
    def state(self) -> State:
        return State(status="running")
