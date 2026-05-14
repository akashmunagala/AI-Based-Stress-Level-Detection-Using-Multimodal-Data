import numpy as np
import pandas as pd
import os




os.makedirs("data/processed", exist_ok=True)
np.random.seed(42)

N = 9000
subjects = 60

def make_class(n, label):
    subject_id = np.random.randint(1, subjects + 1, n)

    # ✅ OVERLAPPING distributions (REALISTIC)
    hr = np.random.normal(80 + label*12, 10, n)
    hrv = np.random.normal(70 - label*20, 12, n)
    eda = np.random.normal(2 + label*2, 1.2, n)
    temp = np.random.normal(36.7 + label*0.4, 0.3, n)
    resp = np.random.normal(16 + label*4, 3, n)
    spo2 = np.random.normal(97 - label*2, 1.5, n)
    sys = np.random.normal(120 + label*10, 10, n)
    dia = np.random.normal(80 + label*5, 6, n)
    cortisol = np.random.normal(8 + label*3, 2.5, n)

    motion = np.random.normal(0.4 + label*0.2, 0.2, n)
    sleep = np.random.normal(7 - label*1.2, 1.2, n)
    activity = np.random.normal(0.5 + label*0.2, 0.2, n)

    bmi = np.random.normal(22 + label*2, 3, n)
    hydration = np.random.normal(2.5 - label*0.3, 0.6, n)
    glucose = np.random.normal(90 + label*20, 15, n)

    df = pd.DataFrame({
        "subject_id": subject_id,
        "heart_rate": hr,
        "hrv": hrv,
        "eda": eda,
        "skin_temp": temp,
        "respiration_rate": resp,
        "spo2": spo2,
        "systolic_bp": sys,
        "diastolic_bp": dia,
        "cortisol": cortisol,
        "motion_level": motion,
        "sleep_quality": sleep,
        "activity_level": activity,
        "bmi": bmi,
        "hydration_level": hydration,
        "blood_glucose": glucose,
        "stress_label": label
    })

    # ✅ ADD NOISE (IMPORTANT)
    noise = np.random.normal(0, 0.3, df.shape)
    df.iloc[:, 1:-1] = df.iloc[:, 1:-1] + noise[:, 1:-1]

    # ✅ CLIPPING (REALISTIC LIMITS)
    df["heart_rate"] = df["heart_rate"].clip(50, 180)
    df["hrv"] = df["hrv"].clip(10, 120)
    df["eda"] = df["eda"].clip(0, 10)
    df["spo2"] = df["spo2"].clip(85, 100)
    df["motion_level"] = df["motion_level"].clip(0, 1)

    return df

# Balanced dataset
df0 = make_class(3000, 0)
df1 = make_class(3000, 1)
df2 = make_class(3000, 2)

df = pd.concat([df0, df1, df2], ignore_index=True)

# ✅ FINAL CLEANING
df = df.drop_duplicates()
df = df.dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df.to_csv("data/processed/dataset.csv", index=False)

print("✔ REALISTIC DATASET CREATED")
print("Shape:", df.shape)
print(df["stress_label"].value_counts())


# Save as Excel
df.to_excel("data/processed/dataset.xlsx", index=False)

print("✔ Excel file saved at: data/processed/dataset.xlsx")
