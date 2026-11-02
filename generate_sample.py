"""Generate a deliberately messy sample dataset (sample_data.csv) for demos and tests."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 400

first = ["Arun", "Priya", "Karthik", "Divya", "Suresh", "Meena", "Rahul", "Anitha", "Vijay", "Lakshmi",
         "Manoj", "Kavya", "Harish", "Sneha", "Naveen", "Pooja", "Gokul", "Deepa", "Ashwin", "Nisha"]
last = ["Kumar", "Raj", "Sharma", "Iyer", "Nair", "Reddy", "Pillai", "Das", "Menon", "Singh"]
cities = ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy", "Bangalore"]
city_noise = {
    "Chennai": ["chennai", "CHENNAI", "Chenai", "Chennai "],
    "Coimbatore": ["coimbatore", "Coimbatore.", "Coimbatore  "],
    "Madurai": ["madurai", "MADURAI", "Maduari"],
    "Salem": ["salem", "SALEM"],
    "Trichy": ["trichy", "Trichy ", "TRICHY"],
    "Bangalore": ["bangalore", "Bengaluru", "BANGALORE"],
}
depts = ["Sales", "Marketing", "Finance", "Operations", "HR"]

rows = []
for i in range(N):
    fn, ln = rng.choice(first), rng.choice(last)
    city = rng.choice(cities)
    if rng.random() < 0.35:
        city = rng.choice(city_noise[city])
    age = int(rng.normal(34, 8))
    salary = int(rng.normal(52000, 12000))
    joined = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 2000)))
    fmt = rng.choice(["%d-%m-%Y", "%Y/%m/%d", "%d %b %Y"])
    email = f"{fn}.{ln}{rng.integers(1, 99)}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}".lower()
    rows.append({
        "Customer ID": f"C{1000 + i}",
        "Full Name": f"  {fn}   {ln} " if rng.random() < 0.15 else f"{fn} {ln}",
        "E-mail": email,
        "Age": age,
        "City": city,
        "Department": rng.choice(depts),
        "Join Date": joined.strftime(fmt),
        "Monthly Salary": f"Rs. {salary:,}" if rng.random() < 0.5 else salary,
        "Performance Score": round(float(np.clip(rng.normal(72, 12), 0, 100)), 1),
        "Active": rng.choice(["Yes", "No", "Y", "N", "yes", "TRUE", "false"]),
    })

df = pd.DataFrame(rows)

# Inject problems ------------------------------------------------------------
idx = lambda p: rng.choice(df.index, size=int(N * p), replace=False)
df.loc[idx(0.08), "Age"] = np.nan
df["Age"] = df["Age"].astype("object")
df.loc[idx(0.05), "Age"] = "N/A"
df.loc[idx(0.07), "Performance Score"] = np.nan
df.loc[idx(0.06), "Department"] = "-"
df.loc[idx(0.05), "City"] = "null"
df.loc[idx(0.04), "E-mail"] = "not_an_email"
df.loc[idx(0.03), "E-mail"] = np.nan
df.loc[idx(0.05), "Monthly Salary"] = "?"
df.loc[rng.choice(df.index, 6, replace=False), "Age"] = rng.choice([180, 250, -5, 999], 6)
df.loc[rng.choice(df.index, 5, replace=False), "Performance Score"] = rng.choice([950, 1200, -40], 5)
df.loc[rng.choice(df.index, 4, replace=False), "Monthly Salary"] = rng.choice([9999999, 5000000], 4)
df["Empty Notes"] = np.nan          # fully empty column
df["Remarks"] = pd.Series(np.where(rng.random(N) < 0.85, None, "check"), dtype="object")  # mostly empty column

dups = df.sample(25, random_state=1)
df = pd.concat([df, dups], ignore_index=True)
blank = pd.DataFrame([[np.nan] * df.shape[1]] * 3, columns=df.columns)
df = pd.concat([df, blank], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)

df.to_csv("sample_data.csv", index=False)
print("Wrote sample_data.csv", df.shape)
