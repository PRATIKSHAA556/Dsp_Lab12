
import pandas as pd

df = pd.read_csv("data.csv")




drop_cols = [col for col in ["Name", "Email", "Phone"] if col in df.columns]
df_anonymized = df.drop(columns=drop_cols)






if "Age" in df.columns:
    age_min, age_max = df["Age"].min(), df["Age"].max()
    bins = list(range(0, (age_max // 10 + 2) * 10, 10))
    df_anonymized["Age_Range"] = pd.cut(df["Age"], bins=bins, right=False).astype(str)
else:
    df_anonymized["Age_Range"] = "UNKNOWN"







if "ZipCode" in df.columns:
    df["ZipCode"] = df["ZipCode"].astype(str)   
    df_anonymized["Zipcode_Masked"] = df["ZipCode"].str[:3] + "***"
else:
    df_anonymized["Zipcode_Masked"] = "UNKNOWN"





df_anonymized = df_anonymized.drop(columns=[c for c in ["Age", "ZipCode"] if c in df_anonymized.columns])





k = 5
group_cols = ["Age_Range", "Zipcode_Masked"]

def calc_risk(data, groups, k):
    """Return violating groups and risk %."""
    group_sizes = data.groupby(groups).size()
    violating = group_sizes[group_sizes < k].index
    risk = len(data[data[groups].apply(tuple, axis=1).isin(violating)]) / len(data) * 100
    return violating, risk, group_sizes




violating_groups, risk_before, group_sizes_before = calc_risk(df_anonymized, group_cols, k)



df_anonymized["Age_Range_Anonymized"] = df_anonymized["Age_Range"]
df_anonymized["Zipcode_Masked_Anonymized"] = df_anonymized["Zipcode_Masked"]

for age, zipc in violating_groups:
    mask = (df_anonymized["Age_Range"] == str(age)) & (df_anonymized["Zipcode_Masked"] == zipc)
    df_anonymized.loc[mask, "Age_Range_Anonymized"] = "ANY"
    df_anonymized.loc[mask, "Zipcode_Masked_Anonymized"] = "*****"



violating_groups_after, risk_after, group_sizes_after = calc_risk(
    df_anonymized, ["Age_Range_Anonymized", "Zipcode_Masked_Anonymized"], k
)



output_file = "pii_sample_100_anonymized.csv"
df_anonymized.to_csv(output_file, index=False)



print(f" Anonymized dataset saved to {output_file}")
print(f" Records: {len(df_anonymized)}")
print(f" Violating groups before: {len(violating_groups)}  | Risk: {risk_before:.2f}%")
print(f" Violating groups after : {len(violating_groups_after)}  | Risk: {risk_after:.2f}%")
print("\nSample of anonymized dataset:\n", df_anonymized.head(10))


