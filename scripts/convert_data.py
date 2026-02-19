import pandas as pd

# -----------------------------
# 1️⃣ Load SAML-D Dataset
# -----------------------------
input_file = "data/SAML-D.csv"
df = pd.read_csv(input_file)

# -----------------------------
# 2️⃣ Combine Date + Time
# -----------------------------
df["timestamp"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    errors="coerce"
)

# -----------------------------
# 3️⃣ Generate transaction_id
# -----------------------------
df.insert(
    0,
    "transaction_id",
    ["TXN_" + str(i).zfill(4) for i in range(1, len(df) + 1)]
)

# -----------------------------
# 4️⃣ Format Sender & Receiver IDs
# -----------------------------
df["sender_id"] = (
    df["Sender_account"]
    .astype(str)
    .str.extract(r"(\d+)")[0]   # extract numeric part if accounts like ACC123
    .fillna(df["Sender_account"])
    .apply(lambda x: "ACC_" + str(x).zfill(3))
)

df["receiver_id"] = (
    df["Receiver_account"]
    .astype(str)
    .str.extract(r"(\d+)")[0]
    .fillna(df["Receiver_account"])
    .apply(lambda x: "ACC_" + str(x).zfill(3))
)

# -----------------------------
# 5️⃣ Format Amount
# -----------------------------
df["amount"] = df["Amount"].astype(float).round(2)

# -----------------------------
# 6️⃣ Standardize Timestamp Format
# -----------------------------
df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# 7️⃣ Final Output Selection
# -----------------------------
final_df = df[
    ["transaction_id", "sender_id", "receiver_id", "amount", "timestamp"]
]

# -----------------------------
# 8️⃣ Export
# -----------------------------
output_file = "formatted_transactions.csv"
final_df.to_csv(output_file, index=False)

print("✅ Conversion complete.")
print(final_df.head())