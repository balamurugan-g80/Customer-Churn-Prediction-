import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CSV_FILE = "Telco-Customer-Churn.csv"

# ================= LOAD DATA =================
df = pd.read_csv(CSV_FILE)
original_df = df.copy()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

df_model = df.drop("customerID", axis=1)
df_model = pd.get_dummies(df_model, drop_first=True)

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

training_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

# ================= GRAPHS =================
def prediction_graph(result):
    labels = ["NOT CHURN", "CHURN"]
    values = [1, 0] if result == 0 else [0, 1]

    plt.figure(figsize=(6,4))
    plt.bar(labels, values)
    plt.title("Customer Churn Prediction")
    plt.ylim(0, 1.2)
    plt.show()

def churn_distribution():
    counts = original_df["Churn"].value_counts()
    plt.figure(figsize=(6,4))
    plt.bar(counts.index, counts.values)
    plt.title("Churn Distribution")
    plt.show()

# ================= EXISTING CUSTOMER =================
def existing_customer():
    cid = input("Enter Customer ID: ").strip()

    row_df = original_df[original_df["customerID"] == cid]

    if row_df.empty:
        print("Customer Not Found")
        return

    row = row_df.iloc[0]

    print("\nCUSTOMER DETAILS")
    for col in original_df.columns:
        print(f"{col}: {row[col]}")

    actual = row["Churn"]

    temp = row_df.drop("customerID", axis=1).copy()
    temp["Churn"] = temp["Churn"].map({"No":0,"Yes":1})
    temp["TotalCharges"] = pd.to_numeric(temp["TotalCharges"], errors="coerce")
    temp["TotalCharges"] = temp["TotalCharges"].fillna(df["TotalCharges"].median())

    temp = temp.drop("Churn", axis=1)
    temp = pd.get_dummies(temp)
    temp = temp.reindex(columns=training_columns, fill_value=0)

    temp_scaled = scaler.transform(temp)
    result = model.predict(temp_scaled)[0]

    print("\nActual Churn:", actual)
    print("Predicted:", "CHURN" if result == 1 else "NOT CHURN")

    prediction_graph(result)

# ================= NEW CUSTOMER =================
def new_customer():
    sample = {
        "gender": input("Gender (Male/Female): "),
        "SeniorCitizen": int(input("SeniorCitizen (0/1): ")),
        "Partner": input("Partner (Yes/No): "),
        "Dependents": input("Dependents (Yes/No): "),
        "tenure": int(input("Tenure: ")),
        "PhoneService": input("PhoneService (Yes/No): "),
        "MultipleLines": input("MultipleLines: "),
        "InternetService": input("InternetService: "),
        "OnlineSecurity": input("OnlineSecurity: "),
        "OnlineBackup": input("OnlineBackup: "),
        "DeviceProtection": input("DeviceProtection: "),
        "TechSupport": input("TechSupport: "),
        "StreamingTV": input("StreamingTV: "),
        "StreamingMovies": input("StreamingMovies: "),
        "Contract": input("Contract: "),
        "PaperlessBilling": input("PaperlessBilling (Yes/No): "),
        "PaymentMethod": input("PaymentMethod: "),
        "MonthlyCharges": float(input("MonthlyCharges: ")),
        "TotalCharges": float(input("TotalCharges: "))
    }

    sample_df = pd.DataFrame([sample])
    sample_df = pd.get_dummies(sample_df)
    sample_df = sample_df.reindex(columns=training_columns, fill_value=0)

    sample_scaled = scaler.transform(sample_df)
    result = model.predict(sample_scaled)[0]

    print("\nNEW CUSTOMER REPORT")
    print("Prediction:", "CHURN" if result == 1 else "NOT CHURN")

    prediction_graph(result)

# ================= MENU =================
while True:
    print("\n===== CUSTOMER CHURN PREDICTION USING SVM =====")
    print("1. Existing Customer")
    print("2. New Customer")
    print("3. Churn Distribution Graph")
    print("4. Accuracy")
    print("5. Classification Report")
    print("6. Exit")

    choice = input("Enter Choice: ")

    if choice == "1":
        existing_customer()
    elif choice == "2":
        new_customer()
    elif choice == "3":
        churn_distribution()
    elif choice == "4":
        print(f"Accuracy: {accuracy*100:.2f}%")
    elif choice == "5":
        print(confusion_matrix(y_test, pred))
        print(classification_report(y_test, pred))
    elif choice == "6":
        break
    else:
        print("Invalid Choice")
