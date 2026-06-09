import requests
import json

url = "http://127.0.0.1:5002/invocations"

data = {
    "dataframe_split": {
        "columns": [
            "CreditScore", "Gender", "Age", "Tenure", "Balance",
            "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
            "Geography_Germany", "Geography_Spain"
        ],
        "data": [[650, 0, 35, 5, 75000.0, 2, 1, 1, 95000.0, 0, 1]]
    }
}

response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)
)

print(f"Status Code: {response.status_code}")
result = response.json()
prediction = result["predictions"][0]
label = "Churn" if prediction == 1 else "Retain"
print(f"Hasil Prediksi: {label} (raw: {prediction})")