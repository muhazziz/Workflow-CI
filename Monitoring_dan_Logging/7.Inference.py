import requests
data = {
    "dataframe_split": {
        "columns": ["CreditScore", "Geography_Germany", "Geography_Spain", "Gender", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"],
        "data": [[600, 0, 0, 1, 40, 3, 60000.0, 2, 1, 1, 50000.0]]
    }
}
response = requests.post("http://127.0.0.1:5002/invocations", json=data)
print("Status Code:", response.status_code)
print("Hasil Prediksi:", response.json())