from prometheus_client import start_http_server, Counter, Gauge, Histogram
import psutil
import time
import random
import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/Bank_Customer_Churn_Prediction/1")

REQUESTS = Counter('http_requests_total', 'Total HTTP Requests')
LATENCY = Histogram('api_latency_seconds', 'API Latency')
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')
CHURN_PREDS = Counter('model_predictions_churn_total', 'Total Churn Predictions')
RETAIN_PREDS = Counter('model_predictions_retain_total', 'Total Retain Predictions')
ERRORS = Counter('api_errors_total', 'Total API Errors')
ACTIVE_CONN = Gauge('active_connections', 'Active Connections')
DISK_USAGE = Gauge('system_disk_usage', 'Disk Usage Percentage')
NETWORK_BYTES = Counter('network_bytes_total', 'Network Bytes Transmitted')

def generate_sample():
    return pd.DataFrame([{
        "CreditScore": random.randint(300, 850),
        "Geography": random.choice([0, 1, 2]),
        "Gender": random.choice([0, 1]),
        "Age": random.randint(18, 70),
        "Tenure": random.randint(0, 10),
        "Balance": round(random.uniform(0, 250000), 2),
        "NumOfProducts": random.randint(1, 4),
        "HasCrCard": random.choice([0, 1]),
        "IsActiveMember": random.choice([0, 1]),
        "EstimatedSalary": round(random.uniform(10000, 200000), 2)
    }])

def collect_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    REQUESTS.inc(random.randint(1, 5))

    start = time.time()
    try:
        sample = generate_sample()
        prediction = model.predict(sample)[0]
        LATENCY.observe(time.time() - start)
        if prediction == 1:
            CHURN_PREDS.inc()
        else:
            RETAIN_PREDS.inc()
    except Exception:
        ERRORS.inc()

    ACTIVE_CONN.set(random.randint(10, 50))
    NETWORK_BYTES.inc(random.randint(500, 2000))

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter berjalan di http://localhost:8000")
    while True:
        collect_metrics()
        time.sleep(5)