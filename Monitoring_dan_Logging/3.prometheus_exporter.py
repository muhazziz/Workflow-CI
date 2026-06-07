import joblib
import pandas as pd
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import psutil
import time
import random

MODEL_PATH = r"C:\Users\muham_ogp6gj0\Eksperimen_SML_Muh-Azizsyah-Putra\Membangun_model\mlartifacts\1\models\m-bf4ee8f846a34a4da977afd1e20cb931\artifacts\model.pkl"
DATA_PATH = r"C:\Users\muham_ogp6gj0\Eksperimen_SML_Muh-Azizsyah-Putra\churn_preprocessing\test.csv"

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)
if 'Exited' in data.columns:
    data = data.drop('Exited', axis=1)

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

def collect_metrics():
    CPU_USAGE.set(psutil.cpu_percent())
    RAM_USAGE.set(psutil.virtual_memory().percent)
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    REQUESTS.inc(random.randint(1, 5))
    ACTIVE_CONN.set(random.randint(10, 50))
    NETWORK_BYTES.inc(random.randint(500, 2000))

    try:
        sample = data.sample(n=1)
        start = time.time()
        prediction = model.predict(sample)[0]
        LATENCY.observe(time.time() - start)
        if prediction == 1:
            CHURN_PREDS.inc()
        else:
            RETAIN_PREDS.inc()
    except Exception:
        ERRORS.inc()

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter berjalan di http://localhost:8000")
    while True:
        collect_metrics()
        time.sleep(5)