import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cfg import *
from data_process import process_input

global monitoring 
monitoring = True
monitor_data = []
warnings.filterwarnings('ignore')
try:
    from py3nvml import py3nvml
    py3nvml.nvmlInit()
    gpu_handle = py3nvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    has_gpu = True
except:
    has_gpu = False

def monitor_system():
    while monitoring:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        ram_used_mb = memory.used / (1024 * 1024)
        if has_gpu:
            gpu_power = py3nvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
        monitor_data.append((cpu_percent, ram_used_mb,gpu_power))
        time.sleep(1)

def main():

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("pca.pkl", "rb") as f:
        pca = pickle.load(f)

    with open("label_col.pkl", "rb") as f:
        label_col = pickle.load(f)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)
    print(y_encoded)
    
    df_cleaned = process_input(path_file = "capture_HUST_C7.csv", path_col = "danh_sach_cot_std.json")
    i = random.randint(0, df_cleaned.shape[0]-1)
    sample_df = df_cleaned.iloc[i:i+1]
    sample_X_scaled = scaler.transform(sample_df)
    sample_X_pca = pca.transform(sample_X_scaled)

    # Load mô hình 
    knn_loaded = joblib.load('knn_model.joblib')
    t = threading.Thread(target=monitor_system)
    t.start()
    time.sleep(5)
    
    start_time = time.time()
    pred_label = knn_loaded.predict(sample_X_pca)
    end_time = time.time()

    time.sleep(5)
    monitoring = False
    t.join()
    with open('log_infer_knn', mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cpu_percent", "ram_used_mb","power"])
            writer.writerows(monitor_data)

    pred_label_name = label_encoder.inverse_transform(pred_label)[0]
    inference_time_ms = (end_time - start_time) * 1000
    print(f'Giá trị dự đoán: {pred_label_name}')
    print(f'Thời gian dự đoán {inference_time_ms} (ms)')

if __name__ == "__main__":
    main()

