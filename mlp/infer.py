
import sys, os
import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cfg import *
from data_process import process_input, load_agent
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

stop_event = threading.Event() 
monitoring_data  = []
process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def monitor_system(interval = 1.0):
    while not stop_event.is_set():
        line = process.stdout.readline()
        if not line:
            break
        line_str = line.decode('utf-8').strip()
        stats = parse_tegrastats_output(line_str)
        monitoring_data.append(stats)
        time.sleep(interval)

def main():
    df_time_cleaned = process_input(path_file = "capture_HUST_C7.csv",
                            path_col= "danh_sach_cot_std.json")


    scaler, label_encoder =  load_agent()
    
    i = random.randint(0, df_time_cleaned.shape[0] - 1)
    print(f"Vị trí ngẫu nhiên được dự đoán là: {i}")
    sample_df = df_time_cleaned.iloc[i:i + 1]
    sample_X_scaled = scaler.transform(sample_df)
    sample_X_tensor = torch.tensor(sample_X_scaled, dtype=torch.float32).to(device)

    # Khởi tạo mô hình MLP
    num_features = sample_X_scaled.shape[1]
    model = MLP(num_features).to(device)
    # Tải trọng số mô hình
    model.load_state_dict(torch.load('mlp_model.pth'))
    model.eval()
    # Thực hiện inference và tính thời gian
    t = threading.Thread(target=monitor_system)
    t.start()
    time.sleep(5)
    start_time = time.perf_counter()
    with torch.no_grad():
        pred_prob = model(sample_X_tensor)
    end_time = time.perf_counter()
    time.sleep(5)
    stop_event.set()
    t.join()
    process.terminate()
    write_to_csv(monitoring_data, "log_infer_mlp.csv")
    # Chuyển xác suất thành nhãn
    pred_prob = pred_prob.cpu().numpy()[0, 0]
    pred_label = 1 if pred_prob >= 0.5 else 0
    pred_label_name = label_encoder.inverse_transform([pred_label])[0]
    inference_time_ms = (end_time - start_time) * 1000

    print(f'Giá trị dự đoán: {pred_label_name}')
    print(f'Thời gian dự đoán: {inference_time_ms:.2f} ms')

if __name__ =="__main__" :
    main()



