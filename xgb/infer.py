
import sys, os
import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cfg import *
from xgboost import XGBClassifier
from data_process import process_input,load_agent

stop_event = threading.Event() 
monitoring_data  = []
process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
def monitor_system(interval = 0.1):
    while not stop_event.is_set():
        line = process.stdout.readline()
        if not line:
            break
        line_str = line.decode('utf-8').strip()
        stats = parse_tegrastats_output(line_str)
        monitoring_data.append(stats)
        time.sleep(interval)

def main():
    df_time_cleaned = process_input(path_file = "capture_HUSTC7_ch48_4_beacon.csv",
                            path_col= "danh_sach_cot_std.json")

    scaler,label_encoder = load_agent()

    xgb_classifier = XGBClassifier()
    xgb_classifier = joblib.load('xgb_model.joblib')
    # n_sample = df_time_cleaned.shape[0]

    # for i in range(0, n_sample):
    #     i = random.randint(0, df_time_cleaned.shape[0] - 1)
    #     print(f"Vị trí ngẫu nhiên được dự đoán là: {i}")
    #     sample_df = df_time_cleaned.iloc[i:i + 1]
    #     sample_X_scaled = scaler.transform(sample_df)

    #     t = threading.Thread(target=monitor_system)
    #     t.start()
    #     time.sleep(5)
    #     start_time = time.time()
    #     pred_prob = xgb_classifier.predict_proba(sample_X_scaled)[:, 1]
    #     end_time = time.time()
    #     time.sleep(5)
    #     stop_event.set()
    #     t.join()
    #     process.terminate()
    #     write_to_csv(monitoring_data, "log_infer_xgb.csv")

    #     pred_label = 1 if pred_prob >= 0.5 else 0
    #     pred_label_name = label_encoder.inverse_transform([pred_label])[0]
    #     inference_time_ms = (end_time - start_time) * 1000

    #     print(f'Giá trị dự đoán: {pred_label_name}')
    #     print(f'Thời gian dự đoán: {inference_time_ms:.2f} ms')

    results_xgb_df = pd.DataFrame(columns=['prediction','inference time (ms)'])
    new_row = {}
    t = threading.Thread(target=monitor_system)
    t.start()
    time.sleep(5)

    for i in range(0, df_time_cleaned.shape[0]):

        sample_df = df_time_cleaned.iloc[i:i+1]
        sample_X_raw = sample_df
        sample_X_scaled = scaler.transform(sample_X_raw)
        start_time = time.time()
        pred_prob = xgb_classifier.predict_proba(sample_X_scaled)[:, 1]
        end_time = time.time()
        pred_label = 1 if pred_prob >= 0.5 else 0
        pred_label_name = label_encoder.inverse_transform([pred_label])[0]
        inference_time_ms = (end_time - start_time) * 1000
       
        new_row = {'prediction': pred_label_name, 'inference time (ms)': round(inference_time_ms, 2)}
        results_xgb_df = pd.concat([results_xgb_df, pd.DataFrame([new_row])], ignore_index=True)

    time.sleep(5)
    stop_event.set()
    t.join()
    process.terminate()
    write_to_csv(monitoring_data, "log_infer_xgb.csv")

    results_xgb_df.to_csv("result_xgb_infer.csv", index=False)

if __name__== "__main__":

    main()



   