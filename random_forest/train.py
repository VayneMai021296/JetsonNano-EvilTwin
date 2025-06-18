
import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cfg import *
from plot import plot_evaluation

stop_event = threading.Event()  # Dùng event để dừng thread
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
        print(stats)
        time.sleep(interval)

def main():

    df_pca_with_label = pd.read_csv("./pca.csv")

    # --- Random Forest ---
    rf_model = RandomForestClassifier(
        n_estimators=100,       # Số lượng cây
        max_depth=None,         # Không giới hạn độ sâu
        random_state=42,
        n_jobs=-1               # Dùng tất cả CPU
    )

    print("\n-------------------- Huấn luyện mô hình Random Forest --------------------")
    X_pca = df_pca_with_label.drop(columns = ["Label"]).to_numpy()
    label_col = df_pca_with_label["Label"]
    num_features = X_pca.shape[1]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)
    print("Classes:", label_encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    t = threading.Thread(target=monitor_system)
    t.start()
    time.sleep(5)

    t0 = time.time()
    rf_model.fit(X_train, y_train)
    t1 = time.time()
    time.sleep(5)
    # Stop monitor
    stop_event.set()
    t.join()
    process.terminate()
    write_to_csv(monitoring_data,"log_train_rf.csv")
   

    print(f'Thời gian huấn luyện: {(t1-t0)*1000} (ms)')
    joblib.dump(rf_model, 'rf_model.joblib')

    # Dự đoán
    y_pred_rf = rf_model.predict(X_test)

    # Đánh giá
    print("\n====== ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST ======")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred_rf):.4f}")

    plot_evaluation(rf_model, X_test, y_test, model_name="Random Forest")

if __name__ =="__main__":
    main()