
import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cfg import *
from plot import plot_evaluation

global monitoring 
monitoring = True
monitor_data = []

def monitor_system():
    while monitoring:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        ram_used_mb = memory.used / (1024 * 1024)
        monitor_data.append((cpu_percent, ram_used_mb))
        time.sleep(1)

def main():

    df_pca_with_label = pd.read_csv("/kaggle/working/pca.csv") 

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

    monitoring = False
    t.join()
    with open('log_train_rf.csv', mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cpu_percent", "ram_used_mb"])
            writer.writerows(monitor_data)


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