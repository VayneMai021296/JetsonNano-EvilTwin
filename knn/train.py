

import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cfg import *
from plot import plot_evaluation
monitoring = True
monitor_data = []
warnings.filterwarnings('ignore')

def monitor_system():
    while monitoring:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        ram_used_mb = memory.used / (1024 * 1024)

        if has_gpu:
            gpu_power = py3nvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
        monitor_data.append((cpu_percent, ram_used_mb, gpu_power))
        time.sleep(1)

try:
    from py3nvml import py3nvml
    py3nvml.nvmlInit()
    gpu_handle = py3nvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
    has_gpu = True
except:
    has_gpu = False

def main():
    # --- KNN ---
    knn_model = KNeighborsClassifier(
        n_neighbors=5,       # Số lượng lân cận
        weights='uniform',   # Trọng số bằng nhau (hoặc 'distance' cho weighted)
        n_jobs=-1            # Dùng tất cả CPU
    )

    print("\n-------------------- Huấn luyện mô hình KNN --------------------")

    df_pca_with_label = pd.read_csv("./pca.csv")
    X_pca = df_pca_with_label.drop(columns = ["Label"]).to_numpy()
    label_col = df_pca_with_label["Label"]

    num_features = X_pca.shape[1]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)
    print("Classes:", label_encoder.classes_)

    # --- CHIA DỮ LIỆU THÀNH TẬP HUẤN LUYỆN VÀ KIỂM ĐỊNH ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # t = threading.Thread(target=monitor_system)
    # t.start()
    # time.sleep(5)
    t0 = time.time()
    knn_model.fit(X_train, y_train)
    t1 = time.time()
    time.sleep(5)

    # monitoring = False
    # t.join()
    # with open('log_train_knn.csv', mode="w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["cpu_percent", "ram_used_mb","power"])
    #         writer.writerows(monitor_data)

    print(f'Thời gian huấn luyện: {(t1-t0)*1000} (ms)')
    joblib.dump(knn_model, 'knn_model.joblib')
    # Dự đoán
    y_pred_knn = knn_model.predict(X_test)

    # Đánh giá
    print("\n====== ĐÁNH GIÁ MÔ HÌNH KNN ======")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_knn):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_knn):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_knn):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred_knn):.4f}")
    plot_evaluation(knn_model, X_test, y_test, model_name="KNN Classifier")

if __name__ =="__main__":
    main()