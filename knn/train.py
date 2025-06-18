import threading
import time
import psutil
import warnings
import csv
import os
import sys
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from cfg import *
from plot import plot_evaluation

stop_event = threading.Event()  # Dùng event để dừng thread
monitoring_data  = []
process = subprocess.Popen(['tegrastats'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
def monitor_system(interval = 1.0):
    while not stop_event.is_set():
        line = process.stdout.readline()
        if not line:
            break
        stats = parse_tegrastats_output(line)
        monitoring_data.append(stats)
        print(stats)
        time.sleep(interval)

# --- Main ---
def main():
    knn_model = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        n_jobs=-1
    )

    print("\n-------------------- Huấn luyện mô hình KNN --------------------")

    df_pca_with_label = pd.read_csv("./pca.csv")
    X_pca = df_pca_with_label.drop(columns=["Label"]).to_numpy()
    label_col = df_pca_with_label["Label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Start monitor thread
    t = threading.Thread(target=monitor_system)
    t.start()
    time.sleep(5)

    t0 = time.time()
    knn_model.fit(X_train, y_train)
    t1 = time.time()
    time.sleep(5)
    # Stop monitor
    stop_event.set()
    t.join()
    process.terminate()
    write_to_csv(monitoring_data, "log_train_knn.csv")
   
    print(f'Thời gian huấn luyện: {(t1 - t0) :.2f} s')
    joblib.dump(knn_model, 'knn_model.joblib')

    y_pred_knn = knn_model.predict(X_test)

    print("\n====== ĐÁNH GIÁ MÔ HÌNH KNN ======")
    print(f"Accuracy : {accuracy_score(y_test, y_pred_knn):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_knn):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred_knn):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred_knn):.4f}")

    plot_evaluation(knn_model, X_test, y_test, model_name="KNN Classifier")

if __name__ == "__main__":
    main()
