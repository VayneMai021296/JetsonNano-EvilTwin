import sys, os
import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cfg import *

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

    df = pd.read_csv("scaled.csv")
    label_col = df["Label"]
    X_pca = df.drop(columns = ["Label"])

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42,stratify=y_encoded)
    print("Số mẫu train:", len(X_train))
    print("Số mẫu test:", len(X_test))

    print(f"\nKích thước tập huấn luyện X: {X_train.shape}")
    print(f"Kích thước tập kiểm định X: {X_test.shape}")

    print(f"Kích thước tập huấn luyện y: {y_train.shape}")
    print(f"Kích thước tập kiểm định y: {y_test.shape}")

    print(f"Tỉ lệ lớp trong y_train (0/1): {np.bincount(y_train) / len(y_train)}")
    print(f"Tỉ lệ lớp trong y_test (0/1): {np.bincount(y_test) / len(y_test)}")


    # --- PHẦN 2: XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH XGBOOST ---
    """ Các siêu tham số của mô hình XGBoost 
        objective: Hàm mục tiêu ('binary:logistic' cho phân loại nhị phân với đầu ra xác suất)
        n_estimators: Số lượng cây boosting
        learning_rate: Tốc độ học (shrinkage)
        max_depth: Độ sâu tối đa của mỗi cây con
        subsample: Tỷ lệ lấy mẫu con của dữ liệu (hàng) cho mỗi cây
        colsample_bytree: Tỷ lệ lấy mẫu con của thuộc tính (cột) cho mỗi cây
        gamma: Mức giảm mất mát tối thiểu cần thiết để thực hiện một phân chia.
        reg_alpha (L1) và reg_lambda (L2): Tham số regularization cho trọng số lá.
        eval_metric: Metric để đánh giá trong quá trình huấn luyện (để theo dõi early stopping)
        use_label_encoder=False: Để tránh cảnh báo deprecation
        tree_method='hist': Sử dụng thuật toán xây dựng cây dựa trên histogram, nhanh hơn cho dữ liệu lớn.
        n_jobs: Số lượng nhân CPU để sử dụng (-1 nghĩa là sử dụng tất cả)
    """

    xgb_classifier = XGBClassifier(
        objective='binary:logistic',  # Thích hợp cho phân loại nhị phân
        n_estimators=300,             # Số lượng cây boosting
        learning_rate=0.05,           # Tốc độ học
        max_depth=6,                  # Độ sâu tối đa của mỗi cây
        subsample=0.7,                # Sử dụng 70% mẫu ngẫu nhiên cho mỗi cây
        colsample_bytree=0.7,         # Sử dụng 70% thuộc tính ngẫu nhiên cho mỗi cây
        gamma=0.1,                    # Minimum loss reduction for a split
        reg_alpha=0.005,              # L1 regularization on weights
        reg_lambda=1,                 # L2 regularization on weights
        use_label_encoder=False,      # Tắt cảnh báo về label encoder (đã deprecated)
        eval_metric='logloss',        # Metric đánh giá (có thể là 'auc', 'error')
        random_state=42,
        n_jobs=-1                     # Sử dụng tất cả các nhân CPU
    )

    print("\n-------------------- Bắt đầu huấn luyện với mô hình XGBoost -------------------- ")
    t = threading.Thread(target=monitor_system)
    t.start()
    time.sleep(5)
    t0 = time.time()
    xgb_classifier.fit(X_train, y_train,
                    eval_set=[(X_test, y_test)], 
                    early_stopping_rounds=20,    
                    verbose=False)              
    print("-------------------- Kết thúc quá trình huấn luyện mô hình --------------------")
    # Số lượng cây thực tế được huấn luyện sau early stopping
    t1=time.time()
    time.sleep(5)
    joblib.dump(xgb_classifier, 'xgb_model.joblib')
    stop_event.set()
    t.join()
    process.terminate()
    write_to_csv(monitoring_data, "log_train_xgb.csv")

    print(f"Huấn luyện hoàn tất. Số lượng cây thực tế: {xgb_classifier.best_iteration + 1 if hasattr(xgb_classifier, 'best_iteration') else xgb_classifier.n_estimators} cây.")
    print(f'Thời gian huấn luyện: {(t1-t0)} (seconds)')


    print("\n-------------------- Đánh giá mô hình XGBoost trên tập Test -------------------- ")

    y_pred_encoded = xgb_classifier.predict(X_test)

    # **BƯỚC QUAN TRỌNG: CHUYỂN ĐỔI y_test VÀ y_pred TRỞ LẠI NHÃN GỐC**
    y_test_original = label_encoder.inverse_transform(y_test)
    y_pred_original = label_encoder.inverse_transform(y_pred_encoded)

    #Tính toán các chỉ số
    accuracy = accuracy_score(y_test, y_pred_encoded)
    auc = roc_auc_score(y_test, y_pred_encoded)
    precision = precision_score(y_test, y_pred_encoded)
    recall = recall_score(y_test, y_pred_encoded)
    f1 = f1_score(y_test, y_pred_encoded)

    #In ra kết quả
    print(f"\nAccuracy (Độ chính xác dự đoán tổng thể của mô hình): {accuracy:.4f}")
    print(f"AUC Score (Mức độ phân biệt 2 class của mô hình): {auc:.4f}")
    print(f"Precision (Tỷ lệ dự đoán của mô hình đối với TP (nhãn EvilTwin) ): {precision:.4f}")
    print(f"Recall (Tỷ lệ dự đoán không bỏ sót TP (nhãn EvilTwin)): {recall:.4f}")
    print(f"F1-score (Trung bình điều hoà giữa Precision và ReCall): {f1:.4f}")

    # Hiển thị báo cáo phân loại chi tiết với NHÃN GỐC
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test_original, y_pred_original, target_names=[str(cls) for cls in label_encoder.classes_]))

    # Hiển thị Ma trận nhầm lẫn với NHÃN GỐC
    conf_matrix = confusion_matrix(y_test_original, y_pred_original)
    print("\nMa trận nhầm lẫn:")
    print(conf_matrix)

    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dự đoán Lớp Evil_Twin', 'Dự đoán Lớp Normal'],
                yticklabels=['Thực tế Lớp Evil_Twin', 'Thực tế Lớp Normal'])
    plt.title('Ma trận nhầm lẫn (XGBoost)')
    plt.ylabel('Giá trị thực tế')
    plt.xlabel('Giá trị dự đoán')
    plt.savefig('Confusion Matrix của XGBoost.png', dpi=500, bbox_inches='tight') 
    
    # AUC-ROC
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_encoded)
    # Tính AUC
    auc_score = roc_auc_score(y_test, y_pred_encoded)

    # Vẽ biểu đồ ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("AUC-xgb.png",dpi=500, bbox_inches='tight')

if __name__ == "__mai__":
    main()


    