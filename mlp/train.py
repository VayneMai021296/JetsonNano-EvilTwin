import sys, os
import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cfg import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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
        time.sleep(interval)

def training(model, train_loader,val_loader,
              criterion, optimizer,scheduler,warmup_scheduler,class_weights_tensor,
              model_name ="mlp_model.pth",
              num_epochs = 50):
    # Huấn luyện mô hình
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    # Đo time 
    t1 = time.time()
    
    for epoch in range(num_epochs):
        # Huấn luyện
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            weights = class_weights_tensor[y_batch.long().squeeze()]
            loss = criterion(outputs, y_batch)
            loss = (loss * weights).mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_acc += calculate_accuracy(outputs, y_batch) * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
    
        # Đánh giá trên validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_acc += calculate_accuracy(outputs, y_batch) * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
        # Cập nhật learning rate
        scheduler.step(val_loss)
        warmup_scheduler.step()
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stop_msg = f"Early stopping triggered at epoch {epoch+1}"
                print(stop_msg)
                break

    print("-------------------- Kết thúc quá trình huấn luyện mô hình -------------------- ")
    t2 = time.time()
    print('Thời gian training là: {} seconds'.format((t2 - t1)))

    return train_losses, train_accuracies, val_losses, val_accuracies

def main():
    df_pca_with_label = pd.read_csv("pca.csv")
    X_pca = df_pca_with_label.drop(columns = ["Label"]).to_numpy()
    label_col = df_pca_with_label["Label"]

    num_features = X_pca.shape[1]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(label_col)
    print("Classes:", label_encoder.classes_)

    #Chia train, val, test loader
    train_loader, val_loader, test_loader = train_test_split_cus(X_pca,y_encoded,batch_size = 512, device = device)

    #Tính class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Class weights:", class_weights)

    #Khởi tạo mô hình
    model = MLP(input_dim = num_features, dropout = 0.35).to(device)

    #Đếm số lượng tham số trained và và ước lượng số lượng FLOPs
    #count_training_parameter(model)

    #Khởi tạo hàm loss, optimizer và scheduler
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-5)

    #Khởi tạo Warm-up scheduler cho quá trình training
    def lr_lambda(epoch):
        if epoch < 5:
            return epoch / 5.0
        return 1.0
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)

    #Traing 
    t = threading.Thread(target=monitor_system)
    t.start()
    time.sleep(5)
    t0 =time.time()
    print("-------------------- Bắt đầu huấn luyện với mô hình MLP -------------------- ")
    train_losses, train_accuracies, val_losses, val_accuracies = training(model,train_loader, val_loader, criterion,optimizer,
                                                                        scheduler,warmup_scheduler,
                                                                        class_weights_tensor,"mlp_model.pth",
                                                                        num_epochs = 20)
    t1 =time.time()
    print(f'Thời gian huấn luyện: {(t1-t0)} (seconds)')
    time.sleep(5)
    stop_event.set()
    t.join()
    process.terminate()
    write_to_csv(monitoring_data, "log_train_mlp.csv")
    
    print("\n-------------------- Đánh giá mô hình MLP trên tập Test -------------------- ")
    # Đánh giá trên tập test
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = (outputs >= 0.5).float().cpu().numpy().flatten()
            test_preds.extend(preds)
            test_labels.extend(y_batch.cpu().numpy().flatten())

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=[str(cls) for cls in label_encoder.classes_]))

    #Confusion Matrix
    conf_matrix = confusion_matrix(test_preds, test_labels)
    print("\nMa trận nhầm lẫn:")
    print(conf_matrix)
    #Chuyển sang định dạng numpy cho các metric
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    #Tính toán các chỉ số
    accuracy = accuracy_score(test_labels, test_preds)
    auc = roc_auc_score(test_labels, test_preds)
    precision = precision_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds)
    f1 = f1_score(test_labels, test_preds)

    #In ra kết quả
    print(f"\nAccuracy (Độ chính xác dự đoán tổng thể của mô hình): {accuracy:.4f}")
    print(f"AUC Score (Mức độ phân biệt 2 class của mô hình): {auc:.4f}")
    print(f"Precision (Tỷ lệ dự đoán của mô hình đối với TP (nhãn EvilTwin) ): {precision:.4f}")
    print(f"Recall (Tỷ lệ dự đoán không bỏ sót TP (nhãn EvilTwin)): {recall:.4f}")
    print(f"F1-score (Trung bình điều hoà giữa Precision và ReCall): {f1:.4f}")

    #Trực quan hoá Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Dự đoán Lớp Evil_Twin', 'Dự đoán Lớp Normal'],
                yticklabels=['Thực tế Lớp Evil_Twin', 'Thực tế Lớp Normal'])
    plt.title('Ma trận nhầm lẫn (Multi-layer Perceptron)')
    plt.ylabel('Giá trị thực tế')
    plt.xlabel('Giá trị dự đoán')
    plt.savefig('Confusion Matrix của Multi-layer Perceptron.png', dpi=500, bbox_inches='tight')

    #Plot loss và accuracy
    plot_and_save(train_losses, val_losses, train_accuracies, val_accuracies, "mlp_plot.png")


    fpr, tpr, thresholds = roc_curve(test_labels, test_preds)
    # Tính AUC
    auc_score = roc_auc_score(test_labels, test_preds)

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
    plt.savefig("AUC-mlp.png",dpi=500, bbox_inches='tight')