


import sys, os
import sys , os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cfg import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def calculate_accuracy(outputs, labels): # Hàm tính Accuracy 
    preds = (outputs >= 0.5).float()
    correct = (preds == labels).float().sum()
    accuracy = correct / labels.shape[0]
    return accuracy

def plot_and_save(train_losses,val_losses,train_accuracies,val_accuracies,file_name_figure = "unknow1.png"):
    plt.figure(figsize=(12, 5))
    
    train_losses = torch.tensor(train_losses).cpu().numpy()
    val_losses = torch.tensor(val_losses).cpu().numpy()
    
    train_accuracies = torch.tensor(train_accuracies).cpu().numpy()
    val_accuracies = torch.tensor(val_accuracies).cpu().numpy()
    
    # Biểu đồ loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(file_name_figure,dpi=500, bbox_inches='tight')
    plt.show()

def train_test_split_cus(X,y_encoded,batch_size = 512):
    # Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    print("Số mẫu train:", len(X_train))
    print("Số mẫu test:", len(X_test))
    # Chia tập train thành train và validation (lấy 20% của tập train làm valid còn lại là train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print("Số mẫu validation:", len(X_val))
    
    # Chuyển dữ liệu sang Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
    
    # Tạo DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_dim, dropout = 0.1):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout)
        )
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

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
    train_loader, val_loader, test_loader = train_test_split_cus(X_pca,y_encoded,batch_size = 512)

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
    print("-------------------- Bắt đầu huấn luyện với mô hình MLP -------------------- ")
    train_losses, train_accuracies, val_losses, val_accuracies = training(model,train_loader, val_loader, criterion,optimizer,
                                                                        scheduler,warmup_scheduler,
                                                                        class_weights_tensor,"best_mlp_model.pth",
                                                                        num_epochs = 20)
    
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
    plt.show()

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
    plt.show()