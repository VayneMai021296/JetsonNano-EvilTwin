
import pickle
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json 
import platform
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
def load():
    
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_col.pkl', 'rb') as f:
        label_col = pickle.load(f)

        return scaler, pca, label_col
    
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

def process_input(path_file = "capture_HUST_C7.csv", path_col ="danh_sach_cot_std.json"):

    if platform.system() == "Windows":
        df_real = pd.read_csv(
            path_file,
            quotechar='"',
            on_bad_lines='skip',
            low_memory=False
        )
    elif platform.system() == "Linux":
        df_real = pd.read_csv(
            path_file,
            quotechar='"',
            error_bad_lines=False,
            low_memory=False
        )
    # Convert time to valid value 
    df_real_clone = df_real

    df_real_clone['frame.time'] = pd.to_datetime(
        df_real_clone['frame.time'].str.replace(' GTB Standard Time', '', regex=False),
        format="%b %d- %Y %H:%M:%S.%f",
        errors='coerce'
    )

    clone = pd.concat([
        df_real_clone,
        df_real_clone['frame.time'].dt.year.rename('frame_year'),
        df_real_clone['frame.time'].dt.month.rename('frame_month'),
        df_real_clone['frame.time'].dt.day.rename('frame_day'),
        df_real_clone['frame.time'].dt.hour.rename('frame_hour'),
        df_real_clone['frame.time'].dt.minute.rename('frame_minute'),
        df_real_clone['frame.time'].dt.second.rename('frame_second'),
        df_real_clone['frame.time'].dt.dayofweek.rename('frame_dayofweek')
    ], axis=1)

    df_time_cleaned = clone.copy()
   
    with open(path_col, "r") as f:
        column_list_sdt = json.load(f)
    
    column_list_sdt.remove("Label")
    df_time_cleaned = df_time_cleaned[column_list_sdt]

    # Fill missing value hoặc lấy giá trị mean hoặc 0
    for col in df_time_cleaned.columns:
        if df_time_cleaned[col].isnull().all():
            # Nếu cột toàn bộ là NaN, điền 0
            df_time_cleaned[col] = df_time_cleaned[col].fillna(0)
        else:
            # Ngược lại, điền bằng giá trị trung bình
            df_time_cleaned[col] = df_time_cleaned[col].fillna(df_time_cleaned[col].mean)

    return df_time_cleaned
