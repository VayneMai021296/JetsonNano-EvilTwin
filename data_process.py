
import pickle
from sklearn.model_selection import train_test_split
import torch 
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import json 
import platform
import numpy as np
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
            on_bad_lines='skip',
            low_memory=False
        )
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

    for col in df_time_cleaned.columns:
        if df_time_cleaned[col].isnull().all():
            df_time_cleaned[col] = df_time_cleaned[col].fillna(0)
        else:
            df_time_cleaned[col] = df_time_cleaned[col].fillna(df_time_cleaned[col].mean)

    return df_time_cleaned

def load_agent():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    print("Classes:", label_encoder.classes_)

    return scaler, label_encoder
