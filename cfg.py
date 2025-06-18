import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from sklearn.utils.class_weight import compute_class_weight
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import seaborn as sns 
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns # Để trực quan hóa confusion matrix
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix)

warnings.filterwarnings('ignore')

import pickle
import random
import joblib

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import psutil
import threading
import csv
import subprocess
import re
from xgboost import XGBClassifier

def parse_tegrastats_output(line):
    ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
    cpu_match = re.findall(r'(\d+)%@', line)
    gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)

    ram_used = int(ram_match.group(1)) if ram_match else None
    ram_total = int(ram_match.group(2)) if ram_match else None
    cpu_usages = list(map(int, cpu_match[:4])) if cpu_match else [0, 0, 0, 0]
    avg_cpu = sum(cpu_usages) / len(cpu_usages)
    gpu_usage = int(gpu_match.group(1)) if gpu_match else None

    return {
        "Time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "RAM Used (MB)": ram_used,
        "RAM Total (MB)": ram_total,
        "CPU Core 0 (%)": cpu_usages[0],
        "CPU Core 1 (%)": cpu_usages[1],
        "CPU Core 2 (%)": cpu_usages[2],
        "CPU Core 3 (%)": cpu_usages[3],
        "Average CPU Usage (%)": avg_cpu,
        "GPU Usage (%)": gpu_usage
    }

def write_to_csv(data, filename="jetson_monitor_log.csv"):
    if not data:
        return
    fieldnames = data[0].keys()
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Dữ liệu đã được lưu vào: {filename}")



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

def train_test_split_cus(X,y_encoded,batch_size = 512, device = 'cuda'):
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
