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
