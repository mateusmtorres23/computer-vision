import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

def normalize_hand_coordinates(row):
    row_norm = row.copy()
    
    if row_norm['Left_x0'] != 0.0 or row_norm['Left_y0'] != 0.0: 
        base_x_l, base_y_l, base_z_l = row_norm['Left_x0'], row_norm['Left_y0'], row_norm['Left_z0']
        for i in range(21):
            row_norm[f'Left_x{i}'] -= base_x_l
            row_norm[f'Left_y{i}'] -= base_y_l
            row_norm[f'Left_z{i}'] -= base_z_l

    if row_norm['Right_x0'] != 0.0 or row_norm['Right_y0'] != 0.0:
        base_x_r, base_y_r, base_z_r = row_norm['Right_x0'], row_norm['Right_y0'], row_norm['Right_z0']
        for i in range(21):
            row_norm[f'Right_x{i}'] -= base_x_r
            row_norm[f'Right_y{i}'] -= base_y_r
            row_norm[f'Right_z{i}'] -= base_z_r
            
    return row_norm

def train_gesture_model(csv_path='Data/hand_landmarks_data.csv', model_path='models/gesture_model.joblib', encoder_path='Models/label_encoder.joblib'):
    if not os.path.exists(csv_path):
        print(f"Erro: O arquivo {csv_path} não foi encontrado.")
        return

    print(f"--- Carregando dados de {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    X_raw = df.drop('label', axis=1)
    y = df['label']
    
    print("--- Normalizando coordenadas espaciais ---")
    X_normalized = X_raw.apply(normalize_hand_coordinates, axis=1)
    
    print(f"Total de amostras: {len(df)}")
    print(f"Gestos detectados: {y.unique()}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    print("\n--- Treinando o Modelo (Random Forest) ---")
    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    print("\n--- Relatório de Classificação ---")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print(f"\n--- Salvando modelo em {model_path} ---")
    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)
    
    print("Sucesso! O modelo está pronto para uso.")

if __name__ == "__main__":
    train_gesture_model()