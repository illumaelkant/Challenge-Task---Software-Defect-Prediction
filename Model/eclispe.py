import pandas as pd
import numpy as np
import os
import glob

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score
from catboost import CatBoostClassifier


DATA_DIRECTORY = 'eclipse_atho' 
N_SPLITS_CV = 5
K_FEATURES_FOR_SELECTION = 15

def preprocess_dataframe(df):
    df_copy = df.copy()
    
    if df_copy.shape[1] < 3:
        return None


    first_feature_col_idx = 2
    target_col_idx = df_copy.shape[1] - 1
    

    for col in range(first_feature_col_idx, target_col_idx):

        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    df_copy[first_feature_col_idx] = df_copy[first_feature_col_idx].apply(lambda x: 1 if x > 1 else x)
    

    df_copy[target_col_idx] = pd.to_numeric(df_copy[target_col_idx], errors='coerce')
    df_copy[target_col_idx] = df_copy[target_col_idx].apply(lambda x: 1 if x > 0 else 0)
    
    return df_copy

def main():
    all_files = glob.glob(os.path.join(DATA_DIRECTORY, '*'))
    
    if not all_files:
        print(f"Không tìm thấy file nào trong thư mục '{DATA_DIRECTORY}'. Vui lòng kiểm tra lại.")
        return

    for filepath in all_files:
        filename = os.path.basename(filepath)
        print(f"\n===== Đang xử lý file: {filename} (sử dụng {N_SPLITS_CV}-fold Cross-Validation) =====")

        try:
            df_raw = pd.read_csv(filepath, header=None)
        except Exception as e:
            print(f"  -> Lỗi khi đọc file {filename}: {e}")
            continue

        df_processed = preprocess_dataframe(df_raw)
        
        if df_processed is None:
            print(f"  -> Bỏ qua file {filename} vì không đủ cột để xử lý.")
            continue

        target_col_idx = df_processed.shape[1] - 1
        X = df_processed.drop(columns=[0, 1, target_col_idx])
        y = df_processed[target_col_idx]

        skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
        fold_results = []


        if np.min(np.bincount(y)) < N_SPLITS_CV:
            print(f"  -> Bỏ qua file {filename} vì số lượng mẫu của lớp thiểu số ({np.min(np.bincount(y))}) nhỏ hơn số fold ({N_SPLITS_CV}).")
            continue

        for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            minority_class_count_train = y_train.value_counts().min()
            smote_k_neighbors = 5 if minority_class_count_train > 5 else max(1, minority_class_count_train - 1)
            
            pipeline_cb = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(score_func=f_classif, k=K_FEATURES_FOR_SELECTION)),
                ('smote', SMOTE(random_state=42, k_neighbors=smote_k_neighbors)),
                ('classifier', CatBoostClassifier(iterations=1000, 
                                                   depth=4, 
                                                   learning_rate=0.1, 
                                                   verbose=0,
                                                   random_state=42))
            ])
            
            pipeline_cb.fit(X_train, y_train)
            y_pred = pipeline_cb.predict(X_test)
            y_proba = pipeline_cb.predict_proba(X_test)[:, 1]

            metrics = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred, average='weighted'),
                "ROC AUC": roc_auc_score(y_test, y_proba),
                "G-Mean": geometric_mean_score(y_test, y_pred, pos_label=1)
            }
            fold_results.append(metrics)
        
        if fold_results:
            df_results = pd.DataFrame(fold_results)
            avg_results = df_results.mean()
            
            print(f"\n--- Kết quả trung bình cho file: {filename} ---")
            print(f"  - Accuracy: {avg_results['Accuracy']:.4f}")
            print(f"  - F1 Score (weighted): {avg_results['F1 Score']:.4f}")
            print(f"  - ROC AUC : {avg_results['ROC AUC']:.4f}")
            print(f"  - G-Mean  : {avg_results['G-Mean']:.4f}")
            print("--------------------------------------------------")

    print("\n===== Đã xử lý xong tất cả các file. =====")

if __name__ == '__main__':
    main()