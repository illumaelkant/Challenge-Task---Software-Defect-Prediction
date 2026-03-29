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

DATA_DIRECTORY = 'dataNASA'
OUTPUT_FILE = 'NASA.xlsx'

N_SPLITS_CV = 5

K_FEATURES_FOR_SELECTION = 15

def main():
    all_files = glob.glob(os.path.join(DATA_DIRECTORY, '*.csv'))
    
    if not all_files:
        print(f"Không tìm thấy file CSV nào trong thư mục '{DATA_DIRECTORY}'. Vui lòng kiểm tra lại.")
        return

    all_project_results = []

    for filepath in all_files:
        filename = os.path.basename(filepath)
        project_name = os.path.splitext(filename)[0]
        
        print(f"\n===== Đang xử lý dự án: {project_name} (sử dụng {N_SPLITS_CV}-fold Cross-Validation) =====")

        try:

            df = pd.read_csv(filepath)



            if 'Defective' not in df.columns or 'id' not in df.columns:
                print(f"  -> Lỗi: File {filename} thiếu cột 'id' hoặc 'Defective'. Bỏ qua.")
                continue


            y = df['Defective'].map({'N': 0, 'Y': 1})
            

            X = df.drop(columns=['id', 'Defective'])


            skf = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=42)
            fold_results = []

            if np.min(np.bincount(y)) < N_SPLITS_CV:
                print(f"  -> CẢNH BÁO: Bỏ qua dự án '{project_name}' vì số lượng mẫu của lớp thiểu số ({np.min(np.bincount(y))}) nhỏ hơn số fold ({N_SPLITS_CV}).")
                continue

            for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                print(f"  -> Đang chạy Fold {fold + 1}/{N_SPLITS_CV}...")
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
                df_fold_results = pd.DataFrame(fold_results)
                avg_metrics = df_fold_results.mean().to_dict()
                avg_metrics['Project'] = project_name
                all_project_results.append(avg_metrics)

                print(f"--- Kết quả trung bình cho dự án '{project_name}' ---")
                print(df_fold_results.mean())

        except Exception as e:
            print(f"  -> Đã xảy ra lỗi khi xử lý dự án {project_name}: {e}")


    if all_project_results:
        final_df = pd.DataFrame(all_project_results)
        cols = ['Project', 'Accuracy', 'F1 Score', 'ROC AUC', 'G-Mean']
        final_df = final_df[cols]
        final_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\n\nHoàn thành! Toàn bộ kết quả đã được lưu vào file: {OUTPUT_FILE}")
    else:
        print("\n\nKhông có dự án nào được xử lý. Vui lòng kiểm tra lại.")

if __name__ == '__main__':
    main()