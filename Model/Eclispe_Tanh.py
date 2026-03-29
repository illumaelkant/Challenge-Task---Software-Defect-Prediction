import pandas as pd
import numpy as np
import os
import glob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score
from catboost import CatBoostClassifier

DATA_DIRECTORY = 'eclipse_tanh(SOICT)' 
K_FEATURES_FOR_SELECTION = 15 

def main():
    # Tìm tất cả các file .csv trong thư mục dữ liệu
    all_files = glob.glob(os.path.join(DATA_DIRECTORY, '*.csv'))
    
    if not all_files:
        print(f"Không tìm thấy file CSV nào trong thư mục '{DATA_DIRECTORY}'. Vui lòng kiểm tra lại.")
        return

    for filepath in all_files:
        filename = os.path.basename(filepath)
        project_name = os.path.splitext(filename)[0] # Lấy tên dự án, ví dụ: 'EQ'
        
        print(f"\n===== Đang xử lý dự án: {project_name} =====")

        try:

            df = pd.read_csv(filepath)


            if 'class' not in df.columns:
                print(f"  -> Lỗi: Không tìm thấy cột 'class' trong file {filename}. Bỏ qua.")
                continue
            
            y = df['class'].map({'buggy': 1, 'clean': 0})
            

            X = df.drop(columns=['id', 'class'])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"  -> Kích thước tập huấn luyện: {X_train.shape[0]} mẫu")
            print(f"  -> Kích thước tập kiểm thử: {X_test.shape[0]} mẫu")


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
                "Project": project_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1 Score": f1_score(y_test, y_pred, average='weighted'),
                "ROC AUC": roc_auc_score(y_test, y_proba),
                "G-Mean": geometric_mean_score(y_test, y_pred, pos_label=1)
            }


            print("--- Kết quả đánh giá ---")
            for key, value in metrics.items():
                if key != "Project":
                    print(f"  - {key}: {value:.4f}")


            results_df = pd.DataFrame([metrics])
            output_filename = f"ket_qua_{project_name}.xlsx"
            results_df.to_excel(output_filename, index=False)
            print(f"  -> Đã lưu kết quả vào file: {output_filename}")

        except Exception as e:
            print(f"  -> Đã xảy ra lỗi khi xử lý file {filename}: {e}")

    print("\n===== Đã xử lý xong tất cả các dự án. =====")

if __name__ == '__main__':
    main()