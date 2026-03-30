import pandas as pd
import os
import glob
import re
from collections import defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.metrics import geometric_mean_score

# --- CẤU HÌNH ---
DATA_DIRECTORY = 'data'
OUTPUT_FILE = 'ket_qua_PROMISE_RandomForest.xlsx'
K_FEATURES_FOR_SELECTION = 15


def get_project_files(directory):
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    projects = defaultdict(list)

    def version_key(filename):
        match = re.search(r'-([\d\.]+)\.csv', os.path.basename(filename))
        if match:
            return tuple(map(int, match.group(1).split('.')))
        return tuple()

    for f in csv_files:
        project_name = os.path.basename(f).split('-')[0]
        projects[project_name].append(f)

    for name in projects:
        projects[name].sort(key=version_key)

    return projects


def preprocess_dataframe(df):
    df_copy = df.copy()
    if 'bug' in df_copy.columns:
        df_copy['bug'] = df_copy['bug'].apply(lambda x: 1 if x > 0 else 0)
    return df_copy


def evaluate_model(y_true, y_pred, y_proba):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_proba)
    gmean = geometric_mean_score(y_true, y_pred, pos_label=1)

    return {
        'Accuracy': accuracy,
        'F1 Score': f1,
        'ROC AUC': roc_auc,
        'G-Mean': gmean,
    }


def main():
    project_files = get_project_files(DATA_DIRECTORY)
    all_project_avg_results = []

    for project_name, files in project_files.items():
        if len(files) < 2:
            print(f"Bỏ qua dự án '{project_name}' vì có ít hơn 2 phiên bản.")
            continue

        print(f"\n===== Đang xử lý dự án: {project_name} =====")
        run_results = []

        for i in range(len(files) - 1):
            train_file = files[i]
            test_file = files[i + 1]

            train_version = os.path.basename(train_file).replace('.csv', '')
            test_version = os.path.basename(test_file).replace('.csv', '')
            print(f"  -> Huấn luyện trên: {train_version}, Kiểm thử trên: {test_version}")

            df_train_raw = preprocess_dataframe(pd.read_csv(train_file))
            df_test_raw = preprocess_dataframe(pd.read_csv(test_file))

            X_train = df_train_raw.drop(columns=['name', 'bug'])
            y_train = df_train_raw['bug']
            X_test = df_test_raw.drop(columns=['name', 'bug'])
            y_test = df_test_raw['bug']

            minority_class_count_train = y_train.value_counts().min()
            smote_k_neighbors = 5 if minority_class_count_train > 5 else max(1, minority_class_count_train - 1)
            k_features = min(K_FEATURES_FOR_SELECTION, X_train.shape[1])

            pipeline_rf = ImbPipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(score_func=f_classif, k=k_features)),
                ('smote', SMOTE(random_state=42, k_neighbors=smote_k_neighbors)),
                ('classifier', RandomForestClassifier(
                    n_estimators=500,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced_subsample',
                )),
            ])

            pipeline_rf.fit(X_train, y_train)
            y_pred = pipeline_rf.predict(X_test)
            y_proba = pipeline_rf.predict_proba(X_test)[:, 1]

            metrics = evaluate_model(y_test, y_pred, y_proba)
            run_results.append(metrics)
            print(f"     Kết quả: F1={metrics['F1 Score']:.4f}, ROC_AUC={metrics['ROC AUC']:.4f}")

        if run_results:
            df_run_results = pd.DataFrame(run_results)
            avg_metrics = df_run_results.mean().to_dict()
            avg_metrics['Project'] = project_name
            all_project_avg_results.append(avg_metrics)
            print(f"----- Kết quả trung bình cho dự án '{project_name}' -----")
            print(df_run_results.mean())

    if all_project_avg_results:
        final_df = pd.DataFrame(all_project_avg_results)
        cols = ['Project', 'Accuracy', 'F1 Score', 'ROC AUC', 'G-Mean']
        final_df = final_df[cols]
        final_df.to_excel(OUTPUT_FILE, index=False)
        print(f"\n\nHoàn thành! Kết quả đã được lưu vào file: {OUTPUT_FILE}")
    else:
        print("\n\nKhông có dự án nào được xử lý. Vui lòng kiểm tra lại cấu trúc thư mục.")


if __name__ == '__main__':
    main()
