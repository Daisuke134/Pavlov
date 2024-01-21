# モデルの訓練と評価
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, confusion_matrix
import joblib

def extract_features(df):
    mean_signal = df.mean(axis=0)
    std_signal = df.std(axis=0)
    max_signal = df.max(axis=0)
    min_signal = df.min(axis=0)
    return np.concatenate((mean_signal, std_signal, max_signal, min_signal))

# Extracting features and labels
def load_data(files, label):
    feature_list = []
    for file in files:
        df = pd.read_csv(file, delimiter=',')
        features = extract_features(df)
        feature_list.append(features)
    labels = [label] * len(feature_list)
    return np.array(feature_list), np.array(labels)

# Loading data for F and MW
f_files = glob.glob('output/F_*.txt')
mw_files = glob.glob('output/MW_*.txt')

X_f, y_f = load_data(f_files, 0)
X_mw, y_mw = load_data(mw_files, 1)

# Combining the data
X = np.concatenate((X_f, X_mw), axis=0)
y = np.concatenate((y_f, y_mw), axis=0)

# 10-fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Random Forest Classifier
classifier = RandomForestClassifier(random_state=42)

# 性能指標を計算する関数
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# 性能指標
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score),
           'recall': make_scorer(recall_score),
           'specificity': make_scorer(specificity_score)}

# モデルの評価
scores = cross_validate(classifier, X, y, scoring=scoring, cv=cv, return_train_score=True)

def to_percentage(score):
    return round(score * 100, 1)

print(f"Average Train Accuracy: {to_percentage(np.mean(scores['train_accuracy']))}%")
print(f"Average Test Accuracy: {to_percentage(np.mean(scores['test_accuracy']))}%")
print(f"Average Precision: {to_percentage(np.mean(scores['test_precision']))}%")
print(f"Average Recall: {to_percentage(np.mean(scores['test_recall']))}%")
print(f"Average Specificity: {to_percentage(np.mean(scores['test_specificity']))}%")

#モデルの訓練
classifier.fit(X, y)
joblib.dump(classifier, 'model.pkl')

