import pandas as pd
import os

# CSVファイルのリストを作成
csv_files = ['mindNEW.csv']

output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 各CSVファイルをループで処理
for file_name in csv_files:
    df = pd.read_csv(file_name)

    # サンプリングレート
    sampling_rate = 10  # Hz

    # MWとFの時間窓を定義（サンプル単位）
    mw_start_offset = -10 * sampling_rate  # MWイベントの10秒前
    mw_end_offset = -2 * sampling_rate     # MWイベントの2秒前
    f_start_offset = 0                     # Fイベントの開始時
    f_end_offset = 8 * sampling_rate       # Fイベントの8秒後

    # EventID列に基づいてイベントのインデックスを取得
    mw_event_indices = df[df['EventID'] == 30].index
    f_event_indices = df[df['EventID'] == 50].index

    # 列名
    columns = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
               'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
               'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
               'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
               'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10']

## 特定のパターンにマッチする列名だけを取り出す
#pattern = 'Delta_|Theta_|Alpha_|Beta_|Gamma_'
#columns = [col for col in df.columns if any(pat in col for pat in pattern.split('|'))]

        # データセグメントの抽出と保存の関数
    def extract_data_segments(indices, start_offset, end_offset, filename_prefix):
        segments = []
        for index in indices:
            # インデックスが範囲外にならないようにチェック
            if (index + start_offset) >= 0 and (index + end_offset) < len(df):
                segment = df.loc[index + start_offset:index + end_offset - 1, columns]
                segments.append(segment)
                output_file = f"{filename_prefix}.{len(segments)}.txt"
                segment.to_csv(os.path.join(output_dir, output_file), index=False)
        return segments

    # MWとFのデータセグメントを抽出して保存
    mw_segments = extract_data_segments(mw_event_indices, mw_start_offset, mw_end_offset, "MW")
    f_segments = extract_data_segments(f_event_indices, f_start_offset, f_end_offset, "F")

    # 抽出されたセグメントの数を確認
    print(f"File: {file_name} - MW Segments: {len(mw_segments)}, F Segments: {len(f_segments)}")














#モデルの訓練＋評価
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, confusion_matrix

def extract_features(df):
    mean_signal = df.mean(axis=0)
    std_signal = df.std(axis=0)
    max_signal = df.max(axis=0)
    min_signal = df.min(axis=0)
    return np.concatenate((mean_signal, std_signal, max_signal, min_signal))

# 特徴量とラベルの抽出
def load_data(files, label):
    feature_list = []
    for file in files:
        df = pd.read_csv(file, delimiter=',')
        features = extract_features(df)
        feature_list.append(features)
    labels = [label] * len(feature_list)
    return np.array(feature_list), np.array(labels)

# FとMWのデータを読み込む
f_files = glob.glob('output/F.*.txt')
mw_files = glob.glob('output/MW.*.txt')

X_f, y_f = load_data(f_files, 0)
X_mw, y_mw = load_data(mw_files, 1)

# データを結合
X = np.concatenate((X_f, X_mw), axis=0)
y = np.concatenate((y_f, y_mw), axis=0)

# 10-fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ランダムフォレスト分類器
classifier = RandomForestClassifier(random_state=42)

# 性能指標を計算する関数
def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

# 性能指標
scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, zero_division=1),
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

#訓練
classifier.fit(X, y)






"""
from pythonosc import dispatcher
from pythonosc import osc_server
import numpy as np
import threading
import time
import os
import pandas as pd
import time
from datetime import datetime

#特徴抽出関数
def extract_features(data):
    # ここに特徴抽出のロジックを実装します。
    df = pd.DataFrame(data, columns=['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
                                     'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
                                     'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
                                     'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
                                     'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10'])
    mean_signal = df.mean(axis=0)
    std_signal = df.std(axis=0)
    max_signal = df.max(axis=0)
    min_signal = df.min(axis=0)
    return np.concatenate((mean_signal, std_signal, max_signal, min_signal))

# CSVファイルのパスを設定
csv_file_path = 'CSV-file.csv'
f = open(csv_file_path, 'w+')

# 最後に処理した行のインデックスを保持する変数
last_processed_index = -1

# データ記録の開始と停止を制御するためのブール変数
recording = True
deltaReceived = thetaReceived = alphaReceived = betaReceived = gammaReceived = False

# 列名をCSVファイルのヘッダーに書き込む関数
def writeFileHeader():
    #ここどうするか？
    #with open(csv_file_path, 'w') as f:
    headerString = "TimeStamp,Delta_TP9,Delta_AF7,Delta_AF8,Delta_TP10," \
                 "Theta_TP9,Theta_AF7,Theta_AF8,Theta_TP10," \
                 "Alpha_TP9,Alpha_AF7,Alpha_AF8,Alpha_TP10," \
                 "Beta_TP9,Beta_AF7,Beta_AF8,Beta_TP10," \
                 "Gamma_TP9,Gamma_AF7,Gamma_AF8,Gamma_TP10\n"
    f.write(headerString)
    #f.flush()

# データ受信時に呼び出される関数
def eeg_handler(address: str, *args):
    global last_processed_index
    global deltaReceived, thetaReceived, alphaReceived, betaReceived, gammaReceived

    # ヘッダーを書き込む
    if not (deltaReceived and thetaReceived and alphaReceived and betaReceived and gammaReceived):
        writeFileHeader()

    if recording:
        # CSVに書き込むためのデータ文字列を作成
        timestampStr = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        fileString = timestampStr + "," + ",".join(map(str, args)) + "\n"

        # すべての波形データが受信されたかチェック
        if deltaReceived and thetaReceived and alphaReceived and betaReceived and gammaReceived:
            #ここどうするか？
            #with open(csv_file_path, 'a') as f:
            f.write(fileString)
            #f.flush() 
            # フラグをリセット
            deltaReceived = thetaReceived = alphaReceived = betaReceived = gammaReceived = False

# 無限ループでCSVファイルを監視
while recording:
    # CSVファイルを読み込む
    df = pd.read_csv(csv_file_path)

    # 新しく追加された行が10行以上あるかチェック
    if len(df) - last_processed_index > 10:
        # 最新の10行を取得
        new_data = df.iloc[last_processed_index + 1:last_processed_index + 11]

        # 特徴抽出と予測
        features = extract_features(new_data)
        prediction = classifier.predict([features])
        print("予測:", prediction)

        if prediction == [1]:
            os.system('afplay -t 0.2 B.mp3')

        # 処理済み行のインデックスを更新
        last_processed_index += 10

    # 1秒待機
    time.sleep(0.5)

#モデルの訓練
classifier.fit(X, y)

# OSCサーバーの設定
dispatcher = dispatcher.Dispatcher()
#dispatcher.map("/muse/eeg", handle_eeg, "EEG")
dispatcher.map("/muse/elements/delta_absolute", eeg_handler,0)
dispatcher.map("/muse/elements/theta_absolute", eeg_handler,1)
dispatcher.map("/muse/elements/alpha_absolute", eeg_handler,2)
dispatcher.map("/muse/elements/beta_absolute", eeg_handler,3)
dispatcher.map("/muse/elements/gamma_absolute", eeg_handler,4)

server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 5000), dispatcher)

# OSCサーバーを別スレッドで実行
server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()


#ここからは余分
#すべての周波数で10以上のデータがある
def plot_update(i):
    global plot_data
    global alpha_sound_threshold

    # 全ての脳波帯でデータポイントが10以上あるか確認
    for wave_data in plot_data:
        if len(wave_data) < 10:
            return  # 10未満のデータポイントを持つ脳波帯があれば関数を中止

    plt.cla()
    # その他のグラフ更新処理...

if all(len(wave_data) > 10 for wave_data in plot_data) and wave == 2:
    test_alpha_relative()


#絶対値の平均を使う場合
def abs_handler(address: str,*args):
    global hsi, abs_waves, rel_waves
    wave = args[0][0]
    
    #If we have at least one good sensor
    if (hsi[0]==1 or hsi[1]==1 or hsi[2]==1 or hsi[3]==1):
        if (len(args)==2): #If OSC Stream Brainwaves = Average Onle
            abs_waves[wave] = args[1] #Single value for all sensors, already filtered for good data
        if (len(args)==5): #If OSC Stream Brainwaves = All Values
            sumVals=0
            countVals=0            
            for i in [0,1,2,3]:
                if hsi[i]==1: #Only use good sensors
                    countVals+=1
                    sumVals+=args[i+1]
            abs_waves[wave] = sumVals/countVals


        if all(len(plot_data[i]) > 10 for i in range(5)):
            # 新しいデータの抽出
            new_data = [plot_data[i][-10:] for i in range(5)]  # 各波の最新10個のデータを取得

            # 特徴抽出と予測
            features = extract_features(new_data)
            prediction = classifier.predict([features])

            # 予測が1の場合、音を再生
            if prediction == [1]:
                os.system('afplay -t 0.2 B.mp3')

"""