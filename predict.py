#リアルタイムMW予測
from pythonosc import dispatcher, osc_server
import numpy as np
import threading
import time
import os
import pandas as pd
import time
from datetime import datetime
import joblib

# 保存されたモデルを読み込み
classifier = joblib.load('model.pkl')

#Network Variables
ip = "0.0.0.0"
port = 5000

#Muse Variables
hsi = [4,4,4,4]
hsi_string = ""

#絶対値の初期化
abs_waves = [[], [], [], [], []]

#初期値
last_sound_time = None  # 最後の音声出力時刻を記録する変数
min_interval = 1 # 音声出力間の最小時間間隔（秒）
first_run = True  # 初回実行フラグ

#特徴抽出関数(全電極使う場合！！)
def extract_features(data):
    features = []
    # 各脳波帯ごとにループ
    for wave_data in data:
        # 各電極ごとにデータを分割
        for electrode in range(4):  # 4つの電極
            electrode_data = [wave_data[i] for i in range(electrode, len(wave_data), 4)]
            # 統計量を計算(4値x4電極x5周波数帯=80特徴)
            mean = np.mean(electrode_data)
            std = np.std(electrode_data)
            max_val = np.max(electrode_data)
            min_val = np.min(electrode_data)
            # 特徴リストに追加
            features.extend([mean, std, max_val, min_val])
    return features

"""
#特徴抽出関数(条件分岐)
def extract_features(data, is_average_only):
    features = []
    
    if is_average_only:
        # 電極の平均値のみの場合の処理
        for wave_value in data:
            mean = np.mean(wave_value)
            std = np.std(wave_value)
            max_val = np.max(wave_value)
            min_val = np.min(wave_value)
            features.extend([mean, std, max_val, min_val])
    else:
        # 各電極値を含む場合の処理
        for wave_data in data:
            for electrode in range(4):
                electrode_data = [wave_data[i] for i in range(electrode, len(wave_data), 4)]
                mean = np.mean(electrode_data)
                std = np.std(electrode_data)
                max_val = np.max(electrode_data)
                min_val = np.min(electrode_data)
                features.extend([mean, std, max_val, min_val])
    return features
"""
#Muse Data handlers
def hsi_handler(address: str,*args):
    global hsi, hsi_string
    hsi = args
    if ((args[0]+args[1]+args[2]+args[3])==4):
        hsi_string_new = "Muse Fit Good"
    else:
        hsi_string_new = "Muse Fit Bad on: "
        if args[0]!=1:
            hsi_string_new += "Left Ear. "
        if args[1]!=1:
            hsi_string_new += "Left Forehead. "
        if args[2]!=1:
            hsi_string_new += "Right Forehead. "
        if args[3]!=1:
            hsi_string_new += "Right Ear."
    if hsi_string!=hsi_string_new:
        hsi_string = hsi_string_new
        print(hsi_string)
"""
#絶対値データの処理
def abs_handler(address: str, *args):
    global hsi, abs_waves
    wave = args[0][0]
    new_data = None
    #print("Received args:", args)
    #print("Current state of abs_waves:", abs_waves)

    # 有効なセンサーが少なくとも1つある場合
    if (hsi[0] == 1 or hsi[1] == 1 or hsi[2] == 1 or hsi[3] == 1):
        # OSCストリームから平均値のみを受信する場合
        if (len(args) == 2):
            abs_waves[wave].append(args[1])
            #print("Received args:", args)
            #print("Updated abs_waves:", abs_waves)
            is_average_only = True
        # OSCストリームからすべての値を受信する場合
        elif (len(args) == 5):
            for i in [0, 1, 2, 3]:
                if hsi[i] == 1:  # 有効なセンサーのみを使用
                    abs_waves[wave].append(args[i + 1])
            is_average_only = False

        # 特徴抽出とMW予測の実行
        if is_average_only and all(len(abs_wave) >= 10 for abs_wave in abs_waves):
            new_data = [abs_wave[-10:] for abs_wave in abs_waves]
        elif not is_average_only and all(len(abs_wave) >= 40 for abs_wave in abs_waves):
            new_data = [abs_wave[-40:] for abs_wave in abs_waves]

        if new_data:
            features = extract_features(new_data, is_average_only)

            prediction = classifier.predict([features])
            # 音声出力
            if prediction == [1]:
                os.system('afplay -t 0.1 B.mp3')
"""
check_count = 0  # 初期化
last_processed_count = [0, 0, 0, 0, 0] 

def abs_handler(address: str, *args):
    global abs_waves, last_sound_time, first_run, last_processed_count, check_count
    wave = args[0][0]
    #print("Received args:", args)
    #print("Current state of abs_waves:", abs_waves)

    # OSCストリームから平均値のみを受信する場合
    if (len(args) == 2):
        abs_waves[wave].append(args[1])
    # OSCストリームからすべての値を受信する場合
    elif len(args) == 5:
        for i in range(4):
            abs_waves[wave].append(args[i + 1])

    #if check_count < 10 and all(len(abs_wave) > 40 for abs_wave in abs_waves):
        #print([len(abs_wave) for abs_wave in abs_waves])  # 各周波数帯のデータ点の数を表示
        #check_count += 1

    # 新しいデータが各周波数帯に160個追加されたかチェック
    if all(len(abs_wave) - last_count >= 80 for abs_wave, last_count in zip(abs_waves, last_processed_count)):
        #最初にdeltaの長さを出力
        if first_run and len(abs_waves[wave]) >= 80:
            print("Delta wave data length:", len(abs_waves[wave]))
            #first_run = False  # フラグを更新
        abs_waves = [abs_wave[-80:] for abs_wave in abs_waves]
        features = extract_features(abs_waves)
        prediction = classifier.predict([features])
        print("Prediction:", prediction)
        last_processed_count = [len(abs_wave) for abs_wave in abs_waves]

        # 音声出力
        if prediction == [1]:
            os.system('afplay -t 0.02 B.mp3')


#Main
if __name__ == "__main__":
  # OSCサーバーを別スレッドで実行
  server_thread = threading.Thread()
  server_thread.daemon = True  
  server_thread.start()

  # OSCサーバーの設定
  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/muse/elements/horseshoe", hsi_handler)
  dispatcher.map("/muse/elements/delta_absolute", abs_handler,0)
  dispatcher.map("/muse/elements/theta_absolute", abs_handler,1)
  dispatcher.map("/muse/elements/alpha_absolute", abs_handler,2)
  dispatcher.map("/muse/elements/beta_absolute", abs_handler,3)
  dispatcher.map("/muse/elements/gamma_absolute", abs_handler,4)

  #サーバー設定
  server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
  print("Listening on UDP port "+str(port))
  server.serve_forever()
