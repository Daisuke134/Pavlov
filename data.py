import pandas as pd
import glob
from datetime import timedelta

# 必要な列
columns = [
    'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
    'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
    'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
    'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
    'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10'
]

#時間窓抽出(４秒間)
def extract_data_segments(df, idx, sampling_rate):
    mw_segment = df.loc[max(idx - 5 * sampling_rate, 0):max(idx - 1 * sampling_rate, 0), columns]
    f_segment = df.loc[min(idx + 1, len(df) - 1):min(idx + 4 * sampling_rate, len(df) - 1), columns]
    return mw_segment, f_segment

# CSVファイルのパスを取得
csv_files = glob.glob('training_data/*.csv')

# 各ファイルを処理
for file in csv_files:
    df = pd.read_csv(file)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    marker_indices = df[df['Elements'].str.contains('/Marker/1', na=False)].index

    pasted_indices = []
    for idx in marker_indices:
        if idx > 0:
            df.loc[idx - 1, 'Elements'] = df.loc[idx, 'Elements']
            pasted_indices.append(idx - 1)

    df = df.drop(marker_indices)

    #ペーストされた行の前後2行を表示
    #for idx in pasted_indices:
        #display_range = df.loc[max(idx-2, 0):min(idx+2, len(df)-1)]
        #print(display_range)

    # 26行ごとにデータを抽出し、/Marker/1を含む行も保持
    selected_indices = set(range(0, len(df), 26)).union(set(pasted_indices))
    df_reduced = df.loc[sorted(selected_indices)]

    # インデックスをリセット
    df_reduced = df_reduced.reset_index(drop=True)

    mw_segments = []
    f_segments = []

    # セグメントを抽出
    for idx in df_reduced[df_reduced['Elements'].str.contains('/Marker/1', na=False)].index:
        mw_segment, f_segment = extract_data_segments(df_reduced, idx, 10)
        mw_segments.append(mw_segment)
        f_segments.append(f_segment)

    # MWとFセグメントをテキストファイルとして保存
    for i, segment in enumerate(mw_segments):
        segment.to_csv(f'output/MW_{i}.txt', index=False)
    for i, segment in enumerate(f_segments):
        segment.to_csv(f'output/F_{i}.txt', index=False)

    print(f"File: {file} - MW Segments: {len(mw_segments)}, F Segments: {len(f_segments)}")






