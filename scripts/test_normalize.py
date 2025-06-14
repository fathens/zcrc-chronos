import datetime
import sys
import os

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.routes import normalize_time_series_data

def test_normalize_time_series_data_irregular_intervals():
    """
    normalize_time_series_data 関数の不規則な間隔のデータに対するテスト
    """
    # 不規則な間隔のデータを作成
    now = datetime.datetime.now()
    timestamps = [
        now - datetime.timedelta(hours=12),
        now - datetime.timedelta(hours=8),  # 4時間の間隔
        now - datetime.timedelta(hours=7),  # 1時間の間隔
        now - datetime.timedelta(hours=3),  # 4時間の間隔
        now,  # 3時間の間隔
    ]
    values = [10.0, 15.0, 17.0, 20.0, 25.0]

    print(f"元のデータポイント数: {len(timestamps)}")

    # 不規則間隔データに対して正規化を実行
    normalized_timestamps, normalized_values = normalize_time_series_data(
        timestamps, values
    )

    print(f"正規化後のデータポイント数: {len(normalized_timestamps)}")

    # 正規化後のデータ点が元のデータ点以上であることを確認
    assert len(normalized_timestamps) >= len(timestamps), "正規化後のデータポイント数が元のデータポイント数より少なくなっています"
    print("テスト成功: 正規化後のデータポイント数が元のデータポイント数以上です")

    # 均等な間隔かどうかを検証
    time_diffs = [
        (normalized_timestamps[i + 1] - normalized_timestamps[i]).total_seconds()
        for i in range(len(normalized_timestamps) - 1)
    ]
    max_diff = max(time_diffs)
    min_diff = min(time_diffs)
    print(f"最大時間間隔: {max_diff}秒, 最小時間間隔: {min_diff}秒")
    assert max_diff - min_diff < 1.0, "時間間隔が均等ではありません"
    print("テスト成功: 時間間隔が均等です")

if __name__ == "__main__":
    test_normalize_time_series_data_irregular_intervals()