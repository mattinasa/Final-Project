import pandas as pd
df = pd.read_parquet('yellow_tripdata_2023-01.parquet')
#print(trips.dtypes)
import pandas as pd


class TaxiDataProcessor:

    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
        self.core_fields = {
            'tpep_pickup_datetime': '上车时间',
            'tpep_dropoff_datetime': '下车时间',
            'trip_distance': '行程距离',
            'passenger_count': '乘客人数',
            'PULocationID': '上车位置',
            'DOLocationID': '下车位置',
            'fare_amount': '车费金额',
            'payment_type': '支付方式'
        }
        self.removal_stats = {}

    def check_data(self):
        core_fields = {
            'tpep_pickup_datetime': '上车时间',
            'tpep_dropoff_datetime': '下车时间',
            'trip_distance': '行程距离',
            'passenger_count': '乘客人数',
            'PULocationID': '上车位置',
            'DOLocationID': '下车位置',
            'fare_amount': '车费金额',
            'payment_type': '支付方式'
        }

        # 1. 缺失率统计
        print("=" * 60)
        print("缺失率统计")
        print("=" * 60)

        missing_df = pd.DataFrame({
            '字段': [core_fields.get(col, col) for col in core_fields.keys()],
            '缺失数量': [self.df[col].isna().sum() for col in core_fields.keys()],
            '缺失率(%)': [(self.df[col].isna().sum() / len(self.df) * 100).round(2) for col in core_fields.keys()]
        })
        print(missing_df.to_string(index=False))

        # 2. 异常值统计
        print("\n" + "=" * 60)
        print("异常值统计")
        print("=" * 60)

        # 计算行程时长（小时）
        duration_hours = (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 3600

        anomaly_data = []

        # 检查各项异常
        anomaly_data.append(['行程时长', '负数', (duration_hours < 0).sum()])
        anomaly_data.append(['行程时长', '>24小时', (duration_hours > 24).sum()])
        anomaly_data.append(['行程距离', '负数', (self.df['trip_distance'] < 0).sum()])
        anomaly_data.append(['行程距离', '0', (self.df['trip_distance'] == 0).sum()])
        anomaly_data.append(['行程距离', '>100英里', (self.df['trip_distance'] > 100).sum()])
        anomaly_data.append(['乘客人数', '负数', (self.df['passenger_count'] < 0).sum()])
        anomaly_data.append(['乘客人数', '0', (self.df['passenger_count'] == 0).sum()])
        anomaly_data.append(['乘客人数', '>6人', (self.df['passenger_count'] > 6).sum()])
        anomaly_data.append(['车费金额', '负数', (self.df['fare_amount'] < 0).sum()])
        anomaly_data.append(['车费金额', '0', (self.df['fare_amount'] == 0).sum()])
        anomaly_data.append(['车费金额', '>200美元', (self.df['fare_amount'] > 200).sum()])
        anomaly_data.append(
            ['支付方式', '不在1-6范围内',
             (~self.df['payment_type'].isin([1, 2, 3, 4, 5, 6]) & self.df['payment_type'].notna()).sum()])

        anomaly_df = pd.DataFrame(anomaly_data, columns=['字段', '异常类型', '数量'])
        anomaly_df['占比(%)'] = (anomaly_df['数量'] / len(self.df) * 100).round(3)
        print(anomaly_df[anomaly_df['数量'] > 0].to_string(index=False))

        # 3. 核心字段统计摘要
        print("\n" + "=" * 60)
        print("字段统计摘要")
        print("=" * 60)

        # 行程时间统计
        print(
            f"\n行程时长(小时): 均值={duration_hours[duration_hours > 0].mean():.2f}, 中位数={duration_hours[duration_hours > 0].median():.2f}")

        # 行程距离统计
        valid_dist = self.df['trip_distance'][self.df['trip_distance'] > 0]
        print(f"行程距离(英里): 均值={valid_dist.mean():.2f}, 中位数={valid_dist.median():.2f}")

        # 乘客人数统计
        valid_pax = self.df['passenger_count'][(self.df['passenger_count'] >= 1) & (self.df['passenger_count'] <= 6)]
        print(f"乘客人数(人): 均值={valid_pax.mean():.2f}, 中位数={valid_pax.median():.2f}")

        # 车费统计
        valid_fare = self.df['fare_amount'][self.df['fare_amount'] > 0]
        print(f"车费金额(美元): 均值={valid_fare.mean():.2f}, 中位数={valid_fare.median():.2f}")

        # 支付方式分布
        print(f"\n支付方式分布:")
        payment_map = {1: '信用卡', 2: '现金', 3: '免单', 4: '争议', 5: '未知', 6: '欠款'}
        payment_counts = self.df['payment_type'].map(payment_map).value_counts()
        for k, v in payment_counts.items():
            print(f"  {k}: {v:,} ({(v / len(self.df) * 100):.1f}%)")