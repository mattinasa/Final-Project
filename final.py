import pandas as pd
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

    def clean(self):#清洗数据策略：缺失值用平均数来填充，异常值直接删除
        print("=" * 60)
        print("开始数据清洗")
        print("=" * 60)
        print(f"清洗前记录数: {len(self.df):,}")

        before = len(self.df)

        duration_hours = (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 3600

        # 记录各类异常删除数量
        anomaly_stats = {}  # 改为英文变量名

        # 1. 删除行程时长异常
        mask_duration = (duration_hours >= 0) & (duration_hours <= 24)
        anomaly_stats['行程时长异常'] = (~mask_duration).sum()
        self.df = self.df[mask_duration]

        # 2. 删除行程距离异常
        mask_distance = (self.df['trip_distance'] > 0) & (self.df['trip_distance'] <= 100)
        anomaly_stats['行程距离异常'] = (~mask_distance).sum()
        self.df = self.df[mask_distance]

        # 3. 删除乘客人数异常
        mask_passenger_abnormal = (self.df['passenger_count'] < 1) | (self.df['passenger_count'] > 6)
        anomaly_stats['乘客人数异常'] = mask_passenger_abnormal.sum()
        self.df = self.df[~mask_passenger_abnormal]

        # 4. 删除车费金额异常
        mask_fare = (self.df['fare_amount'] >= 0) & (self.df['fare_amount'] <= 200)
        anomaly_stats['车费金额异常'] = (~mask_fare).sum()
        self.df = self.df[mask_fare]

        # 5. 删除支付方式异常
        mask_payment = self.df['payment_type'].isin([1, 2, 3, 4, 5, 6])
        anomaly_stats['支付方式异常'] = (~mask_payment).sum()
        self.df = self.df[mask_payment]

        # 处理缺失值（用正常数据的平均值填充）
        mean_passenger = self.df['passenger_count'].mean()
        missing_passenger = self.df['passenger_count'].isna().sum()
        self.df['passenger_count'] = self.df['passenger_count'].fillna(mean_passenger)

        mean_distance = self.df['trip_distance'].mean()
        self.df['trip_distance'] = self.df['trip_distance'].fillna(mean_distance)

        mean_fare = self.df['fare_amount'].mean()
        self.df['fare_amount'] = self.df['fare_amount'].fillna(mean_fare)

        # 记录统计
        total_deleted = sum(anomaly_stats.values())

        self.removal_stats = {
            '删除异常记录': total_deleted,
            '填充乘客人数缺失': missing_passenger,
            '保留比例': f"{len(self.df) / before * 100:.2f}%"
        }

        print(f"\n异常删除统计:")
        for k, v in anomaly_stats.items():
            if v > 0:
                print(f"  {k}: {v:,} 条")
        print(f"\n缺失值填充:")
        print(f"  乘客人数: {missing_passenger:,} 条 (均值={mean_passenger:.2f})")
        print(f"\n清洗后记录数: {len(self.df):,}")
        print(f"保留比例: {self.removal_stats['保留比例']}")
        print("=" * 60)

        return self.df

    def get_cleaned_data(self):
        """获取清洗后的数据"""
        return self.df

    def get_removal_stats(self):
        """获取删除统计信息"""

        return self.removal_stats

def main():
    # 初始化
    trips = pd.read_parquet('yellow_tripdata_2023-01.parquet')
    processor = TaxiDataProcessor(trips)

    # 检查和清洗
    processor.check_data()  # 生成报告
    processor.clean()  # 清洗数据

    # 获取结果
    trips_clean = processor.get_cleaned_data()

    print(f"\n原始数据: {len(trips)} 条")
    print(f"清洗后: {len(trips_clean)} 条")


if __name__ == "__main__":
    main()