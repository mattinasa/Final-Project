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
        self.peak_hours = None
        self.hour_counts = None

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

    def clean(self):  # 清洗数据策略：缺失值用平均数来填充，异常值直接删除
        print("\n" + "=" * 60)
        print("开始数据清洗")
        print("=" * 60)
        print(f"清洗前记录数: {len(self.df):,}")

        before = len(self.df)

        duration_hours = (self.df['tpep_dropoff_datetime'] - self.df['tpep_pickup_datetime']).dt.total_seconds() / 3600

        # 记录各类异常删除数量
        anomaly_stats = {}

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

    def extract_time_features(self):
        """提取时间特征"""
        print("\n" + "=" * 60)
        print("提取时间特征")
        print("=" * 60)

        pickup = self.df['tpep_pickup_datetime']
        dropoff = self.df['tpep_dropoff_datetime']

        # 基础时间特征
        self.df['pickup_hour'] = pickup.dt.hour
        self.df['pickup_weekday'] = pickup.dt.dayofweek
        self.df['is_weekend'] = (pickup.dt.dayofweek >= 5).astype(int)

        # 行程时长
        duration_hours = (dropoff - pickup).dt.total_seconds() / 3600
        self.df['trip_duration_hours'] = duration_hours
        self.df['trip_duration_minutes'] = duration_hours * 60

        # 平均速度
        self.df['avg_speed_mph'] = 0.0
        mask = self.df['trip_duration_hours'] > 0.01
        self.df.loc[mask, 'avg_speed_mph'] = self.df.loc[mask, 'trip_distance'] / self.df.loc[
            mask, 'trip_duration_hours']
        self.df.loc[self.df['avg_speed_mph'] > 100, 'avg_speed_mph'] = 100

        # 收入时间密度
        self.df['revenue_density_per_hour'] = 0.0
        self.df.loc[mask, 'revenue_density_per_hour'] = self.df.loc[mask, 'total_amount'] / self.df.loc[
            mask, 'trip_duration_hours']

        print(
            f"已添加特征: pickup_hour, pickup_weekday, is_weekend, trip_duration_hours, avg_speed_mph, revenue_density_per_hour")
        print("=" * 60)

        return self.df

    def detect_peak_hours(self):
        """基于已有数据进行高峰时段的提取，取行程量最高的两个小时作为高峰时段"""
        print("\n" + "=" * 60)
        print("高峰时段提取")
        print("=" * 60)

        # 统计每个小时的行程数（基于上车时间）
        hour_counts = self.df['pickup_hour'].value_counts().sort_index()

        # 补充缺失的小时
        for h in range(24):
            if h not in hour_counts.index:
                hour_counts[h] = 0
        hour_counts = hour_counts.sort_index()

        # 打印各小时行程数分布
        print(f"\n各小时行程数统计:")
        max_count = hour_counts.max()
        for hour in range(24):
            count = hour_counts[hour]
            bar_len = int(count / max_count * 40)
            bar = '█' * bar_len
            print(f"  {hour:2d}时: {count:8,} {bar}")

        # 找出最高的两个小时
        top2 = hour_counts.nlargest(2)
        peak_hours = top2.index.tolist()
        peak_counts = top2.values.tolist()

        print(f"\n最高两个高峰时段:")
        for i, (hour, count) in enumerate(zip(peak_hours, peak_counts), 1):
            print(f"  第{i}名: {hour:2d}:00 - {hour + 1:2d}:00, 行程数: {count:,}")

        # 存储结果
        self.peak_hours = peak_hours
        self.hour_counts = hour_counts.to_dict()

        print("=" * 60)

        return peak_hours


def main():
    # 读取数据
    trips = pd.read_parquet('yellow_tripdata_2023-01.parquet')

    # 初始化处理器
    processor = TaxiDataProcessor(trips)

    # 1. 检查数据质量
    processor.check_data()

    # 2. 清洗数据
    processor.clean()

    # 3. 提取时间特征
    processor.extract_time_features()

    # 4. 检测高峰时段
    processor.detect_peak_hours()

    # 5. 设置高峰标识
    processor.df['is_peak_hour'] = processor.df['pickup_hour'].isin(processor.peak_hours).astype(int)

    # 6. 获取最终数据
    trips_processed = processor.get_cleaned_data()

    print(f"\n最终数据形状: {trips_processed.shape}")
    print(
        f"高峰时段: {processor.peak_hours[0]}:00-{processor.peak_hours[0] + 1}:00 和 {processor.peak_hours[1]}:00-{processor.peak_hours[1] + 1}:00")
    print(f"\n处理完成！")

    return processor


if __name__ == "__main__":
    main()