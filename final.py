import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

    # 时间特征的一些提取，是否高峰的判别分为两步，一个是提取高峰时间段，在另一个函数，识别高峰在主函数
    #额外增加的两个特征为：1.平均速度：用于评估路径情况 2.收入时间密度：用于衡量经济效益
    def extract_time_features(self):
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

    def plot_travel_demand_patterns(processor):
        """绘制出行需求时间规律图表"""

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 获取处理后的数据
        df = processor.get_cleaned_data()

        # 创建图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('出行需求时间规律', fontsize=16, fontweight='bold')

        # ========== 子图1：分小时平均订单量折线图 ==========
        ax1 = axes[0]

        # 统计每个小时的总订单量
        hourly_orders = df.groupby('pickup_hour').size()

        # 计算平均值作为参考线
        avg_orders = hourly_orders.mean()

        # 绘制折线图
        hours = range(24)
        ax1.plot(hours, [hourly_orders.get(h, 0) for h in hours],
                 marker='o', linewidth=2, markersize=6, color='steelblue')

        # 添加平均线
        ax1.axhline(y=avg_orders, color='red', linestyle='--', linewidth=1.5,
                    label=f'平均值: {avg_orders:,.0f}')

        # 标记高峰时段
        if hasattr(processor, 'peak_hours') and processor.peak_hours:
            for peak_hour in processor.peak_hours:
                ax1.axvline(x=peak_hour, color='orange', linestyle=':', alpha=0.7, linewidth=2)
                peak_value = hourly_orders.get(peak_hour, 0)
                ax1.scatter(peak_hour, peak_value, color='red', s=100, zorder=5)
                ax1.annotate(f'高峰\n{peak_hour}:00',
                             xy=(peak_hour, peak_value),
                             xytext=(peak_hour, peak_value + max(hourly_orders) * 0.05),
                             ha='center', fontsize=9, color='red')

        # 设置图表属性
        ax1.set_title('分小时平均订单量', fontsize=12, fontweight='bold')
        ax1.set_xlabel('小时 (24小时制)', fontsize=10)
        ax1.set_ylabel('订单数量', fontsize=10)
        ax1.set_xticks(range(0, 24, 2))
        ax1.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper right')

        # 添加数值标签（可选，只标记几个关键点）
        for hour in [7, 8, 9, 17, 18, 19] + (processor.peak_hours if hasattr(processor, 'peak_hours') else []):
            if hour in hourly_orders.index:
                ax1.annotate(f'{hourly_orders[hour]:,.0f}',
                             xy=(hour, hourly_orders[hour]),
                             xytext=(hour, hourly_orders[hour] + max(hourly_orders) * 0.01),
                             fontsize=8, ha='center')

        # ========== 子图2：分周末/工作日平均订单量柱状图 ==========
        ax2 = axes[1]

        # 创建周末/工作日标识
        df['day_type'] = df['is_weekend'].map({0: '工作日', 1: '周末'})

        # 统计周末和工作日的总订单量
        day_type_orders = df.groupby('day_type').size()

        # 确保两个类别都存在
        categories = ['工作日', '周末']
        values = [day_type_orders.get(cat, 0) for cat in categories]

        # 绘制柱状图
        bars = ax2.bar(categories, values, color=['steelblue', 'orange'],
                       edgecolor='black', linewidth=1.5, alpha=0.8)

        # 在柱子上方添加数值标签
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                     f'{value:,.0f}\n({value / len(df) * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 计算平均每小时订单量
        df['hour_type'] = df['pickup_hour'].astype(str) + '_' + df['day_type']
        hourly_by_daytype = df.groupby(['pickup_hour', 'day_type']).size().unstack()

        # 添加参考信息
        weekend_avg_per_hour = df[df['is_weekend'] == 1].groupby('pickup_hour').size().mean()
        weekday_avg_per_hour = df[df['is_weekend'] == 0].groupby('pickup_hour').size().mean()

        # 设置图表属性
        ax2.set_title('分周末/工作日平均订单量', fontsize=12, fontweight='bold')
        ax2.set_xlabel('日期类型', fontsize=10)
        ax2.set_ylabel('订单数量', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        # 添加脚注说明
        fig.text(0.5, 0.02,
                 f'注：工作日平均每小时 {weekday_avg_per_hour:,.0f} 单，周末平均每小时 {weekend_avg_per_hour:,.0f} 单',
                 ha='center', fontsize=9, style='italic')

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9, bottom=0.1)

        # 创建目录
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        # 保存图表
        plt.savefig('outputs/travel_demand_patterns.png', dpi=300, bbox_inches='tight')

        # 显示图表
        plt.show()

        return fig

    def plot_region_heatmap(self):
        """绘制区域热度分析热力图"""

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        df = self.df

        # 统计 TOP 10 上车区域和下车区域
        top_pu = df['PULocationID'].value_counts().head(10)
        top_do = df['DOLocationID'].value_counts().head(10)

        print("\n" + "=" * 60)
        print("区域热度分析")
        print("=" * 60)
        print("\nTOP 10 上车区域:")
        for i, (loc_id, count) in enumerate(top_pu.items(), 1):
            print(f"  {i}. 区域 {loc_id}: {count:,} 单 ({count / len(df) * 100:.2f}%)")

        print("\nTOP 10 下车区域:")
        for i, (loc_id, count) in enumerate(top_do.items(), 1):
            print(f"  {i}. 区域 {loc_id}: {count:,} 单 ({count / len(df) * 100:.2f}%)")

        # 创建热力图矩阵
        top_pu_list = top_pu.index.tolist()
        top_do_list = top_do.index.tolist()

        heatmap_data = pd.DataFrame(0, index=top_pu_list, columns=top_do_list)

        for pu in top_pu_list:
            for do in top_do_list:
                count = len(df[(df['PULocationID'] == pu) & (df['DOLocationID'] == do)])
                heatmap_data.loc[pu, do] = count

        # 绘制图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('区域热度分析', fontsize=16, fontweight='bold')

        # 子图1：上下客区域热力图
        ax1 = axes[0]
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
                    ax=ax1, square=True, linewidths=0.5,
                    cbar_kws={'label': '订单数量'})
        ax1.set_title('TOP 10 上下客区域流量热力图', fontsize=12, fontweight='bold')
        ax1.set_xlabel('下车区域 ID', fontsize=10)
        ax1.set_ylabel('上车区域 ID', fontsize=10)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        # 子图2：高峰时段区域热度分析（17-19点）
        ax2 = axes[1]

        top5_pu = top_pu.head(5).index.tolist()
        zone_hour_matrix = pd.DataFrame(0, index=top5_pu, columns=range(24))

        for pu in top5_pu:
            for hour in range(24):
                count = len(df[(df['PULocationID'] == pu) & (df['pickup_hour'] == hour)])
                zone_hour_matrix.loc[pu, hour] = count

        sns.heatmap(zone_hour_matrix, annot=False, fmt='d', cmap='Blues',
                    ax=ax2, cbar_kws={'label': '订单数量'},
                    xticklabels=2, yticklabels=True)

        # 标记高峰时段 17-19点
        peak_hours = [17, 18]
        for peak_hour in peak_hours:
            ax2.axvline(x=peak_hour + 0.5, color='red', linestyle='-', linewidth=2, alpha=0.7)

        # 添加高峰时段标注
        ax2.text(17.5, len(top5_pu) - 0.5, '高峰时段\n17:00-19:00',
                 ha='center', va='top', fontsize=9, color='red', fontweight='bold')

        ax2.set_title('TOP 5 上车区域分时段热度', fontsize=12, fontweight='bold')
        ax2.set_xlabel('小时 (24小时制)', fontsize=10)
        ax2.set_ylabel('上车区域 ID', fontsize=10)
        ax2.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('outputs/region_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\n热力图已保存至: outputs/region_heatmap.png")
        plt.show()

        return fig

    def plot_top_regions_bar(self):
        """绘制 TOP 10 区域柱状图"""

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        df = self.df

        top_pu = df['PULocationID'].value_counts().head(10)
        top_do = df['DOLocationID'].value_counts().head(10)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('TOP 10 区域订单量分布', fontsize=14, fontweight='bold')

        # 上车区域
        ax1 = axes[0]
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, 10))
        bars1 = ax1.barh(range(10), top_pu.values, color=colors)
        ax1.set_yticks(range(10))
        ax1.set_yticklabels([f'区域 {i}' for i in top_pu.index])
        ax1.set_xlabel('订单数量', fontsize=10)
        ax1.set_title('TOP 10 上车区域', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars1, top_pu.values)):
            ax1.text(val, bar.get_y() + bar.get_height() / 2,
                     f'{val:,}', ha='left', va='center', fontsize=9)

        # 下车区域
        ax2 = axes[1]
        bars2 = ax2.barh(range(10), top_do.values, color=colors)
        ax2.set_yticks(range(10))
        ax2.set_yticklabels([f'区域 {i}' for i in top_do.index])
        ax2.set_xlabel('订单数量', fontsize=10)
        ax2.set_title('TOP 10 下车区域', fontsize=12, fontweight='bold')
        ax2.invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars2, top_do.values)):
            ax2.text(val, bar.get_y() + bar.get_height() / 2,
                     f'{val:,}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig('outputs/top_regions_bar.png', dpi=300, bbox_inches='tight')
        print(f"区域柱状图已保存至: outputs/top_regions_bar.png")
        plt.show()

        return fig

    def plot_peak_hour_regions(self):
        """绘制高峰时段区域分布图（17-19点）"""

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        df = self.df

        # 高峰时段 17-19点（17, 18两个整点）
        peak_hours = [17, 18]
        peak_df = df[df['pickup_hour'].isin(peak_hours)]
        peak_top_regions = peak_df['PULocationID'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.Reds(np.linspace(0.4, 0.9, 10))
        bars = ax.bar(range(10), peak_top_regions.values, color=colors, edgecolor='black')

        ax.set_xticks(range(10))
        ax.set_xticklabels([f'区域 {i}' for i in peak_top_regions.index], rotation=45, ha='right')
        ax.set_xlabel('区域 ID', fontsize=12)
        ax.set_ylabel('高峰时段订单数量', fontsize=12)
        ax.set_title(f'高峰时段 (17:00-19:00) TOP 10 上车区域',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, peak_top_regions.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(peak_top_regions.values) * 0.01,
                    f'{val:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('outputs/peak_hour_regions.png', dpi=300, bbox_inches='tight')
        print(f"高峰时段区域分布图已保存至: outputs/peak_hour_regions.png")
        plt.show()

        return fig

    def plot_all_charts(self):
        """绘制所有图表"""
        print("\n" + "=" * 60)
        print("开始绘制图表")
        print("=" * 60)
        self.plot_region_heatmap()
        self.plot_top_regions_bar()
        self.plot_peak_hour_regions()
        print("\n所有图表已保存至 outputs 目录")

    def plot_fare_factors(self):
        """绘制车费影响因素分析散点图"""

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        df = self.df

        # 采样数据（如果数据量太大，随机采样5万条以提高绘图速度）
        if len(df) > 50000:
            plot_df = df.sample(n=50000, random_state=42)
            print(f"数据量较大，随机采样 50,000 条进行绘图")
        else:
            plot_df = df.copy()

        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('车费影响因素分析', fontsize=16, fontweight='bold')

        # ========== 子图1：距离-车费散点图 ==========
        ax1 = axes[0]

        # 限制距离范围（0-50英里）使图表更清晰
        mask_distance = (plot_df['trip_distance'] > 0) & (plot_df['trip_distance'] <= 50)
        distance_data = plot_df[mask_distance]

        ax1.scatter(distance_data['trip_distance'], distance_data['fare_amount'],
                    alpha=0.3, s=10, c='steelblue', edgecolors='none')

        # 添加趋势线
        z = np.polyfit(distance_data['trip_distance'], distance_data['fare_amount'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 50, 100)
        ax1.plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'趋势线 (斜率: {z[0]:.2f})')

        ax1.set_xlabel('行程距离 (英里)', fontsize=12)
        ax1.set_ylabel('车费金额 (美元)', fontsize=12)
        ax1.set_title('距离 vs 车费', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # 添加相关系数
        corr_distance = distance_data['trip_distance'].corr(distance_data['fare_amount'])
        ax1.text(0.95, 0.05, f'相关系数: {corr_distance:.3f}',
                 transform=ax1.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # ========== 子图2：时段-车费散点图 ==========
        ax2 = axes[1]

        # 添加随机抖动避免重叠
        hour_jitter = plot_df['pickup_hour'] + np.random.normal(0, 0.15, len(plot_df))

        ax2.scatter(hour_jitter, plot_df['fare_amount'],
                    alpha=0.3, s=10, c='orange', edgecolors='none')

        # 计算每个小时的平均车费
        hourly_mean = plot_df.groupby('pickup_hour')['fare_amount'].mean()
        ax2.plot(hourly_mean.index, hourly_mean.values, 'r-', linewidth=2, marker='o', markersize=6, label='平均车费')

        # 标记高峰时段（17-19点）
        peak_hours = [17, 18]
        for peak_hour in peak_hours:
            ax2.axvline(x=peak_hour, color='red', linestyle=':', alpha=0.7, linewidth=1.5)

        ax2.set_xlabel('小时 (24小时制)', fontsize=12)
        ax2.set_ylabel('车费金额 (美元)', fontsize=12)
        ax2.set_title('时段 vs 车费', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')

        ax2.text(0.95, 0.95, '红色虚线: 高峰时段(17-19点)', transform=ax2.transAxes,
                 ha='right', va='top', fontsize=9, color='red')

        # ========== 子图3：乘客人数-车费散点图 ==========
        ax3 = axes[2]

        # 只显示1-6人
        mask_passenger = (plot_df['passenger_count'] >= 1) & (plot_df['passenger_count'] <= 6)
        passenger_data = plot_df[mask_passenger]

        # 添加随机抖动避免重叠
        passenger_jitter = passenger_data['passenger_count'] + np.random.normal(0, 0.08, len(passenger_data))

        ax3.scatter(passenger_jitter, passenger_data['fare_amount'],
                    alpha=0.3, s=10, c='green', edgecolors='none')

        # 添加箱线图
        passenger_data.boxplot(column='fare_amount', by='passenger_count', ax=ax3, grid=False, showfliers=False)

        ax3.set_xlabel('乘客人数', fontsize=12)
        ax3.set_ylabel('车费金额 (美元)', fontsize=12)
        ax3.set_title('乘客人数 vs 车费', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(1, 7))
        ax3.set_xticklabels([f'{i}人' for i in range(1, 7)])
        ax3.grid(True, alpha=0.3, linestyle='--')

        # 计算各乘客人数的平均车费
        passenger_mean = passenger_data.groupby('passenger_count')['fare_amount'].mean()
        for i in range(1, 7):
            if i in passenger_mean.index:
                ax3.scatter(i, passenger_mean[i], color='red', s=100, zorder=5, marker='D',
                            label='平均车费' if i == 1 else '')
        ax3.legend(loc='upper right')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # 保存图表
        plt.savefig('outputs/fare_factors_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n车费影响因素分析图已保存至: outputs/fare_factors_analysis.png")
        plt.show()

        return fig

    def plot_speed_analysis(self):
        """绘制平均速度因素分析图"""

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        df = self.df

        # 计算速度
        df_valid = df[df['trip_duration_hours'] > 0.01].copy()
        df_valid['avg_speed'] = df_valid['trip_distance'] / df_valid['trip_duration_hours']
        # 过滤异常速度（5-60英里/小时为合理范围）
        df_valid = df_valid[(df_valid['avg_speed'] >= 5) & (df_valid['avg_speed'] <= 60)]

        print("\n" + "=" * 60)
        print("平均速度因素分析")
        print("=" * 60)

        # ========== 1. 客流量TOP10区域 ==========
        top10_regions = df['PULocationID'].value_counts().head(10).index.tolist()
        print(f"\n客流量TOP10区域: {top10_regions}")

        # ========== 2. 创建热力图数据 ==========
        speed_matrix = pd.DataFrame(np.nan, index=top10_regions, columns=range(24))

        for region in top10_regions:
            for hour in range(24):
                region_data = df_valid[(df_valid['PULocationID'] == region) & (df_valid['pickup_hour'] == hour)]
                if len(region_data) > 0:
                    speed_matrix.loc[region, hour] = region_data['avg_speed'].median()

        # ========== 3. 创建分时段平均速度数据 ==========
        hourly_speed = df_valid.groupby('pickup_hour')['avg_speed'].median()

        # ========== 4. 绘制图表 ==========
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('平均速度因素分析', fontsize=16, fontweight='bold')

        # 子图1：TOP10区域分时段平均速度热力图
        ax1 = axes[0]

        # 绘制热力图
        sns.heatmap(speed_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r',
                    ax=ax1, square=False, linewidths=0.5,
                    cbar_kws={'label': '平均速度 (英里/小时)'},
                    vmin=10, vmax=30, center=20)

        ax1.set_title('TOP10 客流量区域分时段平均速度热力图', fontsize=12, fontweight='bold')
        ax1.set_xlabel('小时 (24小时制)', fontsize=10)
        ax1.set_ylabel('区域 ID', fontsize=10)

        # 修正：设置x轴刻度位置和标签
        ax1.set_xticks(range(0, 24, 2))  # 每2小时一个刻度
        ax1.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)], rotation=45)
        ax1.set_yticklabels([f'区域 {r}' for r in top10_regions], rotation=0)

        # 标记高峰时段（17-19点）
        for peak_hour in [17, 18]:
            ax1.axvline(x=peak_hour + 0.5, color='blue', linestyle='-', linewidth=2, alpha=0.5)
        ax1.text(18, len(top10_regions) - 0.5, '高峰时段\n17:00-19:00',
                 ha='center', va='top', fontsize=9, color='blue', fontweight='bold')

        # 添加颜色说明
        ax1.text(0.02, 0.02, '红色: 拥堵 (低速)\n绿色: 畅通 (高速)',
                 transform=ax1.transAxes, fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 子图2：不同时间段平均速度柱状图
        ax2 = axes[1]

        # 绘制柱状图
        hours = range(24)
        colors = []
        for h in hours:
            if 17 <= h <= 18:  # 晚高峰
                colors.append('red')
            elif 7 <= h <= 9:  # 早高峰
                colors.append('orange')
            else:
                colors.append('steelblue')

        hour_values = [hourly_speed.get(h, 0) for h in hours]
        bars = ax2.bar(hours, hour_values, color=colors, edgecolor='black', alpha=0.7)

        # 添加平均线
        avg_speed = hourly_speed.mean()
        ax2.axhline(y=avg_speed, color='green', linestyle='--', linewidth=2, label=f'全天平均: {avg_speed:.1f} mph')

        ax2.set_xlabel('小时 (24小时制)', fontsize=12)
        ax2.set_ylabel('平均速度 (英里/小时)', fontsize=12)
        ax2.set_title('不同时间段平均速度', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        ax2.set_xticklabels([f'{h}:00' for h in range(0, 24, 2)])
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')

        # 添加数值标签（只标记关键点）
        for hour in [7, 8, 9, 17, 18, 19]:
            if hour in hourly_speed.index and not pd.isna(hourly_speed[hour]):
                ax2.text(hour, hourly_speed[hour] + 0.5, f'{hourly_speed[hour]:.1f}',
                         ha='center', fontsize=8)

        # 添加拥堵时段标注
        if not hourly_speed.empty:
            slowest_hour = hourly_speed.idxmin()
            slowest_speed = hourly_speed.min()
            ax2.annotate(f'最拥堵时段\n{slowest_hour}:00\n{slowest_speed:.1f} mph',
                         xy=(slowest_hour, slowest_speed),
                         xytext=(slowest_hour + 2, slowest_speed + 3),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=9, color='red', ha='center')

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # 保存图表
        plt.savefig('outputs/speed_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n平均速度因素分析图已保存至: outputs/speed_analysis.png")
        plt.show()

        return fig

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

    processor.plot_travel_demand_patterns()

    processor.plot_all_charts()

    processor.plot_fare_factors()

    # 绘制平均速度因素分析图
    processor.plot_speed_analysis()

    return processor


if __name__ == "__main__":
    main()