# 夏元益-25361038-期末大作业
## 1. AI交互日志：

M1
模块1：质量报告
我的prompt：我有一份parquet的数据，我已经用read_parquet完成了读入，接下来请你帮我生成一份代码，用于检测数据的缺失率和异常值，每个字段的正常取值在附件中（附上字典PDF文件）。只需要生成核心代码和思路。
AI输出摘要：AI 给出了 check_data() 方法。
核心思路：
1. 遍历核心字段计算缺失数量及缺失率
2. 计算行程时长，检查行程时长、距离、乘客人数、车费、支付方式的异常值
3. 输出统计摘要（均值、中位数）和支付方式分布
决策：完全采用，符合预期

模块2：数据清洗
我的prompt：帮我对异常数据进行处理
AI输出摘要：直接把所有异常的、缺失的都删除
决策：部分采纳，异常数据直接删除，缺失数据就用正常数据的平均值替代。

模块3：时间特征提取
我的prompt：帮我从行程时间中提取小时、星期、是否高峰等特征，还想要平均速度和收入时间密度，要求呈现核心思路
AI输出摘要：AI 给出 extract_time_features() 和 detect_peak_hours() 方法，核心思路：
1. 从 datetime 提取 hour、weekday、is_weekend
2. 计算行程时长（小时和分钟）
3. 速度 = 距离 / 时间，收入密度 = 总金额 / 时间
4. 统计各小时订单量，取最高的两个小时作为高峰时段
决策：采纳

M2
模块1：出行需求时间规律图
我的 Prompt：绘制图表，包含两个子表：1.分小时平均订单量折线图；2.分周末/工作日平均订单量柱状图。总表名称为出行需求时间规律
AI 输出摘要：输出两个子图代码，子图1：统计各小时订单量，用折线图展示，标记高峰时段和平均值线，子图2：按周末/工作日分组统计总订单量，用柱状图展示并添加百分比标签
决策：修改，由于数据字段都是英文，所以AI给出代码生成的图片是英文，不利于我们直观看，所以要求它增加中文字体并且把名称改成中文，核心字段也改成中文

模块2：区域热度分析图
我的 Prompt：绘制热力图，分析上下客量最高的 TOP 10 区域及高峰时段分布
AI 输出摘要：AI 生成代码：1. 统计 TOP10 上车区域和 TOP10 下车区域
2. 构建上车区域×下车区域的热力图矩阵，用 seaborn.heatmap 绘制
3. 子图2：统计 TOP5 上车区域的分时段订单量，用热力图展示，标记高峰时段
决策：采纳

模块3：车费影响因素分析图
我的 Prompt：绘制行程不同因素-车费散点图：距离-车费、时段-车费、乘客人数-车费
AI 输出摘要：代码：距离-车费：散点图 + 趋势线 + 相关系数；时段-车费：散点图 + 各小时平均车费折线 + 标记高峰时段；乘客人数-车费：散点图 + 箱线图 + 各人数平均车费标记

模块4：平均速度因素分析图
我的 Prompt：绘制平均速度因素分析图：TOP10区域分时段平均速度热力图 + 不同时间段平均速度柱状图
AI 输出摘要：提供了平均速度热力图生成代码，但运行的时候出现错误：TypeError: Invalid value '33.37' for dtype 'int64'
决策：将报错信息重新给AI让它来分析原因并且修改
AI输出摘要：
错误原因：AI 用 pd.DataFrame(0, ...) 创建了 int64 类型的矩阵，但速度值是浮点数
如何发现：运行代码时抛出异常，定位到 speed_matrix.loc[region, hour] = ...
如何修正：改用 pd.DataFrame(np.nan, ...) 创建 float 类型矩阵
修正代码：speed_matrix = pd.DataFrame(np.nan, index=top10_regions, columns=range(24))
决策：由于运行后依旧报错：ValueError: The number of FixedLocator locations (24) does not match the number of labels (12)，因此让AI重新修改
AI输出摘要：
错误原因 ： 热力图有24个刻度（0-23点），但只传了12个标签 
如何发现 ： 运行报错处有提示
如何修正 ：先设置刻度位置，再设置标签：ax1.set_xticks(range(0, 24, 2)) 
修正代码 ：在 set_xticklabels 之前添加 ax1.set_xticks(range(0, 24, 2)) 
决策：采纳。修正后能正常运行并且达到预期输出效果。

M3预测
我的 Prompt：用 TensorFlow 构建神经网络预测出行需求量，并且用随机森林也来预测，把两者对比数据呈现给我
AI 输出摘要：
1. 特征工程：时间特征（hour_sin/cos、weekday_sin/cos）、滞后特征（lag_1/2/24）、滚动平均特征
2. 神经网络：128→64→32→16→1 层，Dropout 防过拟合，早停回调
3. 随机森林：100棵树，输出特征重要性
4. 评估：MAE、RMSE，绘制 loss 曲线和残差对比图
决策：采纳

M4问答系统
我的prompt：设计一个问答系统，可以让用户输入自然语言问题，系统对自然语言分析匹配关键词来进行已有功能的问答，提取关键词作为参数
AI输出摘要：1.设计 System Prompt 描述支持的查询类型和输出格式
2. 提供 fallback 规则匹配
3. 意图识别后调用对应的数据处理函数
决策：拒绝采纳，因为没有调用API
如何发现：当我问：我明天9点出行，可能要交的费用是多少，系统将意图识别为 hour_demand (置信度: 0.5)由此发现只用了规则匹配
如何修正：去deepseek官网充值并且创建API，把key写入代码
修正代码：重新生成用API调用的问答系统
最后决策：采纳，能正常使用。

## 2. 三阶段对比
典型功能：提取高峰时段
自己写：
    hours = []#存储行程的小时
    for index, row in df.iterrows():#遍历文件
        dt = row['tpep_pickup_datetime']#找到上车时间对应的字段
        hour = dt.hour #取整数小时
        hours.append(hour)#存储到hours里面
    hour_counter = {}#存储每个小时出现次数，小时-次数
    for h in hours:#统计每个小时订单数，相当于合并了重复的部分
        if h not in hour_counter:#初始化
            hour_counter[h] = 0
        hour_counter[h] = 1+hour_counter[h]

    sorted_hours = []#存储行程数和小时
    for hour in range(24):#统计订单数
        count = hour_counter.get(hour, 0)
        sorted_hours.append((count, hour))
    
    for i in range(len(sorted_hours)):#冒泡排序找到最大的俩小时
        for j in range(i+1, len(sorted_hours)):
            if sorted_hours[i][0] < sorted_hours[j][0]:
                sorted_hours[i], sorted_hours[j] = sorted_hours[j], sorted_hours[i]
    
    peak_hours = [sorted_hours[0][1], sorted_hours[1][1]]

AI辅
prompt：帮我在 TaxiDataProcessor 类中添加一个方法
功能：统计每个小时的订单量，找出最高的两个小时作为高峰时段。
代码：
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

    # 打印各小时行程数分布（带条形图）
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

    return peak_hours

vibe coding：帮我识别高峰时段
def detect_peak_hours(self):
    """基于已有数据进行高峰时段的提取，取行程量最高的两个小时作为高峰时段"""
    
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
    
    # 存储结果
    self.peak_hours = peak_hours
    
    return peak_hours

反思分析：
1.自己写代码的话，有些关键函数运用可能会忘记，需要找资料回顾，同时手打代码犯错风险高，可能出现很多小的语法错误，报错很多次，并且手打速度慢效率低。优势就是自己写的代码自己能够完全理解。
2.AI辅助写代码能够更快的提升效率，在自己知道应该如何去执行（比如我先设定以整数小时为固定窗口，让它统计并求最大值）的情况下可以大大减少简单代码的手打时间，像排序求最大值这种就不需要再亲历亲为一遍。ai可能会用一些我不熟悉的函数或者方法，这样会让我有时候没有那么深刻理解
3.vibe coding就不需要去深究他到底内部逻辑和结构是怎样的，只要看输出效果就好了，要是不符合预期就再让他改，只有实用性，对理解性很低贡献。理解深度很低。
## 3.反思
AI确实在大工程项目中起着不可或缺的作用，这次作业让我对AI有了新的认识。vibe coding的模式能够让我们从自然语言直接获得想要的结果，而中间的实现过程对我们来说是未知的，也不必要知道。功利一点讲，这确实非常有用，但是他在帮我我们提升理解上可以说是几乎没有作用。
并且不知道其中的操作过程，有些时候可能他会选取错误的方法进行代码编写，导致结果效率很低，这种时候需要我们对内部的执行程序有一定理解，才能更好的指出错误，不然在ai眼里他其实并不算bug。比如损失函数的选择。
项目开发避免不了创新，只靠着vibe coding只能在现有的基础上解决一些已经解决过的问题，但真要开发新项目，我认为人为的比例需要大幅提高。因此，我觉得AI辅助是比较好的方式，让我们避免一些重复性的工作，但是真正的框架，决策，等还是需要我们来做。

