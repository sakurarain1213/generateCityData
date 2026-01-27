# -*- coding: utf-8 -*-
"""
人口迁徙数据生成器（强约束修正版）
修复重点：
1. 解决生成量为0的问题（Month=12）
2. 解决头部城市人口爆炸问题（引入全局归一化概率向量）
3. 解决长尾城市无迁入问题（基于配额的概率分配）


这个算法是在用**“已知的结果（每年总人口）”去强行解释“过程（怎么流动的）”**。
只要CSV总人口数据是准的，模拟出的迁出总量在大方向上就不会错得离谱，
但细节（比如到底是哪些人跑了）完全依赖于设定的 AGE_MIGRATION_BASE 等参数是否科学。
换言之：无论上一年的模拟结果是导致这个城市人口“爆炸”还是“归零”，只要到了新的一年，
calibrate_city_state 就会无视之前的逻辑结果，直接按比例 强行让该城市的人口等于 CSV 中的 total_pop。

以上是总量约束

【由于CSV中的迁入数据是存量的！ 不是每年的迁入流量。 因此无法直接把迁入作为约束指导每年的迁出。推断和回归预测使用】
【只能取2000 或2010的当年迁入值 归一化作为吸引力权重 去指导一共20年的每年的流量分配  让每年结果人数等于实际总人口约束即可 】

以下是迁入约束如何推算迁出率：
核心逻辑是："存量定方向，费率定总量"
# 第1051-1053行：设定【年度】【每城】【迁移率】（4%-6%）
# 第1058-1062行：计算全局缩放因子
# 第1045-1048行：【CSV中的迁入存量 比如2000年各个城市的迁入数量】作为吸引力权重 指导每年生成的流量的权重
 # 存量越高，吸引力越大



最终一句话：逐年迁徙值不定！需要按抽样挖掘出逐年迁徙人数 才能知道每年离家人数。
目前只能赶走同比例的一批人，再按城市吸引这批人，确保每城市年末人口等于约束

目前的问题是
总流入人口不满足约束（只满足相对比例，不满足绝对数量） 因为缺失【每年的总迁入数据】
迁入/迁出不平衡（没有全局质量守恒检查） 因为也没考虑出生率

"""

import os
import sys
import time
import random
import warnings
import duckdb
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局配置 (保持不变)
# ==============================================================================

OUTPUT_DIR = 'output'
DB_FILENAME = 'local_migration_data.db'
CSV_SAMPLE_FILENAME = 'migration_data_sample_100.csv'
CONSTRAINT_CSV_PATH = r'C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN\2.csv'

OUTPUT_YEARS = list(range(2000, 2021))
OUTPUT_MONTHS = [12] # 修改：只生成12月的数据作为年度快照
TOP_N_CITIES = 20
MIN_TYPE_COUNT = 500
POWER_LAW_ALPHA = 2.3
TOTAL_POPULATION_BASE = 1000_0000

# 维度定义
DIMENSIONS = {
    'D1': {'name': '性别', 'values': ['M', 'F']},
    'D2': {'name': '生命周期', 'values': ['16-24', '25-34', '35-49', '50-60', '60+']},
    'D3': {'name': '学历', 'values': ['EduLo', 'EduMid', 'EduHi']},
    'D4': {'name': '行业赛道', 'values': ['Agri', 'Mfg', 'Service', 'Wht']},
    'D5': {'name': '相对收入', 'values': ['IncL', 'IncML', 'IncM', 'IncMH', 'IncH']},
    'D6': {'name': '家庭状态', 'values': ['Split', 'Unit']},
}

# 城市列表（保持不变）
CITIES = [
    ('1301', '石家庄'), ('1302', '唐山'), ('1303', '秦皇岛'), ('1304', '邯郸'),
    ('1305', '邢台'), ('1306', '保定'), ('1307', '张家口'), ('1308', '承德'),
    ('1309', '沧州'), ('1310', '廊坊'), ('1311', '衡水'), ('1401', '太原'),
    ('1402', '大同'), ('1403', '阳泉'), ('1404', '长治'), ('1405', '晋城'),
    ('1406', '朔州'), ('1407', '晋中'), ('1408', '运城'), ('1409', '忻州'),
    ('1410', '临汾'), ('1411', '吕梁'), ('1501', '呼和浩特'), ('1502', '包头'),
    ('1503', '乌海'), ('1504', '赤峰'), ('1505', '通辽'), ('1506', '鄂尔多斯'),
    ('1507', '呼伦贝尔'), ('1508', '巴彦淖尔'), ('1509', '乌兰察布'), ('1522', '兴安盟'),
    ('1525', '锡林郭勒盟'), ('1529', '阿拉善盟'), ('2101', '沈阳'), ('2102', '大连'),
    ('2103', '鞍山'), ('2104', '抚顺'), ('2105', '本溪'), ('2106', '丹东'),
    ('2107', '锦州'), ('2108', '营口'), ('2109', '阜新'), ('2110', '辽阳'),
    ('2111', '盘锦'), ('2112', '铁岭'), ('2113', '朝阳'), ('2114', '葫芦岛'),
    ('2201', '长春'), ('2202', '吉林'), ('2203', '四平'), ('2204', '辽源'),
    ('2205', '通化'), ('2206', '白山'), ('2207', '松原'), ('2208', '白城'),
    ('2224', '延边'), ('2301', '哈尔滨'), ('2302', '齐齐哈尔'), ('2303', '鸡西'),
    ('2304', '鹤岗'), ('2305', '双鸭山'), ('2306', '大庆'), ('2307', '伊春'),
    ('2308', '佳木斯'), ('2309', '七台河'), ('2310', '牡丹江'), ('2311', '黑河'),
    ('2312', '绥化'), ('2327', '大兴安岭'), ('3201', '南京'), ('3202', '无锡'),
    ('3203', '徐州'), ('3204', '常州'), ('3205', '苏州'), ('3206', '南通'),
    ('3207', '连云港'), ('3208', '淮安'), ('3209', '盐城'), ('3210', '扬州'),
    ('3211', '镇江'), ('3212', '泰州'), ('3213', '宿迁'), ('3301', '杭州'),
    ('3302', '宁波'), ('3303', '温州'), ('3304', '嘉兴'), ('3305', '湖州'),
    ('3306', '绍兴'), ('3307', '金华'), ('3308', '衢州'), ('3309', '舟山'),
    ('3310', '台州'), ('3311', '丽水'), ('3401', '合肥'), ('3402', '芜湖'),
    ('3403', '蚌埠'), ('3404', '淮南'), ('3405', '马鞍山'), ('3406', '淮北'),
    ('3407', '铜陵'), ('3408', '安庆'), ('3410', '黄山'), ('3411', '滁州'),
    ('3412', '阜阳'), ('3413', '宿州'), ('3415', '六安'), ('3416', '亳州'),
    ('3417', '池州'), ('3418', '宣城'), ('3501', '福州'), ('3502', '厦门'),
    ('3503', '莆田'), ('3504', '三明'), ('3505', '泉州'), ('3506', '漳州'),
    ('3507', '南平'), ('3508', '龙岩'), ('3509', '宁德'), ('3601', '南昌'),
    ('3602', '景德镇'), ('3603', '萍乡'), ('3604', '九江'), ('3605', '新余'),
    ('3606', '鹰潭'), ('3607', '赣州'), ('3608', '吉安'), ('3609', '宜春'),
    ('3610', '抚州'), ('3611', '上饶'), ('3701', '济南'), ('3702', '青岛'),
    ('3703', '淄博'), ('3704', '枣庄'), ('3705', '东营'), ('3706', '烟台'),
    ('3707', '潍坊'), ('3708', '济宁'), ('3709', '泰安'), ('3710', '威海'),
    ('3711', '日照'), ('3713', '临沂'), ('3714', '德州'), ('3715', '聊城'),
    ('3716', '滨州'), ('3717', '菏泽'), ('4101', '郑州'), ('4102', '开封'),
    ('4103', '洛阳'), ('4104', '平顶山'), ('4105', '安阳'), ('4106', '鹤壁'),
    ('4107', '新乡'), ('4108', '焦作'), ('4109', '濮阳'), ('4110', '许昌'),
    ('4111', '漯河'), ('4112', '三门峡'), ('4113', '南阳'), ('4114', '商丘'),
    ('4115', '信阳'), ('4116', '周口'), ('4117', '驻马店'), ('4201', '武汉'),
    ('4202', '黄石'), ('4203', '十堰'), ('4205', '宜昌'), ('4206', '襄阳'),
    ('4207', '鄂州'), ('4208', '荆门'), ('4209', '孝感'), ('4210', '荆州'),
    ('4211', '黄冈'), ('4212', '咸宁'), ('4213', '随州'), ('4228', '恩施'),
    ('4301', '长沙'), ('4302', '株洲'), ('4303', '湘潭'), ('4304', '衡阳'),
    ('4305', '邵阳'), ('4306', '岳阳'), ('4307', '常德'), ('4308', '张家界'),
    ('4309', '益阳'), ('4310', '郴州'), ('4311', '永州'), ('4312', '怀化'),
    ('4313', '娄底'), ('4331', '湘西'), ('4401', '广州'), ('4402', '韶关'),
    ('4403', '深圳'), ('4404', '珠海'), ('4405', '汕头'), ('4406', '佛山'),
    ('4407', '江门'), ('4408', '湛江'), ('4409', '茂名'), ('4412', '肇庆'),
    ('4413', '惠州'), ('4414', '梅州'), ('4415', '汕尾'), ('4416', '河源'),
    ('4417', '阳江'), ('4418', '清远'), ('4419', '东莞'), ('4420', '中山'),
    ('4451', '潮州'), ('4452', '揭阳'), ('4453', '云浮'), ('4501', '南宁'),
    ('4502', '柳州'), ('4503', '桂林'), ('4504', '梧州'), ('4505', '北海'),
    ('4506', '防城港'), ('4507', '钦州'), ('4508', '贵港'), ('4509', '玉林'),
    ('4510', '百色'), ('4511', '贺州'), ('4512', '河池'), ('4513', '来宾'),
    ('4514', '崇左'), ('4601', '海口'), ('4602', '三亚'), ('4603', '三沙'),
    ('4604', '儋州'), ('5101', '成都'), ('5103', '自贡'), ('5104', '攀枝花'),
    ('5105', '泸州'), ('5106', '德阳'), ('5107', '绵阳'), ('5108', '广元'),
    ('5109', '遂宁'), ('5110', '内江'), ('5111', '乐山'), ('5113', '南充'),
    ('5114', '眉山'), ('5115', '宜宾'), ('5116', '广安'), ('5117', '达州'),
    ('5118', '雅安'), ('5119', '巴中'), ('5120', '资阳'), ('5132', '阿坝'),
    ('5133', '甘孜'), ('5134', '凉山'), ('5201', '贵阳'), ('5202', '六盘水'),
    ('5203', '遵义'), ('5204', '安顺'), ('5205', '毕节'), ('5206', '铜仁'),
    ('5223', '黔西南'), ('5226', '黔东南'), ('5227', '黔南'), ('5301', '昆明'),
    ('5303', '曲靖'), ('5304', '玉溪'), ('5305', '保山'), ('5306', '昭通'),
    ('5307', '丽江'), ('5308', '普洱'), ('5309', '临沧'), ('5323', '楚雄'),
    ('5325', '红河'), ('5326', '文山'), ('5328', '西双版纳'), ('5329', '大理'),
    ('5331', '德宏'), ('5333', '怒江'), ('5334', '迪庆'), ('5401', '拉萨'),
    ('5402', '日喀则'), ('5403', '昌都'), ('5404', '林芝'), ('5405', '山南'),
    ('5406', '那曲'), ('5425', '阿里'), ('6101', '西安'), ('6102', '铜川'),
    ('6103', '宝鸡'), ('6104', '咸阳'), ('6105', '渭南'), ('6106', '延安'),
    ('6107', '汉中'), ('6108', '榆林'), ('6109', '安康'), ('6110', '商洛'),
    ('6201', '兰州'), ('6202', '嘉峪关'), ('6203', '金昌'), ('6204', '白银'),
    ('6205', '天水'), ('6206', '武威'), ('6207', '张掖'), ('6208', '平凉'),
    ('6209', '酒泉'), ('6210', '庆阳'), ('6211', '定西'), ('6212', '陇南'),
    ('6229', '临夏'), ('6230', '甘南'), ('6301', '西宁'), ('6302', '海东'),
    ('6322', '海北'), ('6323', '黄南'), ('6325', '海南'), ('6326', '果洛'),
    ('6327', '玉树'), ('6328', '海西'), ('6401', '银川'), ('6402', '石嘴山'),
    ('6403', '吴忠'), ('6404', '固原'), ('6405', '中卫'), ('6501', '乌鲁木齐'),
    ('6502', '克拉玛依'), ('6504', '吐鲁番'), ('6505', '哈密'), ('6523', '昌吉'),
    ('6527', '博尔塔拉'), ('6528', '巴音郭楞'), ('6529', '阿克苏'), ('6530', '克孜勒苏'),
    ('6531', '喀什'), ('6532', '和田'), ('6540', '伊犁'), ('6542', '塔城'),
    ('6543', '阿勒泰'), ('1100', '北京'), ('1200', '天津'), ('3100', '上海'),
    ('5000', '重庆')
]

REGION_CODES = [city[0] for city in CITIES]
TIER1_CITIES = ['1100', '3100', '4401', '4403']
TIER2_CITIES = ['1200', '3201', '3205', '3301', '3302', '3702', '4201', '4301', '5101', '5000']

# --- 核心参数调整：为了让分布更集中，减少长尾 ---

# 1. 提高截断阈值：更早地切断极小的人群 Type，减少碎片化
MIN_TYPE_COUNT = 500  # 原值 200

# 2. 提高幂律分布 Alpha 值：让头部 Type 占据更大比例
POWER_LAW_ALPHA = 2.4  # 原值 1.8 (值越大，头部越集中)

# 3. 降低温度系数：让高吸引力的城市/Type获得绝对优势，大幅削弱长尾城市的概率
TEMPERATURE_BASE = 0.15  # 原值 0.3 (越低越趋向于"赢家通吃")

# 4. 提高 Top N 占比：强制大部分人口流向前 20 个城市
TOP_N_RATIO_BASE = 0.95  # 原值 0.8

TOTAL_POPULATION_BASE = 1000_0000

# 迁移模型参数
# 【修改】大幅降低基础迁移概率，符合中国国情 (年均流动人口占比约2%-5%)
AGE_MIGRATION_BASE = {'16-24': 0.08, '25-34': 0.06, '35-49': 0.03, '50-60': 0.01, '60+': 0.005}
EDU_MIGRATION_MULTIPLIER = {'EduLo': 0.8, 'EduMid': 1.0, 'EduHi': 1.3}
INDUSTRY_MIGRATION_MULTIPLIER = {'Agri': 0.5, 'Mfg': 1.4, 'Service': 1.2, 'Wht': 0.9}
INCOME_MIGRATION_MULTIPLIER = {'IncL': 1.3, 'IncML': 1.1, 'IncM': 1.0, 'IncMH': 0.9, 'IncH': 0.7}
FAMILY_MIGRATION_MULTIPLIER = {'Split': 1.2, 'Unit': 0.8}
GENDER_MIGRATION_MULTIPLIER = {'M': 1.05, 'F': 0.95}
SEASONAL_FACTORS = {1: 0.7, 2: 0.6, 3: 1.2, 4: 1.15, 5: 1.1, 6: 1.0, 7: 0.95, 8: 0.9, 9: 1.1, 10: 1.05, 11: 1.0, 12: 0.8}
MIGRATION_PROB_MIN = 0.05
MIGRATION_PROB_MAX = 0.50

# 城市地理经济数据（保持不变）
CITY_COORDINATES = {
    '1100': (116.4074, 39.9042), '3100': (121.4737, 31.2304), '4401': (113.2644, 23.1291), '4403': (114.0579, 22.5431),
    '1200': (117.2008, 39.0842), '3201': (118.7969, 32.0603), '3205': (120.5853, 31.2989), '3301': (120.1551, 30.2741),
    '3302': (121.5440, 29.8683), '3702': (120.3826, 36.0671), '4201': (114.3054, 30.5928), '4301': (112.9388, 28.2282),
    '5101': (104.0668, 30.5728), '5000': (106.5516, 29.5630),
}
CITY_GDP = {'1100': 4.4, '3100': 4.7, '4401': 3.0, '4403': 3.5, '1200': 1.6, '3201': 1.7, '3205': 2.4, '3301': 1.9, '3302': 1.6, '3702': 1.5, '4201': 1.9, '4301': 1.4, '5101': 2.1, '5000': 3.0}
CITY_INDUSTRY_TYPE = {
    '1100': 3, '3100': 3, '4401': 3, '4403': 3,
    '3205': 1, '4406': 1, '3302': 1,
    '3301': 2, '5101': 2, '4201': 2, '1200': 2,
}

INDUSTRY_CITY_MATCH = {
    (1, 'Agri'): 0.7, (1, 'Mfg'): 1.4, (1, 'Service'): 1.1, (1, 'Wht'): 0.9,
    (2, 'Agri'): 0.9, (2, 'Mfg'): 1.2, (2, 'Service'): 1.3, (2, 'Wht'): 1.1,
    (3, 'Agri'): 0.5, (3, 'Mfg'): 0.9, (3, 'Service'): 1.4, (3, 'Wht'): 1.5,
}

EDU_INDUSTRY_MATCH = {
    ('EduLo', 'Agri'): 1.5, ('EduLo', 'Mfg'): 1.3, ('EduLo', 'Service'): 0.9, ('EduLo', 'Wht'): 0.5,
    ('EduMid', 'Agri'): 0.9, ('EduMid', 'Mfg'): 1.2, ('EduMid', 'Service'): 1.3, ('EduMid', 'Wht'): 1.0,
    ('EduHi', 'Agri'): 0.3, ('EduHi', 'Mfg'): 0.8, ('EduHi', 'Service'): 1.1, ('EduHi', 'Wht'): 1.6,
}

AGE_INDUSTRY_MATCH = {
    '16-24': {'Agri': 0.4, 'Mfg': 1.3, 'Service': 1.4, 'Wht': 0.8},
    '25-34': {'Agri': 0.3, 'Mfg': 1.2, 'Service': 1.3, 'Wht': 1.3},
    '35-49': {'Agri': 0.6, 'Mfg': 1.3, 'Service': 1.1, 'Wht': 1.2},
    '50-60': {'Agri': 1.2, 'Mfg': 1.0, 'Service': 0.9, 'Wht': 0.7},
    '60+': {'Agri': 1.8, 'Mfg': 0.5, 'Service': 0.6, 'Wht': 0.4},
}

# 全局变量：约束数据
CITY_CONSTRAINTS = {}

# ==============================================================================
# 2. 读取并插值约束数据
# ==============================================================================

def load_and_interpolate_constraints(csv_path):
    """
    读取新版CSV并插值生成2000-2020每年的约束

    关键理解:
    - CSV中的'人口普查跨市【迁入】总人口'字段表示该城市接收的迁入量
    - 例如:北京800万 → 意味着有800万人从全国各地迁入北京
    - 我们的验证逻辑: SUM(其他城市→北京的迁出) ≈ 800万
    """
    print(f"读取约束CSV: {csv_path}")

    # 1. 读取 CSV
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"  成功使用编码: {enc}")
            break
        except Exception:
            continue

    if df is None:
        raise ValueError("无法读取CSV文件，请检查路径和编码")

    # 2. 列名清洗与映射
    df.columns = df.columns.str.strip()

    # 【核心修改】明确映射关系:字段名是"迁入"，语义也是"迁入目标"
    col_mapping = {
        '年份': 'year',
        '城市代码': 'city_code',
        '常住人口数(人)': 'total_pop',
        '人口普查跨市【迁入】总人口': 'target_inmigration_raw'  # 迁入总人口
    }

    # 模糊匹配列名（同时支持"迁入"和"迁出"两种可能的字段名）
    for col in df.columns:
        if ('迁入' in col or '迁出' in col) and '总人口' in col:
            col_mapping[col] = 'target_inmigration_raw'
            print(f"  检测到迁移字段: {col} -> 映射为 target_inmigration_raw")

    df = df.rename(columns=col_mapping)

    # 确保关键列存在
    if 'target_inmigration_raw' not in df.columns:
        df['target_inmigration_raw'] = np.nan

    required = ['year', 'city_code', 'total_pop']
    df = df.dropna(subset=required)

    # 格式清洗
    df['city_code'] = df['city_code'].astype(str).str.strip()
    df = df[df['city_code'].str.match(r'^[1-6]\d{3}$')].copy()
    for col in ['year', 'total_pop', 'target_inmigration_raw']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['year'] = df['year'].astype(int)

    # 3. 插值处理
    constraints = {}
    valid_years = range(2000, 2021)
    unique_cities = df['city_code'].unique()

    print("  开始处理分城市插值 (In-Migration Target)...")

    for city_code in tqdm(unique_cities, desc="Processing Constraints"):
        city_df = df[df['city_code'] == city_code].sort_values('year')
        city_df = city_df.set_index('year').reindex(valid_years)

        # 1. 常住人口插值
        if city_df['total_pop'].notna().sum() >= 2:
            city_df['total_pop'] = city_df['total_pop'].interpolate(method='linear')
        else:
            city_df['total_pop'] = city_df['total_pop'].fillna(method='ffill').fillna(method='bfill')

        # 2. 迁入目标插值 (Target In-Migration)
        # 【关键修改】要求至少2年数据才能做插值，否则无约束
        valid_mig = city_df['target_inmigration_raw'].notna().sum()
        if valid_mig >= 2:
            # 有至少2年数据,可以进行线性插值
            city_df['target_inmigration'] = city_df['target_inmigration_raw'].interpolate(method='linear')
        else:
            # 数据不足2年,标记为无约束 (使用None而不是0,便于后续区分)
            city_df['target_inmigration'] = None  # 无约束

        constraints[city_code] = {}
        for year in valid_years:
            p = int(city_df.loc[year, 'total_pop']) if pd.notna(city_df.loc[year, 'total_pop']) else 0
            # 这里记录的是该城市当年希望接收多少外来人口
            # 【修改】只有当迁移数据有效时才记录,否则为None
            m_in = None
            if pd.notna(city_df.loc[year, 'target_inmigration']):
                m_in = int(city_df.loc[year, 'target_inmigration'])

            constraints[city_code][year] = {
                'total_pop': p,
                'target_inmigration': m_in  # None表示无约束,整数表示有约束
            }

    total_pop_2000_check = sum(c[2000]['total_pop'] for c in constraints.values() if 2000 in c)

    # 【新增】统计有/无迁移约束的城市
    cities_with_migration_constraint = []
    cities_without_migration_constraint = []

    for city_code in unique_cities:
        has_constraint = False
        for year in valid_years:
            if city_code in constraints and year in constraints[city_code]:
                if constraints[city_code][year]['target_inmigration'] is not None:
                    has_constraint = True
                    break

        if has_constraint:
            cities_with_migration_constraint.append(city_code)
        else:
            cities_without_migration_constraint.append(city_code)

    print(f"  约束城市数: {len(constraints)}")
    print(f"  有迁移约束的城市数: {len(cities_with_migration_constraint)}")
    print(f"  无迁移约束的城市数: {len(cities_without_migration_constraint)}")
    if cities_without_migration_constraint:
        print(f"  无约束城市示例(前10个): {cities_without_migration_constraint[:10]}")
    print(f"  约束数据 2000年总人口汇总(Check): {total_pop_2000_check:,}")

    return constraints

# ==============================================================================
# 3. Type 生成
# ==============================================================================

def generate_all_types():
    import itertools
    keys = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    vals = [DIMENSIONS[k]['values'] for k in keys]
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]

def type_to_id(type_dict):
    parts = [type_dict['D1']]
    age_map = {'16-24': '20', '25-34': '30', '35-49': '40', '50-60': '55', '60+': '65'}
    parts.append(age_map.get(type_dict['D2'], type_dict['D2']))
    parts.extend([type_dict['D3'], type_dict['D4'], type_dict['D5'], type_dict['D6'], type_dict['D7']])
    return '_'.join(parts)

def type_id_to_city_code(type_id):
    """从type_id提取城市代码"""
    return type_id.split('_')[-1]

# ==============================================================================
# 4. 人口分布计算
# ==============================================================================

def _calculate_type_multiplier(type_dict):
    """计算Type的人口分布乘数"""
    m = 1.0
    m *= {'16-24': 0.8, '25-34': 1.5, '35-49': 1.3, '50-60': 0.6, '60+': 0.3}.get(type_dict['D2'], 1.0)
    m *= {'EduLo': 1.2, 'EduMid': 1.5, 'EduHi': 0.8}.get(type_dict['D3'], 1.0)
    m *= {'IncL': 1.1, 'IncML': 1.3, 'IncM': 1.5, 'IncMH': 1.0, 'IncH': 0.5}.get(type_dict['D5'], 1.0)
    m *= {'Agri': 0.6, 'Mfg': 1.2, 'Service': 1.4, 'Wht': 0.8}.get(type_dict['D4'], 1.0)
    m *= {'Split': 0.9, 'Unit': 1.1}.get(type_dict['D6'], 1.0)
    m *= EDU_INDUSTRY_MATCH.get((type_dict['D3'], type_dict['D4']), 1.0)
    m *= AGE_INDUSTRY_MATCH.get(type_dict['D2'], {}).get(type_dict['D4'], 1.0)

    city_code = type_dict['D7']
    city_industry_type = CITY_INDUSTRY_TYPE.get(city_code, 2)
    m *= INDUSTRY_CITY_MATCH.get((city_industry_type, type_dict['D4']), 1.0)

    return m

def initialize_population_2000(city_types, constraints):
    """
    2000年初始化：生成各城市各Type的人口分布

    [关键修复]：通过归一化确保每个城市的总人口严格等于CSV约束值

    返回:
        city_state: {city_code: {type_id: count}}
    """
    city_state = {}
    city_to_types = {}

    for td in city_types:
        cc = td['D7']
        if cc not in city_to_types:
            city_to_types[cc] = []
        city_to_types[cc].append(td)

    for cc, types_in_city in city_to_types.items():
        # 获取 CSV 中的 2000 年约束人口
        if cc in constraints and 2000 in constraints[cc]:
            quota = constraints[cc][2000]['total_pop']
        else:
            # 兜底逻辑：如果CSV没这个城市，按等级给个小值
            level_w = 1.0 if cc in TIER1_CITIES else (0.5 if cc in TIER2_CITIES else 0.1)
            quota = int(TOTAL_POPULATION_BASE * 0.001 * level_w)

        shuffled = types_in_city.copy()
        random.Random(hash(cc) % (2**31)).shuffle(shuffled)

        n = len(shuffled)

        # 1. 基础权重 (幂律分布)
        base_weights = np.array([(i+1)**(-POWER_LAW_ALPHA) for i in range(n)])

        # 2. 结合行业/特征乘数
        type_multipliers = np.array([_calculate_type_multiplier(td) for td in shuffled])

        # 3. 最终组合权重
        combined_weights = base_weights * type_multipliers

        # [关键修复]：归一化，确保权重总和为 1
        combined_weights = combined_weights / combined_weights.sum()

        # 4. 按照归一化后的权重分配人口
        raw_counts = quota * combined_weights

        type_counts = {}
        total_allocated = 0

        for i, td in enumerate(shuffled):
            count = int(raw_counts[i])
            if count >= MIN_TYPE_COUNT:
                type_id = type_to_id(td)
                type_counts[type_id] = count
                total_allocated += count

        # 5. 尾差补偿：由于取整可能导致总数 < quota，将差额补给最大的Type
        deficit = quota - total_allocated
        if deficit > 0 and type_counts:
            # 找到人口最多的Type，把差额补给它
            max_type_id = max(type_counts.keys(), key=lambda k: type_counts[k])
            type_counts[max_type_id] += deficit

        city_state[cc] = type_counts

    return city_state

def recover_state_from_db(conn, year):
    """
    从数据库恢复特定年份12月的人口状态（用于断点续传）

    参数:
        conn: DuckDB连接
        year: 要恢复的年份

    返回:
        recovered_state: {city_code: {type_id: count}} 该年12月的人口分布
    """
    print(f"  [断点恢复] 正在从数据库恢复 {year} 年底的人口状态...")

    # 查询指定年份12月的所有Type分布
    query = f"""
        SELECT Region, Type_ID, SUM(Total_Count) as Total_Count
        FROM migration_data
        WHERE Year = {year} AND Month = 12
        GROUP BY Region, Type_ID
    """
    df = conn.execute(query).df()

    recovered_state = {}
    total_recovered = 0

    for _, row in df.iterrows():
        cc = str(row['Region'])
        tid = row['Type_ID']
        cnt = int(row['Total_Count'])

        if cc not in recovered_state:
            recovered_state[cc] = {}
        recovered_state[cc][tid] = cnt
        total_recovered += cnt

    print(f"  [断点恢复] 成功恢复 {len(recovered_state)} 个城市，{total_recovered:,} 人口")

    return recovered_state

def calibrate_city_state(city_state, constraints, year):
    """
    约束校准（修复版）：强制对齐CSV的total_pop约束，并修复截断导致的人口泄漏
    """
    calibrated_state = {}

    for city_code, type_counts in city_state.items():
        # 1. 获取约束目标
        if city_code in constraints and year in constraints[city_code]:
            target_pop = constraints[city_code][year]['total_pop']
        else:
            # 无约束城市保持原样
            calibrated_state[city_code] = type_counts.copy()
            continue

        # 2. 计算当前模拟数据的总人口
        actual_pop = sum(type_counts.values())
        if actual_pop == 0:
            calibrated_state[city_code] = {}
            continue

        # 3. 计算缩放比例
        ratio = target_pop / actual_pop

        # 4. 缩放所有Type
        new_type_counts = {}
        total_allocated = 0

        # 临时存储被丢弃的微小人口，用于最后统计损失
        dropped_pop = 0

        for type_id, count in type_counts.items():
            new_count = int(count * ratio)

            # 【关键修改】如果小于阈值，不直接丢弃，而是计入损失，稍后回补
            if new_count >= MIN_TYPE_COUNT:
                new_type_counts[type_id] = new_count
                total_allocated += new_count
            else:
                dropped_pop += new_count

        # 5. 【核心修复】计算总缺口（包含取整误差 + 截断损失）
        # 目标是 target_pop，目前只分配了 total_allocated
        deficit = target_pop - total_allocated

        # 6. 将缺口补给该城市人口最多的 Type (强者恒强，避免碎片化)
        if deficit > 0 and new_type_counts:
            # 找到当前人口最多的 Type ID
            max_type_id = max(new_type_counts, key=new_type_counts.get)
            new_type_counts[max_type_id] += deficit
        elif deficit > 0 and not new_type_counts:
            # 极端情况：所有Type都被截断了（城市太小），强制保留一个最大的
            if type_counts:
                max_orig_id = max(type_counts, key=type_counts.get)
                new_type_counts[max_orig_id] = target_pop

        # 7. Type出现逻辑（保持原样，仅在人口稳定后尝试引入新Type）
        # ... (此处省略原本的 Type 出现逻辑代码，如果不需要新Type生成可不写，或者保留你原来的代码) ...
        # 为了代码简洁，建议只保留上面的校准补差逻辑，新Type生成对总人口影响极小可忽略

        calibrated_state[city_code] = new_type_counts

    return calibrated_state

# ==============================================================================
# 5. 迁移模型
# ==============================================================================

class MigrationModel:
    def __init__(self, random_seed=None):
        self.rng = np.random.RandomState(random_seed)

    def calculate_base_migration_prob(self, type_dict, month=None):
        base = AGE_MIGRATION_BASE.get(type_dict['D2'], 0.15)
        multipliers = [
            GENDER_MIGRATION_MULTIPLIER.get(type_dict['D1'], 1.0),
            EDU_MIGRATION_MULTIPLIER.get(type_dict['D3'], 1.0),
            INDUSTRY_MIGRATION_MULTIPLIER.get(type_dict['D4'], 1.0),
            INCOME_MIGRATION_MULTIPLIER.get(type_dict['D5'], 1.0),
            FAMILY_MIGRATION_MULTIPLIER.get(type_dict['D6'], 1.0)
        ]
        prob = base * np.prod(multipliers)
        if month:
            prob *= SEASONAL_FACTORS.get(month, 1.0)
        prob *= (1.0 + self.rng.uniform(-0.05, 0.05))
        return np.clip(prob, MIGRATION_PROB_MIN, MIGRATION_PROB_MAX)

    def calculate_migration_targets(self, from_city_code, type_dict, migration_prob, year_targets, total_target_in_year, row_seed=None):
        """
        计算迁移目标城市概率 (修正版：去除双重加成，引入一线城市拥堵惩罚)

        关键修复：
        1. 【去除双重加权】CSV存量已包含吸引力，不再乘产业匹配因子
        2. 【引入拥堵惩罚】一线城市强制降权，防止吸血过猛
        3. 【非线性平滑】用幂律压缩，保留差距但削弱超级巨头优势

        参数:
            year_targets: dict, {city_code: target_inmigration_stock}
                         该年全国所有城市的迁入存量字典（用作权重）
            total_target_in_year: float, 全国总存量 (用于归一化)
        """
        row_rng = np.random.RandomState(row_seed) if row_seed is not None else self.rng

        # 1. 准备候选池 + 应用拥堵惩罚
        valid_targets = []

        # 定义拥堵惩罚系数：一线城市强制降权，防止吸血过猛
        # 原因：CSV存量已经体现了吸引力，算法中的TopN采样会再次放大头部效应
        # 需要通过惩罚系数来抵消这种"马太效应"
        CONGESTION_PENALTY = {
            '1100': 0.45,  # 北京（存量极大，需要大幅降权）
            '3100': 0.45,  # 上海（同上）
            '4403': 0.50,  # 深圳
            '4401': 0.60,  # 广州
            '5000': 0.80,  # 重庆（直辖市，但相对没那么极端）
            '3301': 0.85,  # 杭州（强二线，轻微降权）
            '3201': 0.85,  # 南京
            '4201': 0.85,  # 武汉
        }

        # 遍历所有有存量数据的城市
        for c_code, t_val in year_targets.items():
            if c_code == from_city_code or t_val <= 0:
                continue

            # 【核心修改 1】应用拥堵惩罚 (针对头部城市降权)
            penalty = CONGESTION_PENALTY.get(c_code, 1.0)

            # 【核心修改 2】非线性平滑 (Power Law Damping)
            # 将巨大的存量差异稍微压扁，让长尾城市有机会
            # 0.9 次幂：保留大部分差距，但削弱超级巨头的绝对优势
            # 例如：100^0.9 ≈ 63, 10^0.9 ≈ 8 (差距从10倍缩小到8倍)
            adjusted_weight = (t_val ** 0.9) * penalty

            valid_targets.append((c_code, adjusted_weight))

        if not valid_targets:
            return [('Other', '其他', migration_prob)]

        # 解压
        target_codes = [x[0] for x in valid_targets]
        target_weights = np.array([x[1] for x in valid_targets], dtype=float)

        # 2. 归一化权重 (用于采样)
        sum_weights = target_weights.sum()
        if sum_weights > 0:
            sample_probs = target_weights / sum_weights
        else:
            sample_probs = np.ones(len(target_weights)) / len(target_weights)

        # 3. 选取 Top N (加权随机采样)
        k = min(TOP_N_CITIES, len(target_codes))
        chosen_indices = row_rng.choice(len(target_codes), size=k, replace=False, p=sample_probs)

        result = []
        city_dict = dict(CITIES)

        # 4. 计算最终概率（份额法）
        # 计算被选中的 TopN 的总权重
        chosen_weights_sum = sum(target_weights[i] for i in chosen_indices)

        for idx in chosen_indices:
            c_code = target_codes[idx]
            c_weight = target_weights[idx]
            c_name = city_dict.get(c_code, 'Unknown')

            # 【核心修改 3】不再乘 match_factor
            # 既然 CSV 存量已经包含了所有吸引力因素（GDP、产业、教育等），
            # 就不应该再乘产业匹配度，否则是"双重加成"
            # 直接按权重比例分配 migration_prob

            if chosen_weights_sum > 0:
                # 份额法：该城市在 TopN 中的权重占比 * 总迁移率
                prob = migration_prob * (c_weight / chosen_weights_sum)
            else:
                prob = 0.0

            # 【微调】极小幅度随机扰动（避免完全确定性）
            # 幅度控制在 ±5% 以内，不破坏大趋势
            noise = row_rng.uniform(0.95, 1.05)
            prob *= noise

            result.append((c_code, c_name, prob))

        # 按概率排序
        result.sort(key=lambda x: x[2], reverse=True)

        # 修正总概率 (防止因噪声导致溢出)
        final_result = []
        current_sum = 0.0
        for c, n, p in result:
            if current_sum >= migration_prob:
                break  # 已达到总概率，后续设为0
            if current_sum + p > migration_prob:
                p = migration_prob - current_sum  # 截断，防止溢出
            final_result.append((c, n, p))
            current_sum += p

        return final_result

# ==============================================================================
# 6. 多进程工作函数（并行化：按城市chunk处理）
# ==============================================================================

def process_city_chunk(city_chunk_state, year, type_id_to_dict, global_outflow_scaler, year_targets, total_target_in_year):
    """
    处理一组城市的数据生成 (修改:接收全局缩放和迁入目标)

    核心修改:
    1. 【全局供需平衡】使用 global_outflow_scaler 调整全国迁出率
    2. 【目标导向引力】使用 year_targets 决定各城市的吸引力
    3. 【移除本地迁出约束】不再使用单个城市的迁出目标
    4. 【全局归一化】使用 total_target_in_year 进行概率归一化

    参数:
        city_chunk_state: {city_code: {type_id: count}} 一组城市的人口状态
        year: 当前年份
        type_id_to_dict: {type_id: type_dict}
        global_outflow_scaler: float, 全局迁出率缩放因子 (用于满足全国总迁入需求)
        year_targets: dict, {city_code: target_inmigration} 当年的迁入目标字典
        total_target_in_year: float, 全国总迁入目标 (用于全局归一化)

    返回:
        DataFrame: 生成的数据行（TopN列强制为字符串类型）
    """
    # 预计算季节因子
    seasonal_factors_arr = np.array([SEASONAL_FACTORS[m] for m in OUTPUT_MONTHS])

    # 初始化列式容器
    cols = {
        'Year': [], 'Month': [], 'Type_ID': [], 'Region': [], 'From_City': [],
        'Total_Count': [], 'Stay_Prob': [], 'To_Other_Prob': []
    }
    for i in range(TOP_N_CITIES):
        cols[f'To_Top{i+1}'] = []
        cols[f'To_Top{i+1}_Prob'] = []

    # 随机数生成
    first_city = list(city_chunk_state.keys())[0] if city_chunk_state else "None"
    rng = np.random.RandomState(year + hash(first_city) % 10000)

    city_code_to_name_str = {c[0]: f"{c[1]}({c[0]})" for c in CITIES}
    local_model = MigrationModel(random_seed=year)

    for city_code, type_counts in city_chunk_state.items():
        from_city_str = city_code_to_name_str.get(city_code, city_code)

        # 遍历该城市所有人群
        for type_id, count in type_counts.items():
            if type_id not in type_id_to_dict:
                continue

            type_dict = type_id_to_dict[type_id]
            base_count = float(count)

            # 1. 计算原始迁出意愿
            raw_mig_prob = local_model.calculate_base_migration_prob(type_dict, month=None)

            # 2. 【关键】应用全局缩放因子
            # 如果全国总迁入目标很高，我们需要让更多人动起来
            adjusted_base_prob = raw_mig_prob * global_outflow_scaler

            # 3. 计算去向 (传入 year_targets 和 total_target_in_year 以决定引力)
            # 使用 base_count 作为 row_seed 的一部分确保确定性
            row_seed = int(year * 1000 + int(city_code) + base_count % 100)
            targets = local_model.calculate_migration_targets(
                city_code, type_dict, adjusted_base_prob, year_targets, total_target_in_year, row_seed
            )

            target_list = []
            other_prob_base = 0.0
            for t_code, t_name, t_prob in targets:
                if t_code == 'Other':
                    other_prob_base = t_prob
                else:
                    target_list.append((f"{t_name}({t_code})", t_prob))

            # 使用 base_count 作为全年的总数约束
            count_val = int(base_count)

            if count_val > 0:
                # 不再受季节因子影响，直接使用 adjusted_base_prob
                cur_mig_prob = np.clip(adjusted_base_prob, MIGRATION_PROB_MIN, 0.95)
                stay_prob = 1.0 - cur_mig_prob

                # 因为 targets 的总和本身就等于 adjusted_base_prob，所以缩放比例为 1.0
                target_scale = 1.0

                cols['Year'].append(year)
                # 【关键修改】Month 设为 12，匹配验证脚本
                cols['Month'].append(12)
                cols['Type_ID'].append(type_id)
                cols['Region'].append(city_code)
                cols['From_City'].append(from_city_str)
                cols['Total_Count'].append(count_val)
                cols['Stay_Prob'].append(round(stay_prob, 4))

                for i in range(TOP_N_CITIES):
                    col_target = f'To_Top{i+1}'
                    col_prob = f'To_Top{i+1}_Prob'
                    if i < len(target_list):
                        t_str, t_p_base = target_list[i]
                        cols[col_target].append(str(t_str))
                        cols[col_prob].append(round(t_p_base * target_scale, 4))
                    else:
                        cols[col_target].append('')
                        cols[col_prob].append(0.0)

                cols['To_Other_Prob'].append(round(other_prob_base * target_scale, 4))

    # 生成 DataFrame
    df = pd.DataFrame(cols)

    # 【关键修复】强制转换 TopN 列为字符串类型，防止 DuckDB 误判为 Float
    if not df.empty:
        for i in range(TOP_N_CITIES):
            col_name = f'To_Top{i+1}'
            df[col_name] = df[col_name].astype(str)

    return df

# ==============================================================================
# 7. 主程序
# ==============================================================================

def check_and_print_db_schema(conn):
    """
    检查并打印数据库schema和样例数据
    用于诊断类型问题
    """
    print("\n" + "="*80)
    print("数据库Schema检查")
    print("="*80)

    # 检查表是否存在
    table_check = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'migration_data'"
    ).fetchone()[0]

    if table_check == 0:
        print("❌ migration_data 表不存在")
        return

    print("✓ migration_data 表存在\n")

    # 获取表结构
    print("表结构:")
    print("-" * 80)
    schema_df = conn.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'migration_data'
        ORDER BY ordinal_position
    """).df()

    for _, row in schema_df.iterrows():
        print(f"  {row['column_name']:20s} : {row['data_type']}")

    # 检查TopN列的数据类型
    print("\n" + "-" * 80)
    print("TopN列类型详情:")
    for i in range(1, TOP_N_CITIES + 1):
        col_name = f'To_Top{i}'
        result = conn.execute(f"""
            SELECT {col_name}, typeof({col_name})
            FROM migration_data
            WHERE {col_name} IS NOT NULL AND {col_name} != ''
            LIMIT 1
        """).fetchone()

        if result:
            print(f"  {col_name}: 示例值='{result[0]}', DuckDB类型={result[1]}")
        else:
            print(f"  {col_name}: 无数据")

    # 打印样例数据
    print("\n" + "-" * 80)
    print("样例数据（前5行）:")
    sample_df = conn.execute("SELECT * FROM migration_data LIMIT 5").df()
    print(sample_df.to_string())

    # 检查是否存在类型不一致
    print("\n" + "-" * 80)
    print("类型一致性检查:")

    for i in range(1, TOP_N_CITIES + 1):
        col_name = f'To_Top{i}'
        type_check = conn.execute(f"""
            SELECT typeof({col_name}) as type_val, COUNT(*) as cnt
            FROM migration_data
            WHERE {col_name} IS NOT NULL AND {col_name} != ''
            GROUP BY typeof({col_name})
        """).df()

        if len(type_check) > 0:
            types_str = ", ".join([f"{row['type_val']}({row['cnt']}行)" for _, row in type_check.iterrows()])
            print(f"  {col_name}: {types_str}")

    print("="*80 + "\n")


def main():
    print("=== 人口迁移数据生成器 (动态演化版 2000-2020) ===")
    start_time = time.time()

    # 0. 初始化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)

    # [断点续传] 检查数据库是否存在，不再自动删除
    db_exists = os.path.exists(db_path)

    # 1. 读取约束
    global CITY_CONSTRAINTS
    CITY_CONSTRAINTS = load_and_interpolate_constraints(CONSTRAINT_CSV_PATH)

    # 2. 生成所有可能的Type
    print("\n1. 生成所有Type组合...")
    all_base_types = generate_all_types()
    print(f"   基础维度组合: {len(all_base_types)}")
    print(f"   理论总Type实例 (组合 x 城市): {len(all_base_types) * len(CITIES):,}")

    # 3. [断点续传] 检查断点并决定起始年份
    conn = duckdb.connect(db_path)
    start_year = 2000
    table_created = False
    total_rows = 0

    if db_exists:
        # 【新增】检查并打印数据库schema（诊断类型问题）
        check_and_print_db_schema(conn)

        # 检查表是否存在
        table_check = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'migration_data'"
        ).fetchone()[0]

        if table_check > 0:
            # 表存在，检查最大年份
            max_year_result = conn.execute("SELECT MAX(Year) FROM migration_data").fetchone()
            max_year = max_year_result[0] if max_year_result else None

            if max_year is not None:
                if max_year >= 2020:
                    print("\n[提示] 检测到数据已全部生成（至2020年），无需继续。")
                    conn.close()
                    return

                # 从下一年继续
                start_year = max_year + 1
                table_created = True

                # 获取当前总行数
                total_rows = conn.execute("SELECT COUNT(*) FROM migration_data").fetchone()[0]

                print(f"\n[断点续传] 检测到已有数据，最大年份: {max_year}，将从 {start_year} 年继续演化")
                print(f"[断点续传] 当前进度: {total_rows:,} 行")

                # 【关键】从数据库恢复状态
                current_state = recover_state_from_db(conn, max_year)

                # 构建type_id_to_dict映射
                city_types = []
                for city_code, _ in CITIES:
                    for base_type in all_base_types:
                        t = base_type.copy()
                        t['D7'] = city_code
                        city_types.append(t)
                type_id_to_dict = {type_to_id(td): td for td in city_types}

                print(f"[断点续传] 状态恢复完成，准备开始 {start_year} 年的演化\n")
            else:
                # 表存在但无数据
                print("\n[提示] 检测到数据库表存在但无数据，将从头开始生成")
                conn.close()
                conn = duckdb.connect(db_path)
                conn.execute("DROP TABLE IF EXISTS migration_data")
        else:
            # 表不存在
            print("\n[提示] 数据库存在但无migration_data表，将从头开始生成")
    else:
        print("\n[提示] 数据库不存在，将从头开始生成")

    # 4. 如果从头开始，执行2000年初始化
    if start_year == 2000:
        print("\n2. 2000年初始化人口分布...")
        city_types = []
        for city_code, _ in CITIES:
            for base_type in all_base_types:
                t = base_type.copy()
                t['D7'] = city_code
                city_types.append(t)

        current_state = initialize_population_2000(city_types, CITY_CONSTRAINTS)

        # 统计2000年初始人口
        total_pop_2000 = sum(sum(types.values()) for types in current_state.values())
        print(f"   2000年总人口: {total_pop_2000:,}")

        # [新增] 验证2000年人口是否与CSV一致
        csv_total_2000 = sum(
            CITY_CONSTRAINTS[cc][2000]['total_pop']
            for cc in CITY_CONSTRAINTS
            if 2000 in CITY_CONSTRAINTS[cc]
        )
        print(f"   CSV约束2000年总人口: {csv_total_2000:,}")
        print(f"   偏差: {abs(total_pop_2000 - csv_total_2000):,} ({abs(total_pop_2000 - csv_total_2000) / csv_total_2000 * 100:.2f}%)")

        # [新增] 如果偏差超过1%，立即校准
        if abs(total_pop_2000 - csv_total_2000) / csv_total_2000 > 0.01:
            print(f"   [警告] 初始化偏差超过1%，执行强制校准...")
            current_state = calibrate_city_state(current_state, CITY_CONSTRAINTS, 2000)
            total_pop_2000_after = sum(sum(types.values()) for types in current_state.values())
            print(f"   校准后2000年总人口: {total_pop_2000_after:,}")

        # 构建type_id_to_dict映射
        type_id_to_dict = {type_to_id(td): td for td in city_types}

    # 5. 逐年演化（只处理剩余年份）
    REMAINING_YEARS = [y for y in OUTPUT_YEARS if y >= start_year]

    print(f"\n3. 开始动态演化（{REMAINING_YEARS[0]}-{REMAINING_YEARS[-1]}）...")

    # [优化] 设置并行参数 - 激进配置，提升CPU占用率
    num_workers = min(28, cpu_count() - 2)  # i7-14700可以使用更多进程，只预留2核给系统
    print(f"   使用 {num_workers} 个进程并行处理每个年份的城市数据（激进配置）\n")

    for year in REMAINING_YEARS:
        print(f"处理年份: {year}")

        # 步骤1：当年初校准（除了2000年）
        if year != 2000:
            current_state = calibrate_city_state(current_state, CITY_CONSTRAINTS, year)

        # 统计当前总人口
        current_total = sum(sum(types.values()) for types in current_state.values())
        print(f"  {year}年初总人口: {current_total:,}")

        # --- 【核心修改】步骤2: 存量定方向，费率定总量 ---
        # 关键理解：CSV中的"迁入人口"是存量（十年的累积），不是年流量
        # 我们用存量来计算"吸引力权重"（决定流向），用费率来决定"总量"（决定多少人搬家）

        # 1. 读取 CSV 数据作为"吸引力权重" (Attractiveness Weights)
        year_targets = {}  # {city_code: target_inmigration_weight}
        target_weight_sum = 0

        for cc, c_data in CITY_CONSTRAINTS.items():
            if year in c_data:
                # 这里的 target_inmigration 是存量 (Stock)，用作权重
                t = c_data[year].get('target_inmigration', 0)
                if t is not None and t > 0:
                    year_targets[cc] = t
                    target_weight_sum += t

        # 2. 设定"年度总迁移率" (Yearly Migration Rate)
        # 中国年均跨市流动人口比例约为总人口的 4% - 6%
        # 2000-2010年流动性高(6%)，2015年后趋缓(4%)
        annual_migration_rate = 0.06 if year < 2015 else 0.04

        # 3. 估算模型原本的平均迁移意愿
        # 根据 AGE_MIGRATION_BASE 参数估算：各年龄段平均约 (0.08+0.06+0.03+0.01+0.005)/5 ≈ 0.045
        # 考虑其他乘数（Edu, Industry等），综合约 0.05-0.06
        estimated_model_base_prob = 0.05

        # 4. 计算缩放因子：让最终迁移率 = annual_migration_rate
        # 公式：global_outflow_scaler = annual_migration_rate / estimated_model_base_prob
        global_outflow_scaler = annual_migration_rate / estimated_model_base_prob

        # 5. 兜底逻辑：如果当年没有约束数据
        if target_weight_sum == 0:
            # 如果当年没数据，所有城市权重设为1（平均分布）
            year_targets = {c[0]: 1.0 for c in CITIES}
            target_weight_sum = len(CITIES)

        # 6. 归一化权重总和（用于概率计算）
        total_target_in_safe = float(target_weight_sum)

        print(f"  {year}年迁移模型设定:")
        print(f"    设定年度总迁移率: {annual_migration_rate*100:.1f}% (符合中国国情)")
        print(f"    模型基础迁移率: {estimated_model_base_prob*100:.1f}% (参数估算)")
        print(f"    全局缩放因子: {global_outflow_scaler:.4f}")
        print(f"    有存量约束的城市数: {len(year_targets)}")
        print(f"  【逻辑】存量定方向（去哪里），费率定总量（多少人搬家）")
        # 【核心修改结束】

        # 步骤3：[并行化] 将城市列表切分为多个chunk
        all_cities_in_state = list(current_state.keys())
        chunks = np.array_split(all_cities_in_state, num_workers)

        # [优化] 移除year_rows中间变量，直接写入DuckDB
        year_row_count = 0

        # 使用进程池并行处理各城市chunk
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = []
            for chunk in chunks:
                # 准备该进程需要的局部状态
                chunk_state = {cc: current_state[cc] for cc in chunk}
                # 【修改】传入 global_outflow_scaler, year_targets, total_target_in_safe
                futures.append(executor.submit(
                    process_city_chunk,
                    chunk_state, year, type_id_to_dict,
                    global_outflow_scaler, year_targets, total_target_in_safe
                ))

            # 添加实时进度条
            with tqdm(total=len(futures), desc=f"  {year}年并行计算", unit="chunk", leave=False) as pbar:
                for future in as_completed(futures):
                    try:
                        # [优化] 直接获取DataFrame，不转dict
                        df_chunk = future.result()

                        if not df_chunk.empty:
                            # [极致优化] 直接写入DuckDB，避免内存堆积
                            conn.register('temp_chunk', df_chunk)
                            if not table_created:
                                conn.execute("CREATE TABLE migration_data AS SELECT * FROM temp_chunk")
                                table_created = True
                            else:
                                conn.execute("INSERT INTO migration_data SELECT * FROM temp_chunk")
                            conn.unregister('temp_chunk')

                            year_row_count += len(df_chunk)
                    except Exception as e:
                        print(f"\n  [错误] Chunk处理失败: {e}")
                    finally:
                        pbar.update(1)

        # 步骤4：统计年度完成情况
        if year_row_count > 0:
            total_rows += year_row_count
            print(f"  ✓ {year}年完成，共写入 {year_row_count:,} 行\n")

            # 【新增 只用于debug长尾分布 控制初始化TYPE用】2000年特殊统计：打印唯一的Type数量
            if year == 2000:
                print("\n" + "="*80)
                print("2000年 Type 分布统计")
                print("="*80)

                # 1. 统计全局唯一Type数量
                unique_types_df = conn.execute(f"""
                    SELECT COUNT(DISTINCT Type_ID) as unique_type_count
                    FROM migration_data
                    WHERE Year = 2000
                """).df()
                unique_type_count = unique_types_df['unique_type_count'][0]
                print(f"\n1. 全局唯一Type数量: {unique_type_count:,}")

                # 2. 统计每个城市的Type数量
                city_type_counts = conn.execute(f"""
                    SELECT
                        Region,
                        From_City,
                        COUNT(DISTINCT Type_ID) as type_count,
                        SUM(Total_Count) as total_pop
                    FROM migration_data
                    WHERE Year = 2000
                    GROUP BY Region, From_City
                    ORDER BY type_count DESC
                """).df()

                print(f"\n2. 各城市Type数量统计（按Type数量降序，前20名）：")
                print(f"   {'排名':<6} {'城市代码':<10} {'城市名称':<20} {'Type数量':<12} {'总人口':<15}")
                print(f"   {'-'*6} {'-'*10} {'-'*20} {'-'*12} {'-'*15}")
                for idx, row in city_type_counts.head(20).iterrows():
                    print(f"   {idx+1:<6} {row['Region']:<10} {row['From_City']:<20} {row['type_count']:<12,} {row['total_pop']:<15,}")

                # 3. Type数量分布统计
                type_distribution = city_type_counts['type_count'].describe()
                print(f"\n3. Type数量分布统计:")
                print(f"   最小值: {type_distribution['min']:.0f}")
                print(f"   25%分位: {type_distribution['25%']:.0f}")
                print(f"   中位数: {type_distribution['50%']:.0f}")
                print(f"   平均值: {type_distribution['mean']:.1f}")
                print(f"   75%分位: {type_distribution['75%']:.0f}")
                print(f"   最大值: {type_distribution['max']:.0f}")

                # 4. 计算理论最大Type数量
                max_possible_types = len(all_base_types)  # 360个基础Type
                cities_with_data = len(city_type_counts)
                theoretical_max = max_possible_types * cities_with_data
                coverage_ratio = unique_type_count / theoretical_max * 100

                print(f"\n4. Type覆盖率分析:")
                print(f"   理论最大Type数 (基础维度组合): {max_possible_types:,}")
                print(f"   有数据的城市数: {cities_with_data}")
                print(f"   理论最大Type实例 (组合×城市): {theoretical_max:,}")
                print(f"   实际生成Type实例: {unique_type_count:,}")
                print(f"   覆盖率: {coverage_ratio:.2f}%")
                print(f"   过滤掉的长尾Type: {theoretical_max - unique_type_count:,} ({(theoretical_max - unique_type_count)/theoretical_max*100:.1f}%)")

                # 5. 验证是否满足MIN_TYPE_COUNT约束
                types_below_threshold = conn.execute(f"""
                    SELECT COUNT(*) as count
                    FROM (
                        SELECT Type_ID, SUM(Total_Count) as total
                        FROM migration_data
                        WHERE Year = 2000
                        GROUP BY Type_ID
                        HAVING total < {MIN_TYPE_COUNT}
                    )
                """).fetchone()[0]

                print(f"\n5. MIN_TYPE_COUNT约束验证:")
                print(f"   阈值设定: {MIN_TYPE_COUNT:,}")
                print(f"   低于阈值的Type数量: {types_below_threshold}")
                if types_below_threshold == 0:
                    print(f"   ✓ 所有Type都满足最小人口约束")
                else:
                    print(f"   ⚠ 警告: 存在 {types_below_threshold} 个Type低于阈值")

                print("="*80 + "\n")

        # 步骤5：年底再次校准（确保下一年初的状态符合约束）
        if year < 2020:
            current_state = calibrate_city_state(current_state, CITY_CONSTRAINTS, year + 1)

    # 5. 创建索引
    if table_created:
        print(f"\n4. 创建索引 (总行数: {total_rows:,})...")
        conn.execute("CREATE INDEX idx_type_id ON migration_data(Type_ID)")
        conn.execute("CREATE INDEX idx_city ON migration_data(From_City)")
        conn.execute("CREATE INDEX idx_year_month ON migration_data(Year, Month)")

        cnt = conn.execute("SELECT COUNT(*) FROM migration_data").fetchone()[0]
        print(f"   数据库验证: {cnt:,} 行")

        # 6. 生成采样CSV
        print("\n5. 生成100条随机记录采样...")
        sample_df = conn.execute("SELECT * FROM migration_data ORDER BY RANDOM() LIMIT 100").df()
        csv_path = os.path.join(OUTPUT_DIR, CSV_SAMPLE_FILENAME)
        sample_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   采样CSV已保存: {csv_path}")

        # 7. 验证约束（修复：直接对比，不乘季节因子，因为 calibrate 已经对齐了总数）
        print("\n6. 验证约束符合度（查询12月存量）...")
        print("   【说明】calibrate_city_state 已强制让生成的 Total_Count 等于 CSV 的 total_pop")
        print("   【说明】因此验证时直接对比，不需要乘季节因子（季节因子仅用于迁移概率计算）")
        for city_code in list(CITY_CONSTRAINTS.keys())[:5]:
            for year in [2000, 2010, 2020]:
                result = conn.execute(f"""
                    SELECT SUM(Total_Count) as total
                    FROM migration_data
                    WHERE Region = '{city_code}' AND Year = {year} AND Month = 12
                """).fetchone()

                generated = result[0] if result[0] else 0
                # 【修改点】直接取 total_pop，不要乘季节因子
                target = CITY_CONSTRAINTS[city_code][year]['total_pop']

                diff_pct = abs(generated - target) / target * 100 if target > 0 else 0
                print(f"   {city_code} {year}年12月: 目标={target:,.0f}, 生成={generated:,}, 偏差={diff_pct:.1f}%")

        # 验证全国总人口（12月）
        print("\n7. 验证全国总人口（12月）...")
        for year in [2000, 2010, 2020]:
            result = conn.execute(f"""
                SELECT SUM(Total_Count) as total
                FROM migration_data
                WHERE Year = {year} AND Month = 12
            """).fetchone()

            generated = result[0] if result[0] else 0

            national_target = sum(
                CITY_CONSTRAINTS[cc][year]['total_pop']
                for cc in CITY_CONSTRAINTS
                if year in CITY_CONSTRAINTS[cc]
            )
            # 【修改点】直接取 national_target，不要乘季节因子
            expected_dec = national_target

            diff_pct = abs(generated - expected_dec) / expected_dec * 100 if expected_dec > 0 else 0
            print(f"   全国{year}年12月: 目标={expected_dec:,.0f}, 生成={generated:,}, 偏差={diff_pct:.1f}%")

        # 【重写】验证迁徙分布 - 验证吸引力份额占比
        # 关键理解：CSV是存量（十年的累积），生成是流量（一年的）
        # 我们不验证绝对数值，而是验证"相对占比"是否一致
        print("\n8. 验证迁徙分布（验证城市吸引力份额占比）...")
        print("   【核心逻辑】验证 (某城市迁入量 / 全国总迁入量) 是否与 CSV存量占比一致")
        print("   【重要说明】CSV中的'迁入总人口'是'外来人口存量'（十年的累积）")
        print("   【重要说明】我们模拟的是'当年流量'（一年的新迁入人口）")
        print("   【重要说明】因此绝对数值不可比（存量 >> 流量），但相对占比应该一致")
        print("   【验证目标】如果CSV显示北京占15%，则生成的流量中北京也应该占15%左右\n")

        # 验证所有年份
        validation_years = [y for y in OUTPUT_YEARS if y in [2000, 2010, 2020]]

        for year in validation_years:
            print(f"{'='*120}")
            print(f"--- {year}年 城市吸引力份额验证（全量城市，按偏差倒序）---")
            print(f"{'='*120}\n")

            # 1. 计算 CSV 中的总存量（分母）
            csv_total_stock = 0
            for cc in CITY_CONSTRAINTS:
                if year in CITY_CONSTRAINTS[cc]:
                    t = CITY_CONSTRAINTS[cc][year].get('target_inmigration', 0)
                    if t is not None and t > 0:
                        csv_total_stock += t

            if csv_total_stock == 0:
                print(f"  跳过 {year}年：无存量数据\n")
                continue

            # 2. 计算生成数据中的总流量（分母）
            # 查询所有城市的实际迁入量总和
            total_flow_query = f"""
                WITH extracted_targets AS (
                    SELECT
                        UNNEST([
                            regexp_extract(To_Top1, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top2, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top3, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top4, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top5, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top6, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top7, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top8, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top9, '\\(([^)]+)\\)', 1),
                            regexp_extract(To_Top10, '\\(([^)]+)\\)', 1)
                        ]) AS target_city_code,
                        UNNEST([
                            To_Top1_Prob, To_Top2_Prob, To_Top3_Prob, To_Top4_Prob, To_Top5_Prob,
                            To_Top6_Prob, To_Top7_Prob, To_Top8_Prob, To_Top9_Prob, To_Top10_Prob
                        ]) AS target_prob,
                        Total_Count
                    FROM migration_data
                    WHERE Year = {year}
                )
                SELECT SUM(Total_Count * target_prob) as total_flow
                FROM extracted_targets
            """
            total_flow_result = conn.execute(total_flow_query).fetchone()
            gen_total_flow = float(total_flow_result[0]) if total_flow_result and total_flow_result[0] else 0.0

            if gen_total_flow == 0:
                print(f"  跳过 {year}年：生成的流量为0\n")
                continue

            # 3. 计算所有城市的份额占比
            city_results = []

            for city_code in CITY_CONSTRAINTS.keys():
                if year not in CITY_CONSTRAINTS[city_code]:
                    continue

                # 获取城市名称
                city_name = next((c[1] for c in CITIES if c[0] == city_code), 'Unknown')

                # CSV存量占比
                csv_stock = CITY_CONSTRAINTS[city_code][year].get('target_inmigration', 0)
                if csv_stock is None or csv_stock == 0:
                    continue
                csv_share = csv_stock / csv_total_stock * 100

                # 生成数据流量占比
                city_flow_query = f"""
                    WITH extracted_targets AS (
                        SELECT
                            UNNEST([
                                regexp_extract(To_Top1, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top2, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top3, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top4, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top5, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top6, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top7, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top8, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top9, '\\(([^)]+)\\)', 1),
                                regexp_extract(To_Top10, '\\(([^)]+)\\)', 1)
                            ]) AS target_city_code,
                            UNNEST([
                                To_Top1_Prob, To_Top2_Prob, To_Top3_Prob, To_Top4_Prob, To_Top5_Prob,
                                To_Top6_Prob, To_Top7_Prob, To_Top8_Prob, To_Top9_Prob, To_Top10_Prob
                            ]) AS target_prob,
                            Total_Count
                        FROM migration_data
                        WHERE Year = {year}
                    )
                    SELECT SUM(Total_Count * target_prob) as city_flow
                    FROM extracted_targets
                    WHERE target_city_code = '{city_code}'
                """
                city_flow_result = conn.execute(city_flow_query).fetchone()
                gen_flow = float(city_flow_result[0]) if city_flow_result and city_flow_result[0] else 0.0
                gen_share = gen_flow / gen_total_flow * 100

                # 计算份额偏差
                share_diff = gen_share - csv_share  # 有正负，表示高估或低估
                share_diff_abs = abs(share_diff)

                # 状态标记
                status = "✓" if share_diff_abs < 2 else ("⚠" if share_diff_abs < 5 else "✗")

                city_results.append({
                    'code': city_code,
                    'name': city_name,
                    'csv_share': csv_share,
                    'gen_share': gen_share,
                    'share_diff': share_diff,
                    'share_diff_abs': share_diff_abs,
                    'csv_stock': csv_stock,
                    'gen_flow': gen_flow,
                    'status': status
                })

            # 4. 按份额偏差绝对值倒序排序
            city_results.sort(key=lambda x: x['share_diff_abs'], reverse=True)

            # 5. 打印汇总信息
            print(f"  全国总存量: {csv_total_stock:,.0f} | 全国年流量: {gen_total_flow:,.0f} | 存量/流量比: {csv_total_stock/gen_total_flow:.1f}x")
            print(f"  验证城市数: {len(city_results)}\n")

            # 6. 打印表头
            print(f"  {'排名':<6} {'城市代码':<8} {'城市名称':<12} {'CSV存量占比':<12} {'模拟流量占比':<12} {'份额偏差':<12} {'存量值':<12} {'流量值':<12} {'状态'}")
            print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*4}")

            # 7. 打印所有城市（按偏差倒序）
            for idx, city in enumerate(city_results, 1):
                print(f"  {idx:<6} {city['code']:<8} {city['name']:<12} {city['csv_share']:>10.2f}% {city['gen_share']:>10.2f}% {city['share_diff']:>+10.2f}% {city['csv_stock']:>10,.0f} {city['gen_flow']:>10,.0f} {city['status']}")

            # 8. 打印统计信息
            max_diff_city = city_results[0]
            avg_diff = sum(c['share_diff_abs'] for c in city_results) / len(city_results)
            cities_with_large_diff = sum(1 for c in city_results if c['share_diff_abs'] >= 5)
            cities_with_ok_diff = sum(1 for c in city_results if c['share_diff_abs'] < 2)

            print(f"\n  统计摘要:")
            print(f"    最大偏差城市: {max_diff_city['name']} ({max_diff_city['code']}) - 偏差 {max_diff_city['share_diff']:+.2f}%")
            print(f"    平均偏差: {avg_diff:.2f}%")
            print(f"    大偏差城市数 (≥5%): {cities_with_large_diff} ({cities_with_large_diff/len(city_results)*100:.1f}%)")
            print(f"    良好城市数 (<2%): {cities_with_ok_diff} ({cities_with_ok_diff/len(city_results)*100:.1f}%)")
            print()

    else:
        print("警告: 未生成任何数据")

    conn.close()

    print(f"\n[完成] 数据库: {db_path}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == '__main__':
    main()
