# -*- coding: utf-8 -*-
"""
人口迁徙数据生成器（动态演化版 2000-2020）
核心改进：
1. 2000年初始化 -> 逐年演化 -> 每年约束校准
2. 解决"超过14亿"问题：Total_Count 代表月度人口存量，验证时限定Month
3. Type动态出现/消失：基于阈值和城市产业结构
"""

import os
import sys
import time
import random
import warnings
import traceback
import duckdb
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from scipy import interpolate

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局配置
# ==============================================================================

OUTPUT_DIR = 'output'
DB_FILENAME = 'local_migration_data.db'
CSV_SAMPLE_FILENAME = 'migration_data_sample_100.csv'
CONSTRAINT_CSV_PATH = r'C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN\2.csv'

# 年份和月份范围
OUTPUT_YEARS = list(range(2000, 2021))  # 2000-2020
OUTPUT_MONTHS = list(range(1, 13))
TOP_N_CITIES = 20

# 维度定义
DIMENSIONS = {
    'D1': {'name': '性别', 'values': ['M', 'F'], 'labels': ['男', '女']},
    'D2': {'name': '生命周期', 'values': ['16-24', '25-34', '35-49', '50-60', '60+'], 'labels': ['试错期', '成家期', '稳固期', '回流期', '养老期']},
    'D3': {'name': '学历', 'values': ['EduLo', 'EduMid', 'EduHi'], 'labels': ['初中及以下', '高中或专科', '本科及以上']},
    'D4': {'name': '行业赛道', 'values': ['Agri', 'Mfg', 'Service', 'Wht'], 'labels': ['农业', '蓝领制造', '传统服务', '现代专业服务']},
    'D5': {'name': '相对收入', 'values': ['IncL', 'IncML', 'IncM', 'IncMH', 'IncH'], 'labels': ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']},
    'D6': {'name': '家庭状态', 'values': ['Split', 'Unit'], 'labels': ['分离', '团聚']},
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

MIN_TYPE_COUNT = 200  # 最小Type样本数阈值
TOTAL_POPULATION_BASE = 1000_0000
POWER_LAW_ALPHA = 1.8

# 迁移模型参数（保持不变）
AGE_MIGRATION_BASE = {'16-24': 0.35, '25-34': 0.25, '35-49': 0.15, '50-60': 0.10, '60+': 0.05}
EDU_MIGRATION_MULTIPLIER = {'EduLo': 0.8, 'EduMid': 1.0, 'EduHi': 1.3}
INDUSTRY_MIGRATION_MULTIPLIER = {'Agri': 0.5, 'Mfg': 1.4, 'Service': 1.2, 'Wht': 0.9}
INCOME_MIGRATION_MULTIPLIER = {'IncL': 1.3, 'IncML': 1.1, 'IncM': 1.0, 'IncMH': 0.9, 'IncH': 0.7}
FAMILY_MIGRATION_MULTIPLIER = {'Split': 1.2, 'Unit': 0.8}
GENDER_MIGRATION_MULTIPLIER = {'M': 1.05, 'F': 0.95}
SEASONAL_FACTORS = {1: 0.7, 2: 0.6, 3: 1.2, 4: 1.15, 5: 1.1, 6: 1.0, 7: 0.95, 8: 0.9, 9: 1.1, 10: 1.05, 11: 1.0, 12: 0.8}
MIGRATION_PROB_MIN = 0.05
MIGRATION_PROB_MAX = 0.50
TEMPERATURE_BASE = 0.3
TOP_N_RATIO_BASE = 0.8

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
    """读取CSV并插值生成2000-2020每年的约束"""
    print(f"读取约束CSV: {csv_path}")

    # 尝试多种编码方式读取CSV
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
        raise ValueError(f"无法读取CSV文件，尝试了编码: {encodings}")

    # 打印原始列名用于调试
    print(f"  原始列名: {df.columns.tolist()}")

    # [修复] 过滤非城市行：只保留以1-6开头的4位数字城市代码
    df = df[df.iloc[:, 0].astype(str).str.match(r'^[1-6]\d{3}$')].copy()

    # 检查列名并重命名
    # 由于CSV的实际列名是：县市名, 年份, 跨市迁入总人口, 真实总人口_合计
    # 需要按照实际列名映射
    possible_mappings = [
        # 第一种可能的列名组合（中文）
        {'县市名': 'city_code', '年份': 'year', '跨市迁入总人口': 'migration_in', '真实总人口_合计': 'total_pop'},
        # 第二种可能的列名组合（字段顺序不同）
        {'县市名': 'city_code', '年份': 'year', '跨市迁徙总人口': 'migration_in', '总人口_合计': 'total_pop'},
        # 如果CSV直接是英文列名
        {'city_code': 'city_code', 'year': 'year', 'migration_in': 'migration_in', 'total_pop': 'total_pop'},
    ]

    renamed = False
    for mapping in possible_mappings:
        if all(col in df.columns for col in mapping.keys()):
            df = df.rename(columns=mapping)
            renamed = True
            print(f"  使用列映射: {mapping}")
            break

    if not renamed:
        # 如果都不匹配，直接按位置赋列名（假设CSV格式固定：4列）
        # CSV格式：县市名, 年份, 跨市迁入总人口, 真实总人口_合计
        if len(df.columns) >= 4:
            df.columns = ['city_code', 'year', 'migration_in', 'total_pop'] + list(df.columns[4:])
            print(f"  按位置强制赋列名: {df.columns[:4].tolist()}")
        else:
            raise ValueError(f"CSV列数不足，至少需要4列，实际: {len(df.columns)}列")

    # 验证必需列是否存在
    required_cols = ['city_code', 'year', 'total_pop', 'migration_in']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV缺少必需列: {missing_cols}，当前列: {df.columns.tolist()}")

    constraints = {}
    total_pop_2000_check = 0

    for city_code in df['city_code'].unique():
        city_data = df[df['city_code'] == city_code].sort_values('year')

        if len(city_data) < 2:
            continue

        years = city_data['year'].values
        total_pops = city_data['total_pop'].values
        migration_ins = city_data['migration_in'].values

        f_total = interpolate.interp1d(years, total_pops, kind='linear', fill_value='extrapolate')
        f_migration = interpolate.interp1d(years, migration_ins, kind='linear', fill_value='extrapolate')

        constraints[city_code] = {}
        for year in range(2000, 2021):
            # 获取推算人口
            p = int(f_total(year))
            m = int(f_migration(year))

            # [修复] 防止线性外推产生极端异常值
            min_pop_in_csv = total_pops.min()
            if p < min_pop_in_csv * 0.1:
                p = int(min_pop_in_csv * 0.8)  # 兜底逻辑

            constraints[city_code][year] = {
                'total_pop': p,
                'migration_in': m
            }

            if year == 2000:
                total_pop_2000_check += p

    print(f"约束城市数: {len(constraints)}")
    print(f"约束数据 2000年总人口汇总(Check): {total_pop_2000_check:,} (如果是21亿，请检查CSV原始数据)")
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
    约束校准：强制对齐CSV的total_pop约束

    逻辑:
        1. 计算当前城市总人口
        2. 与约束目标对比，计算缩放比例
        3. 对所有Type进行缩放
        4. 移除低于阈值的Type（Type消失）
        5. 根据城市产业结构随机添加缺失的Type（Type出现）
    """
    calibrated_state = {}

    for city_code, type_counts in city_state.items():
        # 计算当前总人口
        actual_pop = sum(type_counts.values())

        # 获取约束目标
        if city_code in constraints and year in constraints[city_code]:
            target_pop = constraints[city_code][year]['total_pop']
        else:
            # 无约束城市保持不变
            calibrated_state[city_code] = type_counts.copy()
            continue

        if actual_pop == 0:
            calibrated_state[city_code] = {}
            continue

        # 计算缩放比例
        ratio = target_pop / actual_pop

        # 缩放所有Type
        new_type_counts = {}
        for type_id, count in type_counts.items():
            new_count = int(count * ratio)
            if new_count >= MIN_TYPE_COUNT:
                new_type_counts[type_id] = new_count

        # Type出现逻辑：根据城市产业结构随机添加
        city_industry_type = CITY_INDUSTRY_TYPE.get(city_code, 2)

        # 生成该城市潜在的所有Type
        all_base_types = generate_all_types()
        existing_type_ids = set(new_type_counts.keys())

        for base_type in all_base_types:
            base_type['D7'] = city_code
            potential_type_id = type_to_id(base_type)

            if potential_type_id in existing_type_ids:
                continue

            # 检查该Type是否匹配城市产业结构
            industry = base_type['D4']
            match_score = INDUSTRY_CITY_MATCH.get((city_industry_type, industry), 1.0)

            # 高匹配度的Type有小概率出现
            if match_score > 1.2 and random.random() < 0.05:
                # 给予极小初始值
                initial_count = random.randint(MIN_TYPE_COUNT, MIN_TYPE_COUNT * 3)
                new_type_counts[potential_type_id] = initial_count

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

    def calculate_migration_targets(self, from_city_code, type_dict, migration_prob, row_seed=None):
        """计算迁移目标城市概率"""
        row_rng = np.random.RandomState(row_seed) if row_seed is not None else self.rng

        city_scores = []
        for city_code, city_name in CITIES:
            if city_code == from_city_code:
                continue

            attr = 1.0 if city_code in TIER1_CITIES else (0.7 if city_code in TIER2_CITIES else 0.3)

            city_industry_type = CITY_INDUSTRY_TYPE.get(city_code, 2)
            industry = type_dict['D4']
            attr *= INDUSTRY_CITY_MATCH.get((city_industry_type, industry), 1.0)

            from_gdp = CITY_GDP.get(from_city_code, 1.0)
            to_gdp = CITY_GDP.get(city_code, 1.0)
            if to_gdp > from_gdp * 1.5:
                attr *= 1.15
            elif to_gdp > from_gdp * 1.2:
                attr *= 1.1
            elif to_gdp < from_gdp * 0.7:
                attr *= 0.95

            attr *= (1.0 + row_rng.uniform(-0.1, 0.1))

            city_scores.append((city_code, city_name, attr))

        city_scores.sort(key=lambda x: x[2], reverse=True)
        scores = np.array([s[2] for s in city_scores])

        temp = TEMPERATURE_BASE
        exp_scores = np.exp(scores / temp)
        probs = exp_scores / exp_scores.sum()

        top_n_prob = migration_prob * TOP_N_RATIO_BASE
        top_n_probs = probs[:TOP_N_CITIES]
        top_n_probs = top_n_probs / top_n_probs.sum()
        final_top_probs = top_n_probs * top_n_prob

        result = []
        for i, (c_code, c_name, _) in enumerate(city_scores[:TOP_N_CITIES]):
            result.append((c_code, c_name, final_top_probs[i]))

        other_prob = migration_prob * (1 - TOP_N_RATIO_BASE)
        if other_prob > 0:
            result.append(('Other', '其他', other_prob))

        return result

# ==============================================================================
# 6. 多进程工作函数（并行化：按城市chunk处理）
# ==============================================================================

def process_city_chunk(city_chunk_state, year, type_id_to_dict, city_constraints):
    """
    处理一组城市的数据生成（双重强约束版：人口+迁徙严格对齐）

    核心改进：
    1. 【人口硬锁】：在关键年份(00/10/20)，消除随机噪声，确保总人口严格匹配
    2. 【迁徙对齐】：计算模型预测迁徙量 vs CSV约束迁徙量，生成缩放因子，强制对齐迁徙规模
    3. 【类型安全修复】：强制TopN列为字符串类型，防止DuckDB误判为DOUBLE

    参数:
        city_chunk_state: {city_code: {type_id: count}} 一组城市的人口状态
        year: 当前年份
        type_id_to_dict: {type_id: type_dict}
        city_constraints: {city_code: {year: {'total_pop': int, 'migration_in': int}}}

    返回:
        DataFrame: 生成的数据行（TopN列强制为字符串类型）
    """
    # 关键年份定义（人口硬锁）
    ANCHOR_YEARS = [2000, 2010, 2020]
    is_anchor_year = year in ANCHOR_YEARS

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
    first_city = list(city_chunk_state.keys())[0]
    rng = np.random.RandomState(year + hash(first_city) % 10000)

    city_code_to_name_str = {c[0]: f"{c[1]}({c[0]})" for c in CITIES}
    local_model = MigrationModel(random_seed=year)

    for city_code, type_counts in city_chunk_state.items():
        from_city_str = city_code_to_name_str.get(city_code, city_code)

        # --- 步骤 A: 准备该城市所有Type的数据 ---
        city_types_data = []
        total_pop_base = 0

        # 获取该城市的迁徙目标约束 (从CSV)
        target_migration_volume = 0
        if city_code in city_constraints and year in city_constraints[city_code]:
            target_migration_volume = city_constraints[city_code][year]['migration_in']

        # 第一次遍历：计算总人口基数和初步迁徙意愿
        expected_migration_volume = 0

        for type_id, count in type_counts.items():
            if type_id not in type_id_to_dict:
                continue

            type_dict = type_id_to_dict[type_id]
            base_count = float(count)
            total_pop_base += base_count

            # 计算基础迁移率 (Raw Probability)
            raw_mig_prob = local_model.calculate_base_migration_prob(type_dict, month=None)

            # 累计模型预测的迁徙人数 (基数 * 概率)
            expected_migration_volume += base_count * raw_mig_prob

            city_types_data.append({
                'type_id': type_id,
                'base_count': base_count,
                'type_dict': type_dict,
                'raw_mig_prob': raw_mig_prob
            })

        if not city_types_data:
            continue

        # --- 步骤 B: 计算迁徙缩放因子 (Migration Scaling Factor) ---
        # 如果模型预测要走1万人，但CSV说只走了5千人，那么缩放因子 = 0.5
        mig_scale_factor = 1.0
        if target_migration_volume > 0 and expected_migration_volume > 0:
            mig_scale_factor = target_migration_volume / expected_migration_volume

            # [安全限制] 防止因子过大导致概率超过1.0
            # 如果缩放后概率普遍会溢出，说明Constraints和模型参数严重不符，限制最大倍数
            if mig_scale_factor > 5.0:
                mig_scale_factor = 5.0

            # 【调试信息】在关键年份打印迁徙缩放因子
            if is_anchor_year and expected_migration_volume > 10000:  # 只打印大城市
                print(f"    [迁徙约束] {from_city_str} {year}年: 模型预测={expected_migration_volume:,.0f}, CSV约束={target_migration_volume:,}, 缩放因子={mig_scale_factor:.3f}")

        # --- 步骤 C: 生成最终数据 ---
        for item in city_types_data:
            type_id = item['type_id']
            base_count = item['base_count']
            raw_mig_prob = item['raw_mig_prob']
            type_dict = item['type_dict']

            # 计算调整后的基础迁移概率 (基础概率 * 缩放因子)
            adjusted_base_prob = raw_mig_prob * mig_scale_factor

            # 计算目标城市分布 (Target Distribution)
            # 注意：targets 里的概率之和等于 adjusted_base_prob
            targets = local_model.calculate_migration_targets(city_code, type_dict, adjusted_base_prob)

            target_list = []
            other_prob_base = 0.0
            for t_code, t_name, t_prob in targets:
                if t_code == 'Other':
                    other_prob_base = t_prob
                else:
                    target_list.append((f"{t_name}({t_code})", t_prob))

            # ==============================================================================
            # [修改开始]：按年生成单条记录，注释掉按月生成的逻辑
            # ==============================================================================

            # --- 原代码（已注释） ---
            # 1. 人口计算 (Population Hard-Lock)
            # if is_anchor_year:
            #     # 关键年份：强制去除随机噪声，只保留季节波动
            #     noise = np.ones(12)
            # else:
            #     # 非关键年份：保留 ±2% 的随机波动
            #     noise = 1.0 + rng.uniform(-0.02, 0.02, 12)
            #
            # monthly_counts = (base_count * seasonal_factors_arr * noise).astype(int)
            # monthly_counts = np.maximum(monthly_counts, 0)
            #
            # if monthly_counts.sum() == 0:
            #     continue
            #
            # # 2. 迁徙计算 & 展开12个月
            # for m_idx, count_val in enumerate(monthly_counts):
            #     if count_val == 0:
            #         continue
            #
            #     month = m_idx + 1
            #     season_factor = SEASONAL_FACTORS[month]
            #
            #     # 最终概率 = 调整后的基础概率 * 季节因子
            #     cur_mig_prob = np.clip(adjusted_base_prob * season_factor, MIGRATION_PROB_MIN, 0.95)
            #     stay_prob = 1.0 - cur_mig_prob
            #
            #     # 重新计算Target的缩放比例 (因为cur_mig_prob被截断或季节调整了)
            #     target_scale = cur_mig_prob / adjusted_base_prob if adjusted_base_prob > 1e-6 else 0
            #
            #     cols['Year'].append(year)
            #     cols['Month'].append(month)
            #     cols['Type_ID'].append(type_id)
            #     cols['Region'].append(city_code)
            #     cols['From_City'].append(from_city_str)
            #     cols['Total_Count'].append(count_val)
            #     cols['Stay_Prob'].append(round(stay_prob, 4))
            #
            #     for i in range(TOP_N_CITIES):
            #         col_target = f'To_Top{i+1}'
            #         col_prob = f'To_Top{i+1}_Prob'
            #         if i < len(target_list):
            #             t_str, t_p_base = target_list[i]
            #             cols[col_target].append(str(t_str))
            #             cols[col_prob].append(round(t_p_base * target_scale, 4))
            #         else:
            #             cols[col_target].append('')
            #             cols[col_prob].append(0.0)
            #
            #     cols['To_Other_Prob'].append(round(other_prob_base * target_scale, 4))

            # --- 新代码：全年合并为一条，Month=None ---

            # 使用 base_count 作为全年的总数约束
            count_val = int(base_count)

            if count_val > 0:
                # 不再受季节因子影响，直接使用 adjusted_base_prob
                cur_mig_prob = np.clip(adjusted_base_prob, MIGRATION_PROB_MIN, 0.95)
                stay_prob = 1.0 - cur_mig_prob

                # 因为 targets 的总和本身就等于 adjusted_base_prob，所以缩放比例为 1.0
                target_scale = 1.0

                cols['Year'].append(year)
                cols['Month'].append(None)  # 月份留空 (NULL)
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

            # ==============================================================================
            # [修改结束]
            # ==============================================================================

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
            SELECT typeof({col_name}), COUNT(*) as cnt
            FROM migration_data
            WHERE {col_name} IS NOT NULL AND {col_name} != ''
            GROUP BY typeof({col_name})
        """).df()

        if len(type_check) > 0:
            types_str = ", ".join([f"{row['typeof({col_name})']}({row['cnt']}行)" for _, row in type_check.iterrows()])
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

    # [优化] 设置并行参数
    num_workers = min(16, cpu_count() - 4)  # i7-14700建议20个进程，预留4核给系统
    print(f"   使用 {num_workers} 个进程并行处理每个年份的城市数据\n")

    for year in REMAINING_YEARS:
        print(f"处理年份: {year}")

        # 步骤1：当年初校准（除了2000年）
        if year != 2000:
            current_state = calibrate_city_state(current_state, CITY_CONSTRAINTS, year)

        # 统计当前总人口
        current_total = sum(sum(types.values()) for types in current_state.values())
        print(f"  {year}年初总人口: {current_total:,}")

        # 步骤2：[并行化] 将城市列表切分为多个chunk
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
                # 【修改】传入 CITY_CONSTRAINTS 参数用于迁徙约束
                futures.append(executor.submit(process_city_chunk, chunk_state, year, type_id_to_dict, CITY_CONSTRAINTS))

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

        # 步骤3：统计年度完成情况
        if year_row_count > 0:
            total_rows += year_row_count
            print(f"  ✓ {year}年完成，共写入 {year_row_count:,} 行\n")

        # 步骤4：年底再次校准（确保下一年初的状态符合约束）
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

        # 7. 验证约束（修复：Total_Count是存量，直接对比，乘以季节因子）
        print("\n6. 验证约束符合度（查询12月存量）...")
        for city_code in list(CITY_CONSTRAINTS.keys())[:5]:
            for year in [2000, 2010, 2020]:
                result = conn.execute(f"""
                    SELECT SUM(Total_Count) as total
                    FROM migration_data
                    WHERE Region = '{city_code}' AND Year = {year} AND Month = 12
                """).fetchone()

                generated = result[0] if result[0] else 0
                target = CITY_CONSTRAINTS[city_code][year]['total_pop']

                # [修复] 目标就是 target，不需要除以12
                # 因为季节因子12月是0.8，所以生成的可能会比target小20%
                expected_dec = target * SEASONAL_FACTORS.get(12, 1.0)
                diff_pct = abs(generated - expected_dec) / expected_dec * 100 if expected_dec > 0 else 0

                print(f"   {city_code} {year}年12月: 目标(含季节)≈{expected_dec:,.0f}, 生成={generated:,}, 偏差={diff_pct:.1f}%")

        # 验证全国总人口（12月）
        print("\n7. 验证全国总人口（12月）...")
        for year in [2000, 2010, 2020]:
            result = conn.execute(f"""
                SELECT SUM(Total_Count) as total
                FROM migration_data
                WHERE Year = {year} AND Month = 12
            """).fetchone()

            generated = result[0] if result[0] else 0

            # 计算全国总人口（所有约束城市的总和）
            national_target = sum(
                CITY_CONSTRAINTS[cc][year]['total_pop']
                for cc in CITY_CONSTRAINTS
                if year in CITY_CONSTRAINTS[cc]
            )
            # [修复] 乘以12月的季节因子
            expected_dec = national_target * SEASONAL_FACTORS.get(12, 1.0)

            diff_pct = abs(generated - expected_dec) / expected_dec * 100 if expected_dec > 0 else 0
            print(f"   全国{year}年12月: 目标(含季节)≈{expected_dec:,.0f}, 生成={generated:,}, 偏差={diff_pct:.1f}%")

        # 【新增】验证迁徙总量（双重约束验证）
        print("\n8. 验证迁徙总量（年度迁徙约束）...")
        for city_code in list(CITY_CONSTRAINTS.keys())[:5]:
            for year in [2000, 2010, 2020]:
                if year not in CITY_CONSTRAINTS[city_code]:
                    continue

                # 计算该城市当年的迁徙总量（所有Type的迁徙人数总和）
                # 迁徙人数 = Total_Count * (1 - Stay_Prob)
                result = conn.execute(f"""
                    SELECT SUM(Total_Count * (1 - Stay_Prob)) as total_migration
                    FROM migration_data
                    WHERE Region = '{city_code}' AND Year = {year}
                """).fetchone()

                generated_migration = result[0] if result[0] else 0
                target_migration = CITY_CONSTRAINTS[city_code][year]['migration_in']

                if target_migration > 0:
                    diff_pct = abs(generated_migration - target_migration) / target_migration * 100
                    print(f"   {city_code} {year}年: 目标迁徙={target_migration:,}, 生成={generated_migration:,.0f}, 偏差={diff_pct:.1f}%")

    else:
        print("警告: 未生成任何数据")

    conn.close()

    print(f"\n[完成] 数据库: {db_path}")
    print(f"总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == '__main__':
    main()
