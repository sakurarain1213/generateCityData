# -*- coding: utf-8 -*-
"""
人口迁徙数据生成器（全量版 2000-2020）- 修改版

功能：
1. 读取 2.csv (2000-2020年)，对缺失年份的迁入人口数据进行线性插值。
2. 内置全量城市坐标，用于引力模型计算。
3. 生成 DuckDB 数据库并导出采样 CSV。

# 直接验证.db 不再合成.db
cd 0125MAIN; python generate_db222.py --verify
"""

import os
import sys
import time
import warnings
import duckdb
import numpy as np
import pandas as pd
import json
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局配置与路径
# ==============================================================================

# 基础路径 (请确保 2.csv 位于此目录下)
BASE_DIR = r"C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN"
CSV_PATH_MAIN = os.path.join(BASE_DIR, '2.csv')
CITY_NODES_PATH = os.path.join(BASE_DIR, 'city_nodes.jsonl')

# 输出配置
OUTPUT_DIR = BASE_DIR  # 直接输出到同一目录
DB_FILENAME = 'local_migration_data_full.db'
SAMPLE_CSV_NAME = 'sampled_migration_data.csv'

# 生成范围
START_YEAR = 2000
END_YEAR = 2020
TARGET_MONTH = 12
TOP_N_CITIES = 20
MIN_TYPE_COUNT = 10

# 维度定义
DIMENSIONS = {
    'D1': {'name': '性别', 'values': ['M', 'F']},
    'D2': {'name': '生命周期', 'values': ['16-24', '25-34', '35-49', '50-60', '60+']},
    'D3': {'name': '学历', 'values': ['EduLo', 'EduMid', 'EduHi']},
    'D4': {'name': '行业赛道', 'values': ['Agri', 'Mfg', 'Service', 'Wht']},
    'D5': {'name': '相对收入', 'values': ['IncL', 'IncML', 'IncM', 'IncMH', 'IncH']},
    'D6': {'name': '家庭状态', 'values': ['Split', 'Unit']},
}

# ==============================================================================
# 2. 城市地理坐标库 (Knowledge Base)
# ==============================================================================
# 保持原有的坐标数据不变
CITY_COORDS = {
    '1301': (114.51, 38.04), '1302': (118.18, 39.63), '1303': (119.60, 39.93), '1304': (114.54, 36.62),
    '1305': (114.50, 37.07), '1306': (115.46, 38.87), '1307': (114.88, 40.81), '1308': (117.96, 40.97),
    '1309': (116.83, 38.30), '1310': (116.68, 39.54), '1311': (115.66, 37.73), '1401': (112.54, 37.87),
    '1402': (113.29, 40.09), '1403': (113.58, 37.85), '1404': (113.11, 36.19), '1405': (112.85, 35.49),
    '1406': (112.43, 39.33), '1407': (112.75, 37.68), '1408': (111.00, 35.02), '1409': (112.73, 38.41),
    '1410': (111.51, 36.08), '1411': (111.13, 37.52), '1501': (111.69, 40.81), '1502': (109.84, 40.65),
    '1503': (106.82, 39.67), '1504': (118.96, 42.26), '1505': (122.26, 43.61), '1506': (109.99, 39.81),
    '1507': (119.76, 49.21), '1508': (107.41, 40.75), '1509': (113.13, 41.01), '1522': (122.07, 46.07),
    '1525': (116.09, 43.95), '1529': (105.70, 38.84), '2101': (123.43, 41.80), '2102': (121.61, 38.91),
    '2103': (122.99, 41.10), '2104': (123.95, 41.88), '2105': (123.76, 41.29), '2106': (124.39, 40.12),
    '2107': (121.12, 41.10), '2108': (122.22, 40.66), '2109': (121.67, 42.02), '2110': (123.17, 41.27),
    '2111': (122.07, 41.11), '2112': (123.85, 42.29), '2113': (120.45, 41.57), '2114': (120.83, 40.71),
    '2201': (125.32, 43.81), '2202': (126.54, 43.83), '2203': (124.38, 43.16), '2204': (125.13, 42.90),
    '2205': (125.93, 41.72), '2206': (126.42, 41.93), '2207': (124.82, 45.14), '2208': (122.83, 45.61),
    '2224': (129.51, 42.90), '2301': (126.66, 45.75), '2302': (123.96, 47.34), '2303': (130.97, 45.29),
    '2304': (130.29, 47.34), '2305': (131.15, 46.64), '2306': (125.10, 46.59), '2307': (128.89, 47.72),
    '2308': (130.36, 46.80), '2309': (131.00, 45.77), '2310': (129.63, 44.59), '2311': (127.52, 50.24),
    '2312': (126.96, 46.64), '2327': (124.71, 52.33), '3201': (118.79, 32.06), '3202': (120.31, 31.49),
    '3203': (117.28, 34.20), '3204': (119.97, 31.81), '3205': (120.58, 31.29), '3206': (120.89, 31.98),
    '3207': (119.22, 34.59), '3208': (119.01, 33.61), '3209': (120.16, 33.34), '3210': (119.41, 32.39),
    '3211': (119.45, 32.20), '3212': (119.92, 32.45), '3213': (118.27, 33.96), '3301': (120.15, 30.27),
    '3302': (121.55, 29.87), '3303': (120.69, 27.99), '3304': (120.75, 30.74), '3305': (120.08, 30.89),
    '3306': (120.58, 30.03), '3307': (119.64, 29.07), '3308': (118.87, 28.93), '3309': (122.20, 29.98),
    '3310': (121.42, 28.65), '3311': (119.92, 28.46), '3401': (117.22, 31.82), '3402': (118.37, 31.32),
    '3403': (117.39, 32.91), '3404': (116.99, 32.62), '3405': (118.50, 31.67), '3406': (116.79, 33.95),
    '3407': (117.81, 30.93), '3408': (117.06, 30.54), '3410': (118.33, 29.71), '3411': (118.31, 32.30),
    '3412': (115.81, 32.89), '3413': (116.98, 33.65), '3415': (116.50, 31.75), '3416': (115.77, 33.86),
    '3417': (117.49, 30.66), '3418': (118.75, 30.94), '3501': (119.29, 26.07), '3502': (118.08, 24.47),
    '3503': (119.00, 25.45), '3504': (117.63, 26.26), '3505': (118.67, 24.87), '3506': (117.64, 24.51),
    '3507': (118.17, 26.64), '3508': (117.02, 25.07), '3509': (119.54, 26.66), '3601': (115.85, 28.68),
    '3602': (117.17, 29.27), '3603': (113.85, 27.62), '3604': (115.99, 29.71), '3605': (114.91, 27.81),
    '3606': (117.03, 28.24), '3607': (114.93, 25.83), '3608': (114.99, 27.11), '3609': (114.41, 27.81),
    '3610': (116.35, 27.94), '3611': (117.94, 28.45), '3701': (117.12, 36.65), '3702': (120.38, 36.06),
    '3703': (118.05, 36.81), '3704': (117.32, 34.81), '3705': (118.67, 37.43), '3706': (121.44, 37.46),
    '3707': (119.16, 36.62), '3708': (116.58, 35.41), '3709': (117.08, 36.20), '3710': (122.12, 37.51),
    '3711': (119.52, 35.41), '3713': (118.35, 35.05), '3714': (116.35, 37.43), '3715': (115.98, 36.45),
    '3716': (117.97, 37.38), '3717': (115.48, 35.23), '4101': (113.62, 34.74), '4102': (114.30, 34.79),
    '4103': (112.45, 34.61), '4104': (113.19, 33.76), '4105': (114.39, 36.09), '4106': (114.29, 35.74),
    '4107': (113.92, 35.30), '4108': (113.24, 35.21), '4109': (115.02, 35.76), '4110': (113.85, 34.03),
    '4111': (114.01, 33.58), '4112': (111.20, 34.77), '4113': (112.52, 32.99), '4114': (115.65, 34.41),
    '4115': (114.09, 32.14), '4116': (114.69, 33.62), '4117': (114.02, 32.98), '4201': (114.30, 30.59),
    '4202': (115.03, 30.19), '4203': (110.79, 32.64), '4205': (111.28, 30.69), '4206': (112.12, 32.00),
    '4207': (114.89, 30.39), '4208': (112.20, 31.03), '4209': (113.95, 30.90), '4210': (112.23, 30.33),
    '4211': (114.87, 30.45), '4212': (114.32, 29.84), '4213': (113.38, 31.69), '4228': (109.48, 30.27),
    '4301': (112.93, 28.22), '4302': (113.13, 27.82), '4303': (112.94, 27.82), '4304': (112.57, 26.89),
    '4305': (111.46, 27.23), '4306': (113.12, 29.35), '4307': (111.69, 29.03), '4308': (110.47, 29.11),
    '4309': (112.35, 28.55), '4310': (113.01, 25.77), '4311': (111.61, 26.42), '4312': (109.99, 27.55),
    '4313': (112.00, 27.70), '4331': (109.72, 28.31), '4401': (113.26, 23.12), '4402': (113.59, 24.81),
    '4403': (114.05, 22.54), '4404': (113.57, 22.27), '4405': (116.68, 23.35), '4406': (113.12, 23.02),
    '4407': (113.08, 22.57), '4408': (110.35, 21.27), '4409': (110.92, 21.66), '4412': (112.46, 23.04),
    '4413': (114.41, 23.11), '4414': (116.12, 24.28), '4415': (115.37, 22.78), '4416': (114.70, 23.74),
    '4417': (111.98, 21.85), '4418': (113.03, 23.70), '4419': (113.75, 23.02), '4420': (113.39, 22.51),
    '4451': (116.62, 23.65), '4452': (116.37, 23.55), '4453': (112.04, 22.91), '4501': (108.36, 22.81),
    '4502': (109.42, 24.32), '4503': (110.18, 25.23), '4504': (111.27, 23.47), '4505': (109.11, 21.49),
    '4506': (108.35, 21.69), '4507': (108.65, 21.98), '4508': (109.60, 23.11), '4509': (110.16, 22.63),
    '4510': (106.61, 23.90), '4511': (111.56, 24.40), '4512': (107.69, 24.69), '4513': (109.22, 23.76),
    '4514': (107.35, 22.41), '4601': (110.19, 20.04), '4602': (109.51, 18.25), '4603': (112.33, 16.83),
    '4604': (109.57, 19.52), '5101': (104.06, 30.57), '5103': (104.77, 29.33), '5104': (101.71, 26.58),
    '5105': (105.44, 28.87), '5106': (104.39, 31.12), '5107': (104.73, 31.46), '5108': (105.84, 32.43),
    '5109': (105.59, 30.51), '5110': (105.05, 29.58), '5111': (103.76, 29.55), '5113': (106.11, 30.83),
    '5114': (103.84, 30.07), '5115': (104.64, 28.75), '5116': (106.63, 30.45), '5117': (107.50, 31.20),
    '5118': (103.04, 29.98), '5119': (106.75, 31.86), '5120': (104.62, 30.12), '5132': (102.22, 31.89),
    '5133': (101.96, 30.04), '5134': (102.26, 27.88), '5201': (106.63, 26.64), '5202': (104.83, 26.59),
    '5203': (106.92, 27.72), '5204': (105.94, 26.25), '5205': (105.29, 27.29), '5206': (109.18, 27.71),
    '5223': (104.89, 25.08), '5226': (107.98, 26.56), '5227': (107.51, 26.25), '5301': (102.83, 24.88),
    '5303': (103.79, 25.49), '5304': (102.54, 24.35), '5305': (99.16, 25.11), '5306': (103.71, 27.33),
    '5307': (100.22, 26.85), '5308': (100.97, 22.77), '5309': (100.08, 23.88), '5323': (101.52, 25.03),
    '5325': (103.38, 23.36), '5326': (104.21, 23.39), '5328': (100.79, 22.00), '5329': (100.26, 25.59),
    '5331': (98.57, 24.43), '5333': (98.85, 25.85), '5334': (99.70, 27.82), '5401': (91.14, 29.64),
    '5402': (88.88, 29.26), '5403': (97.17, 31.14), '5404': (94.36, 29.64), '5405': (91.77, 29.23),
    '5406': (92.05, 31.47), '5425': (80.10, 32.50), '6101': (108.93, 34.26), '6102': (109.08, 35.08),
    '6103': (107.23, 34.36), '6104': (108.70, 34.33), '6105': (109.51, 34.49), '6106': (109.48, 36.58),
    '6107': (107.02, 33.06), '6108': (109.73, 38.28), '6109': (109.02, 32.68), '6110': (109.94, 33.86),
    '6201': (103.83, 36.06), '6202': (98.28, 39.77), '6203': (102.18, 38.52), '6204': (104.13, 36.54),
    '6205': (105.71, 34.58), '6206': (102.63, 37.92), '6207': (100.44, 38.92), '6208': (106.66, 35.54),
    '6209': (98.49, 39.73), '6210': (107.64, 35.71), '6211': (104.62, 35.58), '6212': (104.92, 33.39),
    '6229': (103.21, 35.60), '6230': (102.91, 34.98), '6301': (101.77, 36.61), '6302': (102.10, 36.50),
    '6322': (100.90, 36.95), '6323': (102.01, 35.51), '6325': (100.62, 36.29), '6326': (100.24, 34.47),
    '6327': (97.00, 33.00), '6328': (97.37, 37.37), '6401': (106.23, 38.48), '6402': (106.38, 39.01),
    '6403': (106.19, 37.99), '6404': (106.28, 36.00), '6405': (105.18, 37.51), '6501': (87.61, 43.79),
    '6502': (84.88, 45.57), '6504': (89.17, 42.94), '6505': (93.51, 42.81), '6523': (87.30, 44.01),
    '6527': (82.06, 44.90), '6528': (86.11, 41.76), '6529': (80.26, 41.16), '6530': (76.16, 39.71),
    '6531': (75.99, 39.47), '6532': (79.92, 37.11), '6540': (81.32, 43.91), '6542': (82.98, 46.74),
    '6543': (88.14, 47.84), '1100': (116.40, 39.90), '1200': (117.20, 39.08), '3100': (121.47, 31.23),
    '5000': (106.55, 29.56)
}

# 城市经济基础分 (用于引力模型 GDP替代)
# 简单分层：直辖市10，省会8，其他4
CITY_TIERS = {}
for code, coords in CITY_COORDS.items():
    if code in ['1100', '1200', '3100', '5000']:
        CITY_TIERS[code] = 10.0
    elif code.endswith('01'): # 省会通常01结尾
        CITY_TIERS[code] = 8.0
    else:
        CITY_TIERS[code] = 5.0

# 引力模型参数
GRAVITY_PARAMS = {
    'dist_decay': 1.5,
    'gdp_pull': 2.0,
    'province_barrier': 0.2,
    'same_dialect': 1.5
}

# ==============================================================================
# 3. 全量人群模板生成
# ==============================================================================

def get_all_type_templates():
    """预生成1200种人群组合及其基础权重"""
    import itertools
    keys = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    vals = [DIMENSIONS[k]['values'] for k in keys]

    all_combinations = list(itertools.product(*vals))  # 2*5*3*4*5*2 = 1200

    # 简单的权重分配逻辑（基于人口学常识）
    type_weights = []
    for combo in all_combinations:
        w = 1.0
        # 年龄分布调整
        if combo[1] == '16-24': w *= 1.2  # 年轻人较多
        if combo[1] == '60+': w *= 0.15   # 老年人较少
        # 学历分布调整
        if combo[2] == 'EduHi': w *= 0.1  # 高学历稀缺
        if combo[2] == 'EduMid': w *= 1.5 # 中等学历较多
        # 收入分布调整
        if combo[4] == 'IncH': w *= 0.05  # 高收入稀缺
        if combo[4] == 'IncM': w *= 2.0   # 中等收入较多
        # 行业分布调整
        if combo[3] == 'Service': w *= 1.5  # 服务业占比高
        if combo[3] == 'Agri': w *= 0.8     # 农业占比降低

        type_weights.append(w)

    # 归一化权重
    total_w = sum(type_weights)
    norm_weights = [f / total_w for f in type_weights]

    return all_combinations, norm_weights

# 在全局先生成好模板
ALL_COMBOS, NORM_WEIGHTS = get_all_type_templates()

# ==============================================================================
# 4. 工具函数
# ==============================================================================

def load_city_names():
    """加载城市ID到名称的映射"""
    city_map = {}
    if not os.path.exists(CITY_NODES_PATH):
        print(f"Warning: {CITY_NODES_PATH} 不存在，将使用默认格式")
        return city_map

    with open(CITY_NODES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                city_map[data['city_id']] = data['name']
    return city_map

def haversine(lon1, lat1, lon2, lat2):
    """计算两点间地理距离(km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

def calculate_gravity_score(origin_code, dest_code):
    """计算引力分数"""
    if origin_code == dest_code:
        return 0.0
    
    o_coord = CITY_COORDS.get(origin_code, (110.0, 35.0))
    d_coord = CITY_COORDS.get(dest_code, (115.0, 35.0))
    o_gdp = CITY_TIERS.get(origin_code, 4.0)
    d_gdp = CITY_TIERS.get(dest_code, 4.0)

    # 1. 距离因子
    dist = haversine(o_coord[0], o_coord[1], d_coord[0], d_coord[1])
    dist = max(dist, 50.0)
    factor_dist = 1.0 / (dist ** GRAVITY_PARAMS['dist_decay'])

    # 2. 经济差距 (GDP/Tier diff)
    gdp_diff = d_gdp - o_gdp
    if gdp_diff > 0:
        factor_gdp = (1 + gdp_diff) ** GRAVITY_PARAMS['gdp_pull']
    else:
        factor_gdp = 0.5 

    # 3. 省界因子
    is_same_prov = (origin_code[:2] == dest_code[:2])
    factor_barrier = GRAVITY_PARAMS['same_dialect'] if is_same_prov else GRAVITY_PARAMS['province_barrier']

    return factor_dist * factor_gdp * factor_barrier

# ==============================================================================
# 4. 数据读取与预处理 (修改版)
# ==============================================================================

def process_city_group(group):
    """
    处理单个城市的时间序列数据：
    1. 重新索引确保 2000-2020 年份完整。
    2. 对 'inflow' 和 'resident' 进行线性插值。
    """
    # 设置年份为索引
    group = group.sort_values('year').set_index('year')
    
    # 重新索引到完整年份范围
    full_range = pd.Index(range(START_YEAR, END_YEAR + 1), name='year')
    group = group.reindex(full_range)
    
    # 线性插值，limit_direction='both' 确保开头和结尾如果缺失也能被填充（如果有部分数据）
    # 如果某列全是 NaN，插值后仍为 NaN
    group['inflow'] = group['inflow'].interpolate(method='linear', limit_direction='both')
    group['resident'] = group['resident'].interpolate(method='linear', limit_direction='both')
    
    return group.reset_index()

def load_combined_constraints():
    """
    读取 2.csv 并处理，返回全局约束字典
    Structure: constraints[year][city_code] = {'resident': int, 'inflow': int}
    """
    global_constraints = {}
    
    print(f"1. 读取并处理全量约束: {CSV_PATH_MAIN}")
    
    if not os.path.exists(CSV_PATH_MAIN):
        print(" [Error] 2.csv 不存在，请检查路径。")
        sys.exit(1)
        
    try:
        # 读取 CSV
        df = pd.read_csv(CSV_PATH_MAIN, engine='python')
        df.columns = df.columns.str.strip()
        
        # 字段映射
        req_cols = {
            '年份': 'year',
            '城市代码': 'code',
            '常住人口数(人)': 'resident',
            '人口普查跨市【迁入】总人口': 'inflow'
        }
        
        # 检查列是否存在
        missing_cols = [c for c in req_cols.keys() if c not in df.columns]
        if missing_cols:
            print(f" [Error] CSV 缺少必要列: {missing_cols}")
            sys.exit(1)
            
        # 提取并改名
        df = df[list(req_cols.keys())].rename(columns=req_cols)
        
        # 数据清洗
        # 确保 code 是字符串且 clean
        df['code'] = df['code'].astype(str).str.split('.').str[0].str.strip()
        df = df[df['code'].str.len() == 4] # 简单过滤非城市行
        
        # 确保 year 是数值
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year', 'code'])
        df['year'] = df['year'].astype(int)
        
        # 过滤年份范围 (稍微放宽一点以便插值，但最后只取范围内)
        df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
        
        # 确保数值列是数字
        df['resident'] = pd.to_numeric(df['resident'], errors='coerce')
        df['inflow'] = pd.to_numeric(df['inflow'], errors='coerce')

        print(f"   -> 原始数据行数: {len(df)}")
        print("   -> 正在进行线性插值处理 (By City)...")

        # 按城市分组进行插值
        # 只有当城市至少有一条有效数据时才有意义
        valid_cities = df.groupby('code').filter(lambda x: x['resident'].notnull().any() or x['inflow'].notnull().any())
        
        if valid_cities.empty:
            print(" [Error] 有效数据为空。")
            sys.exit(1)

        # 应用插值逻辑
        df_interpolated = valid_cities.groupby('code').apply(process_city_group).reset_index(drop=True)
        
        # 填充插值后仍为 NaN 的值 (例如某城市完全没有迁入数据)，默认补 0
        df_interpolated['resident'] = df_interpolated['resident'].fillna(0)
        df_interpolated['inflow'] = df_interpolated['inflow'].fillna(0)
        
        # 再次过滤 code (groupby apply 可能会产生空 code 行，视 pandas 版本而定，保险起见)
        df_interpolated['code'] = df_interpolated['code'].fillna(method='ffill') # 理论上不需要
        
        # 转为字典结构
        count_records = 0
        for _, row in tqdm(df_interpolated.iterrows(), total=len(df_interpolated), desc="   -> 构建字典"):
            y = int(row['year'])
            code = str(row['code'])
            
            if y < START_YEAR or y > END_YEAR: continue
            
            if y not in global_constraints:
                global_constraints[y] = {}
            
            global_constraints[y][code] = {
                'resident': int(round(row['resident'])),
                'inflow': int(round(row['inflow']))
            }
            count_records += 1
            
        print(f"   -> 处理完成，共生成 {count_records} 条年度城市记录。")

    except Exception as e:
        print(f" [Error] 数据处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    return global_constraints

# ==============================================================================
# 5. 核心处理逻辑 (按年)
# ==============================================================================

def solve_gravity_flow(year, constraints):
    """
    根据当年的约束，解算引力矩阵
    返回: dict[origin_city] -> {total_natives, stay_prob, destinations:[(dest, prob)]}
    """
    valid_cities = list(constraints.keys())
    # 确保只处理有坐标的城市，或者为其分配默认值
    valid_cities = [c for c in valid_cities if constraints[c]['resident'] > 0]
    
    if not valid_cities: return {}

    # 1. 计算所有流向 Dest 的权重分布
    # temp_outflows[origin][dest] = count
    temp_outflows = {c: {} for c in valid_cities}
    
    # 只计算有迁入需求的城市
    targets = [c for c in valid_cities if constraints[c]['inflow'] > 0]
    
    # 这里的逻辑是：对于每个有迁入需求的城市，看看谁会被吸过来
    for dest in targets:
        target_in = constraints[dest]['inflow']
        
        weights = []
        origins = []
        sum_w = 0
        
        for origin in valid_cities:
            if origin == dest: continue
            
            # 引力计算
            w = calculate_gravity_score(origin, dest)
            
            # 乘以来源地人口基数 (近似发射能力)
            # 因为此时还没算出流出量，用常住人口近似权重
            w *= constraints[origin]['resident'] 
            
            weights.append(w)
            origins.append(origin)
            sum_w += w
            
        if sum_w == 0: continue
        
        # 分配流量
        for o, w in zip(origins, weights):
            flow = target_in * (w / sum_w)
            if flow > 0.5:
                temp_outflows[o][dest] = flow

    # 2. 汇总 Origin 视角的流出，计算 Native Stay
    final_flows = {}

    for code in valid_cities:
        resident = constraints[code]['resident']
        inflow_target = constraints[code]['inflow'] # 该城市接收的外来人口

        # 核心恒等式: 常住 = 本地留存 + 外来迁入
        # 本地留存 = 常住 - 外来迁入
        native_stay = max(0, resident - inflow_target)

        # 获取该城市流出的去向
        dest_dict = temp_outflows[code]

        # === 归一化逻辑开始 ===

        # 【关键修复】使用CSV约束的常住人口作为总人口基数
        # 这样可以确保每个城市的Total_Count总和严格等于CSV约束
        total_natives = resident

        if total_natives < 1: continue

        # 1. 先排序，取前 N 个
        sorted_dests = sorted(dest_dict.items(), key=lambda x: x[1], reverse=True)[:TOP_N_CITIES]

        # 2. 计算这 Top N 个城市的流量总和
        top_n_outflow_sum = sum([flow for _, flow in sorted_dests])

        # 3. 计算留存概率和流出概率
        # 留存概率 = 本地留存人数 / 总人口
        stay_prob = native_stay / total_natives if total_natives > 0 else 1.0

        # 4. 生成去向概率列表
        # 需要确保：stay_prob + sum(dest_probs) = 1.0
        # 并且：sum(dest_probs) * total_natives = native_stay 以内的流出人数

        dests_probs = []

        if top_n_outflow_sum > 0 and native_stay > 0:
            # 计算可用于流出的概率空间
            available_outflow_prob = 1.0 - stay_prob

            # 按比例分配流出概率
            for d_c, d_f in sorted_dests:
                # 每个目的地的概率 = (该目的地流量 / 总流出量) * 可用流出概率
                dest_prob = (d_f / top_n_outflow_sum) * available_outflow_prob
                dests_probs.append((d_c, dest_prob))
        else:
            # 没有流出，所有概率都是0
            for d_c, _ in sorted_dests:
                dests_probs.append((d_c, 0.0))

        final_flows[code] = {
            'total_natives': int(total_natives),  # 使用CSV约束的常住人口
            'stay_prob': stay_prob,
            'destinations': dests_probs
        }

        # === 归一化逻辑结束 ===

    return final_flows

def generate_types(city_code, total_count):
    """
    为每个城市分配全量1200种人群
    确保 total_count = sum(types.count)
    """
    types = []
    age_map = {'16-24': '20', '25-34': '30', '35-49': '40', '50-60': '55', '60+': '65'}

    assigned_total = 0
    for i, combo in enumerate(ALL_COMBOS):
        # 按比例计算该类人群数量
        count = int(total_count * NORM_WEIGHTS[i])

        # 修正：即使比例很低，也保证至少有1个人（除非总人口实在太少）
        if count == 0 and total_count > 5000:
            count = 1

        if count > 0:
            # 组装 TypeID
            attrs = list(combo)
            type_id_parts = [attrs[0], age_map.get(attrs[1], attrs[1])] + attrs[2:] + [city_code]
            type_id = "_".join(type_id_parts)

            types.append((type_id, count))
            assigned_total += count

    # 尾差处理：确保总数绝对相等
    diff = total_count - assigned_total
    if diff != 0 and len(types) > 0:
        # 将差值补在人数最多的人群上
        max_idx = 0
        max_count = types[0][1]
        for idx, (_, cnt) in enumerate(types):
            if cnt > max_count:
                max_count = cnt
                max_idx = idx
        types[max_idx] = (types[max_idx][0], types[max_idx][1] + diff)

    return types

# ==============================================================================
# 6. 独立验证函数
# ==============================================================================

def verify_top20_distribution(db_path=None):
    """
    独立验证函数：检查 Top20 去向总和是否等于 Outflow_Count
    可以在数据库生成后单独运行

    参数:
        db_path: 数据库路径，如果为None则使用默认路径
    """
    if db_path is None:
        db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)

    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return

    print("=" * 80)
    print("Top20 去向分布验证")
    print("=" * 80)
    print(f"数据库路径: {db_path}\n")

    conn = duckdb.connect(db_path, read_only=True)

    try:
        # 查询所有数据
        print("正在加载数据...")
        query = f"""
            SELECT
                Year, Month, Type_ID, Birth_Region, From_City,
                Total_Count, Stay_Prob, Outflow_Count,
                {', '.join([f'To_Top{i}_Count' for i in range(1, TOP_N_CITIES + 1)])}
            FROM migration_data
        """
        df = conn.execute(query).df()

        print(f"总记录数: {len(df):,}\n")

        # 计算 Top20 去向总和
        top_cols = [f'To_Top{i}_Count' for i in range(1, TOP_N_CITIES + 1)]
        df['Top20_Sum'] = df[top_cols].sum(axis=1)

        # 计算差异
        df['Diff'] = df['Top20_Sum'] - df['Outflow_Count']
        df['Diff_Abs'] = df['Diff'].abs()
        df['Match'] = (df['Diff'] == 0)

        # 统计匹配情况
        total_rows = len(df)
        matched_rows = df['Match'].sum()
        match_rate = (matched_rows / total_rows * 100) if total_rows > 0 else 0

        print("=" * 80)
        print("总体统计")
        print("=" * 80)
        print(f"总记录数: {total_rows:,}")
        print(f"完全匹配记录数: {matched_rows:,}")
        print(f"匹配率: {match_rate:.2f}%")
        print(f"不匹配记录数: {total_rows - matched_rows:,}")

        # 如果有不匹配的记录
        if matched_rows < total_rows:
            print("\n" + "=" * 80)
            print("不匹配样本分析（按 Outflow_Count 和 Top20_Sum 绝对值倒序）")
            print("=" * 80)

            # 筛选不匹配的记录
            mismatch_df = df[~df['Match']].copy()

            # 按 Outflow_Count 和 Top20_Sum 的绝对值倒序排序
            mismatch_df['Total_Flow'] = mismatch_df['Outflow_Count'] + mismatch_df['Top20_Sum']
            mismatch_df = mismatch_df.sort_values('Total_Flow', ascending=False)

            # 打印前20条
            print(f"\n前20条不匹配记录:\n")

            print(f"{'序号':<6} {'年份':<6} {'城市':<10} {'Type_ID':<45} "
                  f"{'总人数':<12} {'流出数':<12} {'Top20和':<12} {'差异':<10}")
            print("-" * 130)

            for idx, (_, row) in enumerate(mismatch_df.head(20).iterrows(), 1):
                # 截断Type_ID以适应显示
                type_id = row['Type_ID'][:43] if len(row['Type_ID']) > 43 else row['Type_ID']
                print(f"{idx:<6} {row['Year']:<6} {row['Birth_Region']:<10} {type_id:<45} "
                      f"{row['Total_Count']:<12,} {row['Outflow_Count']:<12,} "
                      f"{row['Top20_Sum']:<12,} {row['Diff']:<+10,}")

            # 按年份统计
            print("\n" + "=" * 80)
            print("按年份统计不匹配情况")
            print("=" * 80)

            print(f"\n{'年份':<8} {'不匹配数':<12} {'平均差异':<12} {'最大差异':<12} "
                  f"{'总流出':<15} {'Top20总和':<15}")
            print("-" * 90)

            for year in sorted(mismatch_df['Year'].unique()):
                year_data = mismatch_df[mismatch_df['Year'] == year]
                count = len(year_data)
                mean_diff = year_data['Diff_Abs'].mean()
                max_diff = year_data['Diff_Abs'].max()
                total_outflow = year_data['Outflow_Count'].sum()
                total_top20 = year_data['Top20_Sum'].sum()

                print(f"{year:<8} {count:<12,} {mean_diff:<12,.0f} {max_diff:<12,.0f} "
                      f"{total_outflow:<15,} {total_top20:<15,}")

            # 按Type_ID统计（显示不匹配最多的前10个Type）
            print("\n" + "=" * 80)
            print("不匹配最多的前10个Type_ID")
            print("=" * 80)

            type_stats = mismatch_df.groupby('Type_ID').agg({
                'Diff_Abs': ['count', 'mean', 'sum']
            }).round(2)
            type_stats.columns = ['不匹配次数', '平均差异', '总差异']
            type_stats = type_stats.sort_values('不匹配次数', ascending=False).head(10)

            print(f"\n{'Type_ID':<50} {'不匹配次数':<15} {'平均差异':<15} {'总差异':<15}")
            print("-" * 95)

            for type_id, row in type_stats.iterrows():
                # 截断Type_ID
                type_id_display = type_id[:48] if len(type_id) > 48 else type_id
                print(f"{type_id_display:<50} {int(row['不匹配次数']):<15,} "
                      f"{row['平均差异']:<15,.0f} {row['总差异']:<15,.0f}")

            # 按城市统计（只显示不匹配最多的前10个城市）
            print("\n" + "=" * 80)
            print("不匹配最多的前10个城市")
            print("=" * 80)

            city_stats = mismatch_df.groupby('Birth_Region').agg({
                'Diff_Abs': ['count', 'mean', 'sum']
            }).round(2)
            city_stats.columns = ['不匹配次数', '平均差异', '总差异']
            city_stats = city_stats.sort_values('不匹配次数', ascending=False).head(10)

            print(f"\n{'城市代码':<10} {'不匹配次数':<15} {'平均差异':<15} {'总差异':<15}")
            print("-" * 60)

            for city_code, row in city_stats.iterrows():
                print(f"{city_code:<10} {int(row['不匹配次数']):<15,} "
                      f"{row['平均差异']:<15,.0f} {row['总差异']:<15,.0f}")

        else:
            print("\n✓ 所有记录的 Top20 去向总和都等于 Outflow_Count！")

        print("\n" + "=" * 80)
        print("验证完成")
        print("=" * 80)

    except Exception as e:
        print(f"验证过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

# ==============================================================================
# 7. 多进程工作函数（按城市分块并行）
# ==============================================================================

def process_city_chunk_for_year(city_codes_chunk, year, year_constraints, city_names_map):
    """
    处理一组城市在某个年份的数据生成（按城市分块并行）

    参数:
        city_codes_chunk: 城市代码列表（这个进程负责的城市）
        year: 年份
        year_constraints: 该年份的约束数据
        city_names_map: 城市名称映射

    返回:
        DataFrame: 该chunk的数据
    """
    batch_data = []

    # 1. 解算引力 (复用原有函数)
    flow_model = solve_gravity_flow(year, year_constraints)

    if not flow_model:
        return pd.DataFrame()

    # 2. 只处理分配给这个进程的城市
    for city_code in city_codes_chunk:
        if city_code not in flow_model:
            continue

        info = flow_model[city_code]

        # Type 生成
        type_list = generate_types(city_code, info['total_natives'])

        # 格式化 From_City
        city_name = city_names_map.get(city_code, f"City_{city_code}")
        from_city_str = f"{city_name}({city_code})"

        for type_id, count in type_list:
            outflow_count = int(count * (1 - info['stay_prob']))

            row = {
                'Year': year,
                'Month': TARGET_MONTH,
                'Type_ID': type_id,
                'Birth_Region': city_code,
                'From_City': from_city_str,
                'Total_Count': count,
                'Stay_Prob': round(info['stay_prob'], 6),
                'Outflow_Count': outflow_count
            }

            dests = info['destinations']
            for i in range(TOP_N_CITIES):
                k_city = f'To_Top{i+1}'
                k_count = f'To_Top{i+1}_Count'
                if i < len(dests):
                    d_c, d_p = dests[i]
                    dest_name = city_names_map.get(d_c, f"City_{d_c}")
                    row[k_city] = f"{dest_name}({d_c})"
                    row[k_count] = int(count * d_p)
                else:
                    row[k_city] = ""
                    row[k_count] = 0

            batch_data.append(row)

    # 转为 DataFrame 返回
    return pd.DataFrame(batch_data)

# ==============================================================================
# 7. 主程序 (多进程版)
# ==============================================================================

def main():
    print("=== 初始化 (高效多进程版) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)

    # 重置数据库
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    # 建表
    cols = [
        'Year INTEGER', 'Month INTEGER', 'Type_ID VARCHAR', 'Birth_Region VARCHAR',
        'From_City VARCHAR', 'Total_Count INTEGER',
        'Stay_Prob DOUBLE', 'Outflow_Count INTEGER'
    ]
    for i in range(1, TOP_N_CITIES + 1):
        cols.append(f'To_Top{i} VARCHAR')
        cols.append(f'To_Top{i}_Count INTEGER')
    conn.execute(f"CREATE TABLE migration_data ({', '.join(cols)})")

    # 1. 加载所有约束 (主进程加载一次)
    all_constraints = load_combined_constraints()
    city_names = load_city_names()  # 加载映射
    print(f"已加载 {len(city_names)} 个城市名称映射")

    total_written = 0
    years_to_process = [y for y in range(START_YEAR, END_YEAR + 1) if y in all_constraints]

    print(f"准备处理年份: {years_to_process}")

    # === 并行策略：按城市分块，而不是按年份 ===
    # 参考 generate_db.py 的高效策略
    import multiprocessing
    num_workers = min(28, multiprocessing.cpu_count() - 2)  # 激进配置，预留2核给系统
    print(f"使用 {num_workers} 个并发进程（按城市分块并行）\n")

    # 逐年处理
    for year in years_to_process:
        print(f"处理年份: {year}")

        if year not in all_constraints:
            print(f"  跳过 {year}: 无约束数据\n")
            continue

        year_constraints = all_constraints[year]

        # 获取该年所有有效城市
        valid_cities = [c for c in year_constraints.keys() if year_constraints[c]['resident'] > 0]

        if not valid_cities:
            print(f"  跳过 {year}: 无有效城市\n")
            continue

        print(f"  有效城市数: {len(valid_cities)}")

        # 将城市列表切分为多个chunk（关键优化！）
        city_chunks = np.array_split(valid_cities, num_workers)

        year_row_count = 0

        # 使用进程池并行处理各城市chunk
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = []
            for chunk in city_chunks:
                if len(chunk) > 0:
                    futures.append(executor.submit(
                        process_city_chunk_for_year,
                        chunk.tolist(), year, year_constraints, city_names
                    ))

            # 添加实时进度条
            with tqdm(total=len(futures), desc=f"  {year}年并行计算", unit="chunk", leave=False) as pbar:
                for future in as_completed(futures):
                    try:
                        # 直接获取DataFrame
                        df_chunk = future.result()

                        if not df_chunk.empty:
                            # 直接写入DuckDB，避免内存堆积
                            conn.register('temp_chunk', df_chunk)
                            conn.execute("INSERT INTO migration_data SELECT * FROM temp_chunk")
                            conn.unregister('temp_chunk')

                            year_row_count += len(df_chunk)

                            # 释放内存
                            del df_chunk
                    except Exception as e:
                        print(f"\n  [错误] Chunk处理失败: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        pbar.update(1)

        if year_row_count > 0:
            total_written += year_row_count
            print(f"  ✓ {year}年完成，共写入 {year_row_count:,} 行\n")
        else:
            print(f"  ✗ {year}年未生成数据\n")

    print(f"\n所有年份处理完成！共写入 {total_written:,} 行数据")

    # === 后续处理：排序 ===
    print("\n=== 数据排序 (按 Year, Birth_Region, Type_ID) ===")
    # 创建临时表，按照要求排序
    conn.execute("""
        CREATE TABLE migration_data_sorted AS
        SELECT * FROM migration_data
        ORDER BY Year ASC, Birth_Region ASC, Type_ID ASC
    """)
    # 删除原表，重命名排序后的表
    conn.execute("DROP TABLE migration_data")
    conn.execute("ALTER TABLE migration_data_sorted RENAME TO migration_data")

    # === 后续处理：索引 ===
    print("\n=== 收尾工作 ===")
    conn.execute("CREATE INDEX idx_yr_reg ON migration_data(Year, Birth_Region)")
    print(f"总写入行数: {total_written:,}")

    # 采样导出代码保持不变
    print("正在导出采样数据...")
    sample_sql = "SELECT * FROM migration_data USING SAMPLE 20"
    try:
        df_sample = conn.execute(sample_sql).df()

        print("\n=== Sample Output (Top 5 cols) ===")
        print(df_sample[['Year', 'Type_ID', 'From_City', 'Total_Count', 'Stay_Prob', 'To_Top1']].to_string())

        csv_out = os.path.join(OUTPUT_DIR, SAMPLE_CSV_NAME)
        df_sample.to_csv(csv_out, index=False, encoding='utf-8-sig')
        print(f"\n采样文件已保存: {csv_out}")

        # === 验证逻辑：检查 Top20 去向总和是否等于 Outflow_Count ===
        print("\n=== 验证 Top20 去向总和 vs Outflow_Count ===")

        # 计算每行的 Top20 去向总和
        top_cols = [f'To_Top{i}_Count' for i in range(1, TOP_N_CITIES + 1)]
        df_sample['Top20_Sum'] = df_sample[top_cols].sum(axis=1)

        # 比较 Top20_Sum 和 Outflow_Count
        df_sample['Match'] = (df_sample['Top20_Sum'] == df_sample['Outflow_Count'])

        # 计算差异和差异百分比
        df_sample['Diff'] = df_sample['Top20_Sum'] - df_sample['Outflow_Count']
        df_sample['Diff_Pct'] = df_sample.apply(
            lambda row: abs(row['Diff']) / row['Outflow_Count'] * 100 if row['Outflow_Count'] > 0 else 0,
            axis=1
        )

        # 统计匹配情况
        total_rows = len(df_sample)
        matched_rows = df_sample['Match'].sum()
        match_rate = (matched_rows / total_rows * 100) if total_rows > 0 else 0

        print(f"采样数据总行数: {total_rows}")
        print(f"Top20总和 == Outflow_Count 的行数: {matched_rows}")
        print(f"匹配率: {match_rate:.2f}%")

        # 显示不匹配的样本（如果有），按差异百分比降序排列
        if matched_rows < total_rows:
            print("\n不匹配的样本示例（按差异百分比降序，前10行）:")
            mismatch_df = df_sample[~df_sample['Match']].copy()
            mismatch_df = mismatch_df.sort_values('Diff_Pct', ascending=False)
            display_cols = ['Year', 'From_City', 'Outflow_Count', 'Top20_Sum', 'Diff', 'Diff_Pct']
            print(mismatch_df[display_cols].head(10).to_string(index=False))
        else:
            print("\n✓ 所有采样数据的 Top20 去向总和都等于 Outflow_Count！")

    except Exception as e:
        print(f"采样失败: {e}")

    conn.close()
    print(f"数据库已保存: {db_path}")

    # === 可选：运行独立验证 ===
    print("\n" + "=" * 80)
    run_verification = input("是否运行详细的 Top20 分布验证？(y/n，默认n): ").strip().lower()
    if run_verification == 'y':
        print()
        verify_top20_distribution(db_path)

if __name__ == '__main__':
    import sys

    # 支持命令行参数：直接运行验证
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        # 直接运行验证函数
        db_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(OUTPUT_DIR, DB_FILENAME)
        verify_top20_distribution(db_path)
    else:
        # 正常运行主程序
        main()