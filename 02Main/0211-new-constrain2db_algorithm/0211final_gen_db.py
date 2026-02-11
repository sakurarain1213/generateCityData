import pandas as pd
import numpy as np
import re
import itertools
import json
import os
import sys
import duckdb
import multiprocessing as mp
import warnings
import contextlib  # <--- 新增：用于静音
import time        # <--- 新增：用于重试
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm

# ==========================================
# 0. 导入外部核心算法
# ==========================================
# 确保 from_single2multiple_dimension.py 在同一目录下
try:
    from from_single2multiple_dimension import CityPopulationSynthesizer
except ImportError:
    print("错误: 未找到 from_single2multiple_dimension.py，请确保该文件在当前目录下。")
    sys.exit(1)

# ==========================================
# 1. 全局配置与路径
# ==========================================

GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)

# 路径配置
BASE_DIR = r"C:\Users\w1625\Desktop\CityDBGenerate"
INPUT_EXCEL = os.path.join(BASE_DIR, r"0211constrain2\constrain2.xlsx")
IPF_DATA_DIR = os.path.join(BASE_DIR, r"0211处理后城市数据")
CITY_JSONL_PATH = "city.jsonl"
DB_FILE = "output.db"
TABLE_NAME = "migration_data"

# 维度定义 (保持与 external script 一致以便映射)
DIMENSIONS = {
    'D1': {'name': '性别', 'values': ['M', 'F']},
    'D2': {'name': '生命周期', 'values': ['20', '30', '40', '55', '65']},
    'D3': {'name': '学历', 'values': ['EduLo', 'EduMid', 'EduHi']},
    'D4': {'name': '行业赛道', 'values': ['Agri', 'Mfg', 'Service', 'Wht']},
    'D5': {'name': '相对收入', 'values': ['IncL', 'IncML', 'IncM', 'IncMH', 'IncH']},
    'D6': {'name': '家庭状态', 'values': ['Split', 'Unit']}
}

# 目标输出 Schema
FINAL_SCHEMA = [
    'Year', 'Month', 'Type_ID', 'Birth_Region', 'From_City',
    'Total_Count', 'Stay_Prob', 'Outflow_Count'
]
for i in range(1, 21):
    FINAL_SCHEMA.extend([f'To_Top{i}', f'To_Top{i}_Count'])

# ==========================================
# 2. 基础工具与城市映射
# ==========================================

CITY_ID_MAPPING = {
    '1101': '1100', '1102': '1100', # 北京
    '1201': '1200', '1202': '1200', # 天津
    '3101': '3100', '3102': '3100', # 上海
    '5001': '5000', '5002': '5000', # 重庆
}

CITY_NAME_MAP = {}
# 坐标数据
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
def load_city_metadata():
    """加载城市名称和坐标数据"""
    # 1. 加载名称
    if os.path.exists(CITY_JSONL_PATH):
        with open(CITY_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    CITY_NAME_MAP[data['city_id']] = data['name']
                except:
                    pass
    # 2. 坐标数据已经在全局变量 CITY_COORDS 中定义，无需额外操作
    pass

def get_city_display(city_id):
    """获取城市显示名称 Name(ID)"""
    # 先尝试直接获取
    name = CITY_NAME_MAP.get(str(city_id))
    # 如果没找到，尝试映射后的ID
    if not name:
        mapped_id = CITY_ID_MAPPING.get(str(city_id))
        if mapped_id:
            name = CITY_NAME_MAP.get(mapped_id)
    
    if name:
        return f"{name}({city_id})"
    return f"未知{city_id}({city_id})"

def deterministic_choice(indices, probs, seed):
    rng = np.random.RandomState(seed)
    return rng.choice(indices, p=probs)

def create_empty_result_row():
    row = {
        'Year': 0, 'Month': 12, 'Type_ID': '', 'Birth_Region': '', 'From_City': '',
        'Total_Count': 0, 'Stay_Prob': 0.0, 'Outflow_Count': 0
    }
    for i in range(1, 21):
        row[f'To_Top{i}'] = 'None'
        row[f'To_Top{i}_Count'] = 0
    return row

# ==========================================
# 3. 业务逻辑类
# ==========================================

class CityProfiler:
    """城市画像：提供基础属性判断"""
    @staticmethod
    def get_city_tier(city_code):
        if city_code in ['1100', '1200', '3100', '5000', '4401', '4403']: return 1
        elif city_code[:2] in ['32', '33', '44', '37']: return 2
        elif city_code.endswith('01'): return 3
        else: return 4

class MigrationEngine:
    """迁移引擎：计算流出与去向 (保留原逻辑)"""
    
    @staticmethod
    def calculate_outflow_prob(segment, city_code, year):
        """计算流出倾向"""
        # segment 是 tuple: (Sex, Age, Edu, Ind, Inc, Fam)
        sex, age, edu, ind, inc, fam = segment
        tier = CityProfiler.get_city_tier(city_code)
        
        prob = 0.05 # 基础
        
        # 经济推力
        if tier >= 3 and inc in ['IncL', 'IncML']: prob += 0.20
        # 行业推力
        if ind == 'Agri' and age in ['20', '30']: prob += 0.25
        # 家庭推力
        if fam == 'Split': prob += 0.15
        # 年龄修正
        if age == '20': prob *= 2.0
        if age == '65': prob *= 0.1
        # 一线挤出
        if tier == 1 and inc in ['IncL', 'IncML'] and fam == 'Unit': prob += 0.1
        
        return min(0.95, prob)

    @staticmethod
    def get_destination_affinity(segment, origin_code, dest_code):
        """计算去向适配度"""
        sex, age, edu, ind, inc, fam = segment
        
        origin_coord = CITY_COORDS.get(origin_code, (0,0))
        dest_coord = CITY_COORDS.get(dest_code, (0,0))
        dest_tier = CityProfiler.get_city_tier(dest_code)
        
        score = 1.0
        
        # 距离衰减
        dist = np.sqrt((origin_coord[0]-dest_coord[0])**2 + (origin_coord[1]-dest_coord[1])**2)
        if dist > 0: score /= (dist + 0.5)
        
        # 梯度迁移
        if edu == 'EduHi' or ind == 'Wht':
            if dest_tier <= 2: score *= 3.0
            
        # 制造业导向
        if ind == 'Mfg':
            if dest_code.startswith('44') or dest_code.startswith('32'): score *= 2.5
            
        # 返乡逻辑
        if dest_code.startswith('51') or dest_code.startswith('41'):
            if age in ['40', '55'] and fam == 'Split': score *= 2.0
            
        return score

# ==========================================
# 4. 多进程 Worker 逻辑
# ==========================================

# 全局变量，用于在子进程中缓存 IPF 模型
generator = None

def init_worker(data_path, seed):
    """Worker 初始化函数"""
    global generator
    warnings.filterwarnings('ignore')
    try:
        # 在初始化时也静音，防止初始化日志刷屏
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            generator = CityPopulationSynthesizer(data_dir=data_path, seed=seed)
    except Exception as e:
        # 初始化错误还是要打印出来
        sys.__stdout__.write(f"[Worker {os.getpid()}] Init Error: {e}\n")

def parse_type_string(type_str):
    """
    将 IPF 输出的 type 字符串解析为 tuple
    Input: "M_20_EduLo_Agri_IncL_Split_5000"
    Output: ('M', '20', 'EduLo', 'Agri', 'IncL', 'Split')
    """
    parts = type_str.split('_')
    # 假设最后一部分是 city_code，前面 6 部分是维度
    if len(parts) >= 7:
        return tuple(parts[:6])
    return None

def process_row_task(args):
    """处理单行 Excel 数据 - 已添加静音处理"""
    idx, row_dict = args
    global generator

    # === 关键修改：使用 contextlib 屏蔽所有 print 输出 ===
    # 这样外部算法产生的 [DEBUG] 日志就不会破坏进度条了
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            year = int(row_dict['Year'])

            # 解析出发城市
            # 支持 "重庆市(5000)" 或 纯数字 格式
            from_city_raw = str(row_dict['From_City'])
            match = re.search(r'\((\d{4})\)', from_city_raw)
            if match:
                raw_id = match.group(1)
            else:
                # 尝试直接提取数字
                raw_id = ''.join(filter(str.isdigit, from_city_raw))

            from_city_id = CITY_ID_MAPPING.get(raw_id, raw_id)
            if not from_city_id:
                return idx, [], "Invalid City ID"

            total_pop = int(row_dict['Total_Count'])
            # 处理可能为空的 Outflow
            outflow_target = row_dict.get('Outflow_Count', 0)
            if pd.isna(outflow_target): outflow_target = 0
            outflow_target = int(outflow_target)

            # 解析 Top 去向目标 (列名 To_Top1, To_Top1_Count ...)
            top_destinations = {}
            for i in range(1, 21):
                dest_name_col = f'To_Top{i}'
                dest_count_col = f'To_Top{i}_Count'

                if dest_name_col not in row_dict or dest_count_col not in row_dict:
                    continue

                dest_str = str(row_dict[dest_name_col])
                count_val = row_dict[dest_count_col]

                if pd.isna(dest_str) or dest_str in ['nan', 'None', '']: continue
                if pd.isna(count_val): count_val = 0

                # 提取 ID
                d_match = re.search(r'\((\d{4})\)', dest_str)
                if d_match:
                    d_id = d_match.group(1)
                else:
                    d_id = ''.join(filter(str.isdigit, dest_str))

                d_id = CITY_ID_MAPPING.get(d_id, d_id)

                if d_id and d_id in CITY_COORDS:
                    top_destinations[d_id] = top_destinations.get(d_id, 0) + int(count_val)

            # --------------------------------------------------------
            # 核心步骤 1: 调用 IPF 算法生成 1200 种微观分布
            # --------------------------------------------------------
            if generator is None:
                 return idx, [], "Generator not initialized"

            # 调用 from_single2multiple_dimension 的算法 (静音环境下执行)
            df_ipf = generator.generate_single(from_city_id, year, debug=False)

            if df_ipf.empty:
                return idx, [], f"IPF Generation failed for {from_city_id} {year}"

            # --------------------------------------------------------
            # 核心步骤 2: 将 IPF 概率转换为人口数
            # --------------------------------------------------------
            segment_keys = []
            segment_pops = []
            base_probs = df_ipf['Probability'].values

            # 将总人口分配给各 Segment
            current_pops = np.floor(base_probs * total_pop).astype(int)
            # 补齐余数
            diff = total_pop - current_pops.sum()
            if diff > 0:
                # 确定性补齐
                sort_indices = np.argsort(base_probs)[::-1] # 降序
                for k in range(diff):
                    current_pops[sort_indices[k % len(sort_indices)]] += 1

            # 解析 Type 字符串为 Tuple
            type_strings = df_ipf['Type'].values
            for t_str in type_strings:
                seg = parse_type_string(t_str)
                if seg:
                    segment_keys.append(seg)
                else:
                    # Fallback (should not happen)
                    segment_keys.append(('M','20','EduLo','Agri','IncL','Split'))

            segment_pops = current_pops

            # --------------------------------------------------------
            # 核心步骤 3: 计算流出与分配 (Migration Engine)
            # --------------------------------------------------------

            # 3.1 计算流出
            outflow_probs_raw = []
            for seg in segment_keys:
                p = MigrationEngine.calculate_outflow_prob(seg, from_city_id, year)
                outflow_probs_raw.append(p)
            outflow_probs_raw = np.array(outflow_probs_raw)

            # 缩放以匹配总流出目标
            expected_outflow = np.sum(segment_pops * outflow_probs_raw)
            scale_factor = outflow_target / expected_outflow if expected_outflow > 0 else 0
            final_outflow_probs = np.clip(outflow_probs_raw * scale_factor, 0, 1.0)

            segment_outflows = np.floor(segment_pops * final_outflow_probs).astype(int)
            # 补齐流出余数
            out_diff = outflow_target - segment_outflows.sum()
            if out_diff > 0:
                # 优先补给流出概率高的
                top_indices = np.argsort(final_outflow_probs)[-100:]
                for k in range(out_diff):
                    seed = int(from_city_id) * 1000 + k
                    idx_choice = deterministic_choice(top_indices, None, seed)
                    if segment_pops[idx_choice] > segment_outflows[idx_choice]:
                        segment_outflows[idx_choice] += 1

            # 3.2 分配去向
            top_dest_ids = list(top_destinations.keys())
            affinity_matrix = np.zeros((len(segment_keys), len(top_dest_ids)))

            for i, seg in enumerate(segment_keys):
                for j, d_id in enumerate(top_dest_ids):
                    affinity_matrix[i, j] = MigrationEngine.get_destination_affinity(seg, from_city_id, d_id)

            segment_remaining_outflow = segment_outflows.copy()
            segment_dest_map = [{} for _ in range(len(segment_keys))]

            # 优先满足 Top 20 目标
            for j, d_id in enumerate(top_dest_ids):
                target_count = top_destinations[d_id]
                # 权重 = Affinity * 剩余流出量
                weights = affinity_matrix[:, j] * segment_remaining_outflow
                total_w = np.sum(weights)

                if total_w == 0: continue

                allocations = np.floor(weights / total_w * target_count).astype(int)
                # 补齐
                alloc_diff = target_count - allocations.sum()
                if alloc_diff > 0:
                    valid_indices = np.where(segment_remaining_outflow > allocations)[0]
                    if len(valid_indices) > 0:
                        w_sub = weights[valid_indices]
                        if w_sub.sum() > 0:
                            p_indices = w_sub / w_sub.sum()
                            for k in range(alloc_diff):
                                seed = int(d_id) * 100 + k
                                chosen = deterministic_choice(valid_indices, p_indices, seed)
                                allocations[chosen] += 1

                # 更新
                for i in range(len(segment_keys)):
                    if allocations[i] > 0:
                        actual = min(allocations[i], segment_remaining_outflow[i])
                        segment_dest_map[i][d_id] = actual
                        segment_remaining_outflow[i] -= actual

            # 长尾分配
            other_cities = [k for k in CITY_COORDS.keys() if k not in top_dest_ids and k != from_city_id]
            for i in range(len(segment_keys)):
                rem = segment_remaining_outflow[i]
                if rem > 0:
                    if other_cities:
                        seed = int(from_city_id) * 2000 + i
                        rand_dest = deterministic_choice(other_cities, None, seed)
                        segment_dest_map[i][rand_dest] = segment_dest_map[i].get(rand_dest, 0) + rem
                    else:
                        segment_dest_map[i]['Unknown'] = rem

            # --------------------------------------------------------
            # 4. 组装结果 (已修复排序和噪声丢失问题)
            # --------------------------------------------------------
            output_rows = []

            # 【修复点 1 & 2】
            # 不再优先遍历 top20_ids，而是完全基于当前 Segment 的实际去向数据进行排序
            # 这样既保证了 To_Top1 > To_Top2 (降序)，也保证了长尾城市(Noise)能进入列表

            for i, seg in enumerate(segment_keys):
                row = create_empty_result_row()

                # Type ID: M_20_..._5000
                type_id = f"{'_'.join(seg)}_{from_city_id}"

                row['Year'] = year
                row['Month'] = 12
                row['Type_ID'] = type_id
                row['Birth_Region'] = from_city_id
                row['From_City'] = get_city_display(from_city_id)
                row['Total_Count'] = int(segment_pops[i])
                row['Stay_Prob'] = round(1.0 - final_outflow_probs[i], 6)
                row['Outflow_Count'] = int(segment_outflows[i])

                dest_map = segment_dest_map[i]

                # 1. 核心修复：对所有去向（含固定Top20和随机长尾）统一按人数降序排序
                sorted_all_dests = sorted(dest_map.items(), key=lambda x: x[1], reverse=True)

                # 2. 依次填入前 20 个坑位
                # 如果实际去向不足 20 个，剩下的保持 None/0
                for k in range(20):
                    col_name = f'To_Top{k+1}'
                    col_cnt_name = f'To_Top{k+1}_Count'

                    if k < len(sorted_all_dests):
                        cid, cnt = sorted_all_dests[k]
                        # 只有当 count > 0 时才填入，避免填入大量 0 的记录（可视情况调整）
                        if cnt > 0:
                            row[col_name] = get_city_display(cid)
                            row[col_cnt_name] = int(cnt)
                        else:
                            # 理论上 sorted 后不会先出现 0，除非总数就是 0
                            row[col_name] = 'None'
                            row[col_cnt_name] = 0
                    else:
                        # 坑位没填满，保持默认
                        row[col_name] = 'None'
                        row[col_cnt_name] = 0

                output_rows.append(row)

            return idx, output_rows, None

        except Exception as e:
            # 这里的 print 不会被看到（因为在 devnull 块里），所以返回错误信息给主进程打印
            import traceback
            return idx, [], f"Exception: {str(e)}\n{traceback.format_exc()}"

# ==========================================
# 5. DB 写手进程
# ==========================================

def writer_worker(queue, db_path, table_name):
    """写数据库进程 - 显式定义Schema，彻底解决类型推断错误"""
    con = None
    retry_count = 0
    max_retries = 10

    # -------------------------------------------------------
    # 1. 连接重试机制
    # -------------------------------------------------------
    while retry_count < max_retries:
        try:
            con = duckdb.connect(db_path)
            print(f"[Writer] Successfully connected to DB.")
            break
        except Exception as e:
            retry_count += 1
            print(f"[Writer] Connect attempt {retry_count} failed: {e}. Retrying in 2s...")
            time.sleep(2)

    if con is None:
        print("[Writer] FATAL: Could not connect to database after retries.")
        return

    # -------------------------------------------------------
    # 2. 显式建表 (关键修改步骤)
    # -------------------------------------------------------
    try:
        # 根据 FINAL_SCHEMA 动态生成 SQL 类型定义
        # 规则如下：
        # 1. Stay_Prob -> DOUBLE (浮点数)
        # 2. Year, Month, 以及所有以 Count 结尾的列 -> BIGINT (整数)
        # 3. 其他所有列 (Type_ID, Birth_Region, From_City, To_TopX) -> VARCHAR (字符串)

        col_defs = []
        for col in FINAL_SCHEMA:
            if col == 'Stay_Prob':
                sql_type = 'DOUBLE'
            elif col in ['Year', 'Month'] or col.endswith('Count') or col == 'Total_Count':
                sql_type = 'BIGINT'  # 使用 BIGINT 防止人口数过大溢出
            else:
                # 这一步确保 Type_ID 被强制定义为字符串
                sql_type = 'VARCHAR'

            col_defs.append(f"{col} {sql_type}")

        # 拼接 SQL 语句: CREATE TABLE IF NOT EXISTS table (Year BIGINT, Month BIGINT, Type_ID VARCHAR, ...)
        create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(col_defs)})"

        # 执行建表
        con.execute(create_sql)
        # print(f"[Writer] Table schema initialized with explicit types.")

    except Exception as e:
        sys.__stdout__.write(f"[Writer] Create Table Error: {e}\n")
        return

    # -------------------------------------------------------
    # 3. 写入循环
    # -------------------------------------------------------
    while True:
        payload = queue.get()
        if payload is None: break

        try:
            df_batch = pd.DataFrame(payload)

            # --- 数据清洗与类型强制转换 ---

            # 1. 补全缺失列
            for col in FINAL_SCHEMA:
                if col not in df_batch.columns:
                    if col == 'Stay_Prob':
                        df_batch[col] = 0.0
                    elif col in ['Year', 'Month'] or col.endswith('Count') or col == 'Total_Count':
                        df_batch[col] = 0
                    else:
                        df_batch[col] = 'None'

            # 2. 强制对齐列顺序
            df_batch = df_batch[FINAL_SCHEMA]

            # 3. *** 关键：显式将字符串列转换为 str 类型 ***
            # 这一步防止 Pandas 里的对象类型混杂，确保传给 DuckDB 的全是纯字符串
            str_cols = [c for c in FINAL_SCHEMA if c != 'Stay_Prob' and c not in ['Year', 'Month'] and not c.endswith('Count') and c != 'Total_Count']

            for c in str_cols:
                # 转换为字符串，并将 'nan' 或 'None' 统一清洗
                df_batch[c] = df_batch[c].astype(str)

            # 4. 显式将数值列转换为数字
            num_cols = [c for c in FINAL_SCHEMA if c not in str_cols]
            for c in num_cols:
                df_batch[c] = pd.to_numeric(df_batch[c], errors='coerce').fillna(0)

            # --- 执行插入 ---
            con.register("df_batch", df_batch)
            con.execute(f"INSERT INTO {table_name} SELECT * FROM df_batch")
            con.unregister("df_batch")

        except Exception as e:
            # 错误信息打印到标准输出
            sys.__stdout__.write(f"[Writer] Insert Error: {e}\n")
            # 打印第一行数据方便调试
            if not df_batch.empty:
                 sys.__stdout__.write(f"[Writer] Problematic Row Sample: {df_batch.iloc[0].to_dict()}\n")

    con.close()
    print("[Writer] Finished and closed.")

# ==========================================
# 6. 后处理函数 (修复 Unknown)
# ==========================================

def post_process_fix_names(db_path):
    print("\n" + "=" * 60)
    print("执行后处理：修复未知城市名称")
    print("=" * 60)
    
    if not os.path.exists(db_path): return
    
    con = duckdb.connect(db_path)
    try:
        # 获取所有文本列
        cols = ['From_City'] + [f'To_Top{i}' for i in range(1, 21)]
        
        count = 0
        for city_id, name in tqdm(CITY_NAME_MAP.items(), desc="修复进度"):
            target_str = f"{name}({city_id})"
            # 查找 '未知{city_id}({city_id})'
            unknown_pattern = f"%未知%{city_id}%"
            
            for col in cols:
                query = f"UPDATE {TABLE_NAME} SET {col} = '{target_str}' WHERE {col} LIKE '{unknown_pattern}'"
                con.execute(query)
                
        print("后处理完成。")
    except Exception as e:
        print(f"后处理出错: {e}")
    finally:
        con.close()

# ==========================================
# 7. 主程序入口
# ==========================================

if __name__ == "__main__":
    mp.freeze_support()
    
    # 1. 加载基础数据
    print("正在加载城市元数据...")
    load_city_metadata()
    if not CITY_COORDS:
        # 尝试加载 City Coords (如果在脚本中未定义)
        # 这里为了演示，再次提醒
        pass
        
    # 2. 读取 Excel 约束文件
    if not os.path.exists(INPUT_EXCEL):
        print(f"错误: 找不到输入文件 {INPUT_EXCEL}")
        sys.exit(1)
        
    print(f"正在读取约束文件: {INPUT_EXCEL}")
    df_constraints = pd.read_excel(INPUT_EXCEL)
    
    # 3. 准备任务
    tasks = []
    processed_keys = set()

    # === 关键修改：检查断点后立即关闭连接，防止锁死 ===
    print("检查断点...")
    try:
        con = duckdb.connect(DB_FILE)
        # 检查表是否存在
        tables = con.execute("SHOW TABLES").fetchall()
        table_exists = any(t[0] == TABLE_NAME for t in tables)

        if table_exists:
            res = con.execute(f"SELECT DISTINCT Year, Birth_Region FROM {TABLE_NAME}").fetchall()
            for r in res:
                processed_keys.add((int(r[0]), str(r[1])))
            print(f"发现断点: 已处理 {len(processed_keys)} 个组合")
        else:
            print("未发现现有表，准备全量运行。")
    except Exception as e:
        print(f"读取断点时发生非致命错误: {e}")
    finally:
        # !!! 必须关闭，否则 Writer 进程打不开文件 !!!
        try:
            con.close()
        except:
            pass
        print("断点检查完毕，数据库连接已释放。")

    print("生成任务列表...")
    skipped = 0
    for idx, row in df_constraints.iterrows():
        try:
            y = int(row['Year'])
            fc_str = str(row['From_City'])
            match = re.search(r'\((\d{4})\)', fc_str)
            raw_id = match.group(1) if match else ''.join(filter(str.isdigit, fc_str))
            city_id = CITY_ID_MAPPING.get(raw_id, raw_id)
            
            if (y, city_id) in processed_keys:
                skipped += 1
                continue
                
            tasks.append((idx, row.to_dict()))
        except Exception as e:
            print(f"Skipping row {idx} due to parse error: {e}")

    print(f"总任务: {len(df_constraints)}, 跳过: {skipped}, 待处理: {len(tasks)}")
    
    if not tasks:
        print("无任务需要执行。")
        sys.exit(0)

    # 4. 并行执行
    # 这里的 initargs 传入 IPF 数据路径，确保子进程能找到数据
    worker_count = min(len(tasks), max(1, mp.cpu_count() - 2)) # 留点余地
    print(f"启动 {worker_count} 个 Worker 进程...")
    
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=worker_count * 2)
    writer = ctx.Process(target=writer_worker, args=(queue, DB_FILE, TABLE_NAME))
    writer.start()
    
    try:
        # 使用 tqdm 监控进度
        # 因为 Worker 里的 print 被屏蔽了，现在的 tqdm 应该会非常稳定
        with ctx.Pool(processes=worker_count, initializer=init_worker, initargs=(IPF_DATA_DIR, GLOBAL_RANDOM_SEED)) as pool:
            # ncols=100 让进度条定宽，防止跳动
            with tqdm(total=len(tasks), desc="合成进度", unit="task", ncols=100) as pbar:
                for idx, rows, err in pool.imap_unordered(process_row_task, tasks):
                    if rows:
                        queue.put(rows)
                    if err:
                        # 只有真的报错时，使用 tqdm.write 打印，不会破坏进度条
                        pbar.write(f"Row {idx} Error: {err}")
                    pbar.update(1)
    finally:
        queue.put(None)
        writer.join()
        
    print("合成完成。")
    
    # 5. 执行后处理
    post_process_fix_names(DB_FILE)
    
    print("所有流程结束。")