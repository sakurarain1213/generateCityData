import pandas as pd
import numpy as np
import re
import itertools
import json
import os
import duckdb  # <--- 新增
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ==========================================
# 0. 全局随机种子设置 (确保可复现性)
# ==========================================

GLOBAL_RANDOM_SEED = 42
np.random.seed(GLOBAL_RANDOM_SEED)

# ==========================================
# 1. 基础配置与维度定义
# ==========================================

DIMENSIONS = {
    'D1': {'name': '性别', 'values': ['M', 'F']},
    'D2': {'name': '生命周期', 'values': ['20', '30', '40', '55', '65']},
    'D3': {'name': '学历', 'values': ['EduLo', 'EduMid', 'EduHi']},
    'D4': {'name': '行业赛道', 'values': ['Agri', 'Mfg', 'Service', 'Wht']},
    'D5': {'name': '相对收入', 'values': ['IncL', 'IncML', 'IncM', 'IncMH', 'IncH']},
    'D6': {'name': '家庭状态', 'values': ['Split', 'Unit']}
}

SEGMENT_KEYS = list(itertools.product(
    DIMENSIONS['D1']['values'],
    DIMENSIONS['D2']['values'],
    DIMENSIONS['D3']['values'],
    DIMENSIONS['D4']['values'],
    DIMENSIONS['D5']['values'],
    DIMENSIONS['D6']['values']
))

# ==========================================
# 2. 目标输出 Schema 定义 (修改版)
# ==========================================

# 调整后的列顺序
FINAL_SCHEMA = [
    'Year', 'Month', 'Type_ID', 'Birth_Region', 'From_City',
    'Total_Count', 'Stay_Prob', 'Outflow_Count'
]
for i in range(1, 21):
    FINAL_SCHEMA.extend([f'To_Top{i}', f'To_Top{i}_Count'])

def create_empty_result_row():
    """创建一个符合 FINAL_SCHEMA 的空行字典"""
    row = {
        'Year': 0,
        'Month': 12,           # 固定为 12
        'Type_ID': '',         # 新增：复合键 M_20...
        'Birth_Region': '',    # 新增：四位ID
        'From_City': '',       # 修改：Name(ID)
        'Total_Count': 0,
        'Stay_Prob': 0.0,
        'Outflow_Count': 0
    }
    for i in range(1, 21):
        row[f'To_Top{i}'] = 'None' # 将由ID变为 Name(ID)
        row[f'To_Top{i}_Count'] = 0
    return row

# ==========================================
# 3. 确定性随机工具函数
# ==========================================

def deterministic_choice(indices, probs, seed):
    """
    使用指定种子的确定性随机选择
    确保对于相同的输入,输出永远一致
    """
    rng = np.random.RandomState(seed)
    return rng.choice(indices, p=probs)

def deterministic_shuffle(arr, seed):
    """使用指定种子的确定性洗牌"""
    rng = np.random.RandomState(seed)
    arr_copy = arr.copy()
    rng.shuffle(arr_copy)
    return arr_copy

# --- 城市映射逻辑 (新增) ---
# 将常见的辖区代码映射到主城市代码
CITY_ID_MAPPING = {
    '1101': '1100', '1102': '1100', # 北京市辖区/县 -> 北京
    '1201': '1200', '1202': '1200', # 天津
    '3101': '3100', '3102': '3100', # 上海
    '5001': '5000', '5002': '5000', # 重庆
}

CITY_NAME_MAP = {}

def load_city_names(jsonl_path="city.jsonl"):
    if not os.path.exists(jsonl_path):
        return
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                CITY_NAME_MAP[data['city_id']] = data['name']
            except:
                pass

def get_city_display(city_id):
    name = CITY_NAME_MAP.get(str(city_id), f"未知{city_id}")
    return f"{name}({city_id})"

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

# ==========================================
# 2. 核心算法类
# ==========================================

class CityProfiler:
    """城市画像与演化模型：模拟城市经济特征随时间的变化"""
    
    @staticmethod
    def get_city_tier(city_code):
        if city_code in ['1100', '1200', '3100', '5000', '4401', '4403']:
            return 1 # 超一线/直辖市
        elif city_code[:2] in ['32', '33', '44', '37']: # 江浙粤鲁
            return 2 # 沿海发达
        elif city_code.endswith('01'):
            return 3 # 省会
        else:
            return 4 # 普通地级市

    @staticmethod
    def get_industry_weights(city_code, year):
        """返回该城市该年份的行业分布权重 [Agri, Mfg, Service, Wht]"""
        tier = CityProfiler.get_city_tier(city_code)
        
        # 基础权重
        if tier == 1:
            base = [0.05, 0.25, 0.40, 0.30]
        elif tier == 2:
            base = [0.15, 0.45, 0.30, 0.10]
        elif tier == 3:
            base = [0.25, 0.30, 0.35, 0.10]
        else:
            base = [0.50, 0.20, 0.25, 0.05]
            
        # 时间演化修正 (模拟产业升级)
        if year > 2010:
            # 2010后，Agri减少，Service/Wht增加
            base[0] *= 0.8
            base[2] *= 1.1
            base[3] *= 1.2
        if year > 2018:
            base[2] *= 1.1
            base[3] *= 1.1
            
        # 归一化
        total = sum(base)
        return [x/total for x in base]

class PopulationSynthesizer:
    """人口合成器：生成1200细分人群的初始分布"""
    
    @staticmethod
    def generate_weights(city_code, year):
        # 1. 行业分布 (强地理相关)
        ind_probs = CityProfiler.get_industry_weights(city_code, year)
        ind_map = dict(zip(DIMENSIONS['D4']['values'], ind_probs))
        
        weights = []
        for segment in SEGMENT_KEYS:
            sex, age, edu, ind, inc, fam = segment
            
            w = 1.0
            
            # --- 维度间相关性规则 (Bayesian Priors) ---
            
            # P(Ind | City)
            w *= ind_map[ind]
            
            # P(Inc | Ind) - 收入与行业强相关
            if ind == 'Agri':
                if inc in ['IncL', 'IncML']: w *= 3.0
                elif inc in ['IncMH', 'IncH']: w *= 0.01
            elif ind == 'Wht':
                if inc in ['IncMH', 'IncH']: w *= 2.5
                elif inc == 'IncL': w *= 0.05
            
            # P(Edu | Ind) - 学历与行业强相关
            if ind == 'Wht' and edu == 'EduHi': w *= 4.0
            if ind == 'Agri' and edu == 'EduLo': w *= 3.0
            
            # [修改点]：更新年龄判断逻辑
            # P(Age | Fam)
            if age == '20': # 原 16-24
                if fam == 'Split': w *= 1.5
            elif age in ['40', '55']: # 原 35-49, 50-60
                if fam == 'Unit': w *= 2.0

            # P(Age | Edu)
            if year > 2010 and age == '30' and edu == 'EduHi': w *= 1.5 # 原 25-34
            if age == '65' and edu == 'EduHi': w *= 0.3 # 原 60+
            
            # P(Sex | Ind) - 行业性别偏好 (添加随机噪声)
            if ind == 'Const' and sex == 'M': w *= 2.0 # 如果有建筑业
            
            weights.append(w)
            
        # 归一化权重
        total_w = sum(weights)
        return np.array([w/total_w for w in weights])

class MigrationEngine:
    """迁移引擎：计算流出概率与去向分配"""
    
    @staticmethod
    def calculate_outflow_prob(segment, city_code, year):
        """计算某细分人群离开当前城市的倾向 (0-1 score)"""
        sex, age, edu, ind, inc, fam = segment
        tier = CityProfiler.get_city_tier(city_code)
        
        prob = 0.05 # 基础流动率
        
        # --- 推力因素 (Push Factors) ---
        
        # 1. 经济推力
        if tier >= 3 and inc in ['IncL', 'IncML']:
            prob += 0.20 # 穷困地区低收入者倾向外出
            
        # [修改点]：更新行业推力年龄判断
        if ind == 'Agri' and age in ['20', '30']: # 原 16-24, 25-34
            prob += 0.25

        # 3. 家庭推力
        if fam == 'Split':
            prob += 0.15 # 异地家庭倾向于流动（团聚或返乡）

        # [修改点]：更新年龄因素
        if age == '20': prob *= 2.0 # 原 16-24
        if age == '65': prob *= 0.1 # 原 60+
        
        # 5. 一线城市挤出效应 (高房价)
        if tier == 1 and inc in ['IncL', 'IncML'] and fam == 'Unit':
            prob += 0.1 # 买不起房离开
            
        return min(0.95, prob)

    @staticmethod
    def get_destination_affinity(segment, origin_code, dest_code):
        """计算某细分人群去往特定目的地的适配度 (Affinity Score)"""
        sex, age, edu, ind, inc, fam = segment
        
        origin_coord = CITY_COORDS.get(origin_code, (0,0))
        dest_coord = CITY_COORDS.get(dest_code, (0,0))
        
        dest_tier = CityProfiler.get_city_tier(dest_code)
        
        score = 1.0
        
        # 1. 距离衰减 (Gravity)
        dist = np.sqrt((origin_coord[0]-dest_coord[0])**2 + (origin_coord[1]-dest_coord[1])**2)
        if dist > 0:
            score /= (dist + 0.5)
            
        # 2. 梯度迁移逻辑
        # 高学历/白领 -> 偏好一线/强二线
        if edu == 'EduHi' or ind == 'Wht':
            if dest_tier <= 2: score *= 3.0
            
        # 制造业 -> 偏好沿海/工业城市 (44xx, 32xx)
        if ind == 'Mfg':
            if dest_code.startswith('44') or dest_code.startswith('32'): score *= 2.5
            
        # [修改点]：更新返乡逻辑年龄判断
        if dest_code.startswith('51') or dest_code.startswith('41'):
            if age in ['40', '55'] and fam == 'Split': score *= 2.0 # 原 35-49, 50-60
            
        return score

# ==========================================
# 3. 主处理流程
# ==========================================

def process_row(row_data_str: str):
    """处理CSV中的一行原始数据"""
    
    parts = row_data_str.strip().split('\t')
    if len(parts) < 6: return []
    
    try:
        year = int(parts[0])
        # 提取城市ID "重庆市(5001)" -> "5001"
        from_city_match = re.search(r'\((\d{4})\)', parts[2])
        raw_from_city = from_city_match.group(1) if from_city_match else '0000'
        # 归一化源城市ID
        from_city_id = CITY_ID_MAPPING.get(raw_from_city, raw_from_city)
        
        total_pop = int(float(parts[3]))
        outflow_target = int(float(parts[5]))
        
        # [关键修改] 解析并归一化 Top 去向 {city_id: count}
        top_destinations = {}
        curr_idx = 6
        while curr_idx < len(parts) - 1:
            dest_str = parts[curr_idx]
            cnt_str = parts[curr_idx+1]
            if not dest_str or not cnt_str: break
            
            dest_match = re.search(r'\((\d{4})\)', dest_str)
            if dest_match:
                raw_dest_id = dest_match.group(1)
                
                # 1. 归一化ID (1101 -> 1100)
                d_id = CITY_ID_MAPPING.get(raw_dest_id, raw_dest_id)
                
                # 2. 严格过滤: 只允许出现在CITY_COORDS中的城市
                if d_id not in CITY_COORDS:
                    # 如果归一化后依然不在坐标库中，跳过或记录警告
                    # 这里选择跳过，保证严格合规
                    curr_idx += 2
                    continue

                d_cnt = int(float(cnt_str))
                
                # 3. 聚合计数 (防止csv中出现1101和1102被分别列出，需合并到1100)
                top_destinations[d_id] = top_destinations.get(d_id, 0) + d_cnt
                
            curr_idx += 2
            
    except Exception as e:
        print(f"Error parsing row: {e}")
        return []

    # 2. 生成1200个细分人群的基准人口
    base_probs = PopulationSynthesizer.generate_weights(from_city_id, year)

    # 将总人口分配给1200组 (需取整)
    segment_pops = np.floor(base_probs * total_pop).astype(int)
    # 补齐余数 (使用确定性方法: 将余数分配给权重最大的 segment)
    diff = total_pop - segment_pops.sum()
    if diff > 0:
        # 使用固定的排序方式,确保每次选择相同的 segment
        max_idx = np.argmax(base_probs)
        segment_pops[max_idx] += diff
    
    # 3. 计算流出人口
    # 计算每组的流出倾向
    outflow_probs = np.array([
        MigrationEngine.calculate_outflow_prob(k, from_city_id, year) 
        for k in SEGMENT_KEYS
    ])
    
    # 缩放流出概率以匹配 Outflow_Count
    expected_outflow = np.sum(segment_pops * outflow_probs)
    if expected_outflow == 0: scale_factor = 0
    else: scale_factor = outflow_target / expected_outflow
    
    # 应用缩放并限制在 0-1 之间
    final_outflow_probs = np.clip(outflow_probs * scale_factor, 0, 1.0)
    
    segment_outflows = np.floor(segment_pops * final_outflow_probs).astype(int)
    # 补齐流出余数 (使用确定性方法)
    out_diff = outflow_target - segment_outflows.sum()
    if out_diff > 0:
        # 选择概率最高的100个 segment
        top_indices = np.argsort(final_outflow_probs)[-100:]
        # 使用确定性选择,基于 from_city_id + year + segment_index 作为种子
        for i in range(out_diff):
            seed = int(from_city_id) * 1000 + year * 100 + i
            chosen_idx = deterministic_choice(top_indices, None, seed)
            segment_outflows[chosen_idx] += 1
            
    # 4. 分配去向 (Destination Allocation)
    
    results = []
    
    # 使用归一化后的 keys
    top_dest_ids = list(top_destinations.keys())
    affinity_matrix = np.zeros((len(SEGMENT_KEYS), len(top_dest_ids)))
    
    for i, seg in enumerate(SEGMENT_KEYS):
        for j, dest_id in enumerate(top_dest_ids):
            affinity_matrix[i, j] = MigrationEngine.get_destination_affinity(seg, from_city_id, dest_id)
            
    segment_remaining_outflow = segment_outflows.copy()
    segment_dest_map = [{} for _ in range(len(SEGMENT_KEYS))]
    
    for j, dest_id in enumerate(top_dest_ids):
        target_count = top_destinations[dest_id]
        
        # 该列所有segment的权重 = Affinity * 剩余流出量
        weights = affinity_matrix[:, j] * segment_remaining_outflow
        total_w = np.sum(weights)
        
        if total_w == 0: continue
        
        # 分配
        allocations = np.floor(weights / total_w * target_count).astype(int)
        
        # 修正分配总数误差 (使用确定性方法)
        alloc_diff = target_count - allocations.sum()
        if alloc_diff > 0:
            valid_indices = np.where(segment_remaining_outflow > allocations)[0]
            if len(valid_indices) > 0:
                # 按照权重概率来补齐,使用确定性随机选择
                p_indices = weights[valid_indices] / weights[valid_indices].sum()
                for i in range(alloc_diff):
                    # 使用目标城市ID和迭代索引作为种子
                    seed = int(dest_id) * 1000 + j * 100 + i
                    chosen = deterministic_choice(valid_indices, p_indices, seed)
                    allocations[chosen] += 1
        
        # 更新记录
        for i in range(len(SEGMENT_KEYS)):
            if allocations[i] > 0:
                # [关键] 优先满足Target City的需求，即使需要透支一点点流出量 (Relax constraint slightly to fit Target)
                # 但为了逻辑严密，这里还是取 min，如果 min 导致偏差，说明总流出量不足
                actual = min(allocations[i], segment_remaining_outflow[i])
                segment_dest_map[i][dest_id] = actual
                segment_remaining_outflow[i] -= actual

    # 5. 处理剩余流出 (Non-Top20) -> 确定性随机分配到其他合理城市
    other_cities = [k for k in CITY_COORDS.keys() if k not in top_dest_ids and k != from_city_id]

    for i in range(len(SEGMENT_KEYS)):
        rem = segment_remaining_outflow[i]
        if rem > 0:
            # 使用确定性选择,基于 segment 索引和城市ID作为种子
            if other_cities:
                seed = int(from_city_id) * 10000 + year * 100 + i
                rand_dest = deterministic_choice(other_cities, None, seed)
                segment_dest_map[i][rand_dest] = segment_dest_map[i].get(rand_dest, 0) + rem
            else:
                segment_dest_map[i]['Unknown'] = segment_dest_map[i].get('Unknown', 0) + rem

    # 6. 格式化输出 (符合 FINAL_SCHEMA)
    output_rows = []

    # 获取排序后的 Top 20 去向城市 (按流量降序)
    sorted_top_dests = sorted(top_destinations.items(), key=lambda x: x[1], reverse=True)
    top20_cities = [city_id for city_id, _ in sorted_top_dests[:20]]

    for i, seg in enumerate(SEGMENT_KEYS):
        # 创建符合 Schema 的行
        row = create_empty_result_row()

        # 构造 Type_ID (原 From_City 的 ID 组合部分)
        type_id_str = f"{seg[0]}_{seg[1]}_{seg[2]}_{seg[3]}_{seg[4]}_{seg[5]}_{from_city_id}"

        # 填充基础信息
        row['Year'] = year
        row['Month'] = 12  # 固定为12
        row['Type_ID'] = type_id_str
        row['Birth_Region'] = from_city_id # 采用与当前出发城市相同的四位数编码
        row['From_City'] = get_city_display(from_city_id) # 格式: Name(ID)

        row['Total_Count'] = int(segment_pops[i])
        row['Stay_Prob'] = round(1.0 - final_outflow_probs[i], 6)
        row['Outflow_Count'] = int(segment_outflows[i])

        # 填充 Top 20 去向
        dest_map = segment_dest_map[i]

        # 将 dest_map 按流量降序排序
        sorted_dests = sorted(dest_map.items(), key=lambda x: x[1], reverse=True)

        # 先填充 CSV 中定义的 Top 20 城市
        filled_count = 0
        for city_id in top20_cities:
            count = dest_map.get(city_id, 0)
            if count > 0 or filled_count < len(sorted_dests):
                # 修改：这里写入 Name(ID) 格式
                row[f'To_Top{filled_count+1}'] = get_city_display(city_id)
                row[f'To_Top{filled_count+1}_Count'] = int(count)
                filled_count += 1

        # 如果还有其他城市,继续填充
        for city_id, count in sorted_dests:
            if city_id not in top20_cities and filled_count < 20:
                # 修改：这里写入 Name(ID) 格式
                row[f'To_Top{filled_count+1}'] = get_city_display(city_id)
                row[f'To_Top{filled_count+1}_Count'] = int(count)
                filled_count += 1

        output_rows.append(row)

    return output_rows, top_destinations

# ==========================================
# 4. 执行入口 (Example)
# ==========================================

def validate_results(results, target_pop, target_outflow, top_destinations,
                    show_summary=True, show_top20_validation=True,
                    show_long_tail=True, show_statistics=True):
    """
    验证生成结果的准确性,并打印长尾分布
    适配 Name(ID) 格式

    参数:
    - show_summary: 是否打印总体验证
    - show_top20_validation: 是否打印 Top20 城市验证
    - show_long_tail: 是否打印长尾去向统计
    - show_statistics: 是否打印统计摘要
    """

    if show_summary:
        print("\n" + "="*120)
        print("数据验证报告".center(120))
        print("="*120)

    # 1. 验证总人口和总流出人口
    total_pop_check = sum(r['Total_Count'] for r in results)
    total_out_check = sum(r['Outflow_Count'] for r in results)

    if show_summary:
        print(f"\n【总体验证】")
        print(f"  总人口:   生成值={total_pop_check:,}, 目标值={target_pop:,}, 偏差={total_pop_check - target_pop:,}")
        print(f"  总流出:   生成值={total_out_check:,}, 目标值={target_outflow:,}, 偏差={total_out_check - target_outflow:,}")

    # 统计生成数据中每个城市的接收量
    generated_dest_totals = {}

    # 预编译正则，用于提取 ID
    id_pattern = re.compile(r'\((\d{4})\)')

    for r in results:
        for i in range(1, 21):
            dest_str = r[f'To_Top{i}'] # 这里现在是 "Name(ID)" 或 "None"
            count = r[f'To_Top{i}_Count']

            if dest_str != 'None' and dest_str != 'Unknown' and count > 0:
                # 从 Name(ID) 中提取 ID
                match = id_pattern.search(dest_str)
                if match:
                    dest_id = match.group(1)
                else:
                    dest_id = dest_str # 容错 fallback

                generated_dest_totals[dest_id] = generated_dest_totals.get(dest_id, 0) + count

    # ---------------------------------------------------------
    # 2. 验证 CSV 中定义的 Top 20 城市 (刚性约束)
    # ---------------------------------------------------------
    all_match = True
    top20_sum = 0

    for dest_id in sorted(top_destinations.keys()):
        generated = generated_dest_totals.get(dest_id, 0)
        target = top_destinations[dest_id]
        diff = generated - target
        top20_sum += target
        ratio = target / target_outflow * 100  # 计算占比

        if diff == 0:
            status = "[OK] 匹配"
        else:
            status = f"[ERROR] 偏差 {diff:,}"
            all_match = False

    if show_top20_validation:
        print(f"\n【核心去向验证】(CSV中定义的Top20)")
        print(f"{'城市ID':<8} {'城市名称':<15} {'生成值':>12} {'目标值':>12} {'偏差':>12} {'占比':>10} {'状态'}")
        print("-" * 110)

        for dest_id in sorted(top_destinations.keys()):
            generated = generated_dest_totals.get(dest_id, 0)
            target = top_destinations[dest_id]
            diff = generated - target
            ratio = target / target_outflow * 100

            if diff == 0:
                status = "[OK] 匹配"
            else:
                status = f"[ERROR] 偏差 {diff:,}"

            print(f"{dest_id:<8} {get_city_display(dest_id):<15} {generated:>12,} {target:>12,} {diff:>12,} {ratio:>9.4f}% {status}")

        print("-" * 110)
        # 打印Top20汇总
        top20_ratio = top20_sum / target_outflow * 100
        print(f"  Top20合计: {top20_sum:>12,} (占总流出: {top20_ratio:>9.4f}%)")
        if all_match:
            print("[OK] 所有Top20城市的去向数据完全匹配!")
        else:
            print("[ERROR] 部分城市的去向数据存在偏差")

    # ---------------------------------------------------------
    # 3. 打印长尾去向 (Top 20 以外的城市)
    # ---------------------------------------------------------
    # 计算长尾总流量
    long_tail_total = target_outflow - top20_sum
    long_tail_ratio = long_tail_total / target_outflow * 100

    # 筛选出不在 top_destinations 中的城市
    long_tail_cities = []
    for dest_id, count in generated_dest_totals.items():
        if dest_id not in top_destinations:
            long_tail_cities.append((dest_id, count))

    # 按流量降序排列
    long_tail_cities.sort(key=lambda x: x[1], reverse=True)

    if show_long_tail:
        top20_ratio = top20_sum / target_outflow * 100
        print(f"\n【长尾去向统计】(CSV未定义，由算法补齐)")
        print(f"  总流出: {target_outflow:,}")
        print(f"  Top20总和: {top20_sum:,} (占 {top20_ratio:>9.4f}%)")
        print(f"  长尾流量: {long_tail_total:,} (占 {long_tail_ratio:>9.4f}%)")
        print(f"  这些流量被分配到了以下非Top20城市：")
        print("-" * 100)
        print(f"{'排名':<6} {'城市ID':<8} {'城市名称':<15} {'生成流向数量':>12} {'占比':>10}")
        print("-" * 100)

        # 打印前 50 个长尾城市 (避免列表过长)
        for idx, (d_id, cnt) in enumerate(long_tail_cities):
            if idx >= 50: # 只显示前50个，防止刷屏，可自行调整
                break
            ratio = cnt / total_out_check * 100
            print(f"{idx+1:<6} {d_id:<8} {get_city_display(d_id):<15} {cnt:>12,} {ratio:>9.4f}%")

        if len(long_tail_cities) > 50:
            print(f"... 以及其他 {len(long_tail_cities) - 50} 个长尾城市")

        print("-" * 100)
        print(f"  长尾城市总数: {len(long_tail_cities)} 个")

    # ---------------------------------------------------------
    # 4. 完整性统计
    # ---------------------------------------------------------
    if show_statistics:
        print(f"\n【统计摘要】")
        print(f"  生成记录数: {len(results)} 条Segment")
        print(f"  有流向的城市总数: {len(generated_dest_totals)} 个 (Top20 + 长尾)")

    if show_summary or show_top20_validation or show_long_tail or show_statistics:
        print("="*120)

def save_to_duckdb(results, db_path="output.db", table_name="migration_data"):
    """
    将结果写入 DuckDB 数据库
    """
    if not results:
        print("无数据需要写入。")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(results)
    df = df[FINAL_SCHEMA] # 确保列顺序

    print(f"\n正在写入 DuckDB: {db_path} ...")

    try:
        # 连接 DuckDB (文件模式)
        con = duckdb.connect(db_path)

        # 写入数据 (如果表存在则追加，这里简单演示覆盖或新建，视需求可调整)
        # 这里的 behavior='replace' 会覆盖同名表，如果需要追加请改用 append
        con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df LIMIT 0")
        con.execute(f"INSERT INTO {table_name} SELECT * FROM df")

        # 获取行数确认
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

        # 采样查询 10 条
        sample_df = con.execute(f"SELECT * FROM {table_name} LIMIT 10").fetch_df()

        con.close()

        print(f"[OK] 写入成功！当前表 {table_name} 总行数: {count}")
        return sample_df

    except Exception as e:
        print(f"[ERROR] 写入 DuckDB 失败: {e}")
        return None


if __name__ == "__main__":
    import os
    import time
    import duckdb

    # ==========================================
    # 核心配置
    # ==========================================
    # 模式选择: 'SINGLE' (单行调试) 或 'ALL' (全表生成)
    PROCESS_MODE = 'ALL'  
    
    # 数据库路径
    DB_FILE = "output.db"
    CSV_FILE = "constrain.csv"
    TABLE_NAME = "migration_records"

    # --- 单行模式配置 (仅当 PROCESS_MODE='SINGLE' 时生效) ---
    TARGET_ROW_INDEX = 19       # 要处理的 CSV 行索引 (从 0 开始计数，不含表头)
    
    # --- 打印开关配置 (主要用于单行模式调试) ---
    SHOW_DUCKDB_SAMPLE = True        # 是否打印 DuckDB 采样数据
    SHOW_VALIDATION_SUMMARY = True   # 是否打印总体验证
    SHOW_TOP20_VALIDATION = True     # 是否打印 Top20 验证
    SHOW_LONG_TAIL = True            # 是否打印长尾统计
    SHOW_STATISTICS = True           # 是否打印统计摘要

    # --- 全表模式配置 ---
    BATCH_SIZE = 50  # 每处理多少行 CSV 写入一次数据库 (提高效率)

    # ==========================================
    # 辅助函数：获取已处理的记录标识
    # ==========================================
    def get_processed_keys(db_path, table_name):
        """获取数据库中已存在的 (Year, From_City) 组合，用于断点续传"""
        processed = set()
        # 初始化 DuckDB 连接并检查表是否存在
        try:
            con = duckdb.connect(db_path)
            # 检查表是否存在
            try:
                # DuckDB 查看所有表的方式
                con.execute(f"SELECT count(*) FROM {table_name}")
            except:
                # 表不存在，直接返回
                con.close()
                return processed
            
            print("正在读取断点信息 (可能需要几秒钟)...")
            # 提取 Year 和 Birth_Region (即 From_City ID) 用于去重
            rows = con.execute(f"SELECT DISTINCT Year, Birth_Region FROM {table_name}").fetchall()
            for r in rows:
                processed.add((str(r[0]), str(r[1]))) 
            con.close()
            print(f"检测到断点：已处理 {len(processed)} 个城市-年份组合。")
        except Exception as e:
            print(f"读取断点失败 (如果是首次运行请忽略): {e}")
        return processed

    # ==========================================
    # 主逻辑
    # ==========================================

    # 1. 初始化资源
    load_city_names("city.jsonl")

    if not os.path.exists(CSV_FILE):
        print(f"错误: 找不到文件 {CSV_FILE}")
        exit(1)

    print(f"正在读取文件: {CSV_FILE}")
    
    # [修复点] 使用 utf-8-sig 自动处理 BOM 头
    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    if not lines:
        print("CSV 文件为空")
        exit(1)

    # 识别分隔符
    first_line = lines[0].strip()
    separator = '\t' if '\t' in first_line else ','
    
    # [修复点] 更稳健的表头识别逻辑
    start_idx = 0
    # 如果第一行不是数字开头，或者显式包含 Year/年份，则认为是表头
    if not first_line[0].isdigit() or first_line.startswith('Year') or first_line.startswith('年份'):
        start_idx = 1
    
    data_lines = lines[start_idx:]
    total_lines = len(data_lines)
    print(f"共发现 {total_lines} 行待处理数据 (已跳过表头: {'是' if start_idx==1 else '否'})。")

    # -------------------------------------------------------------------------
    # 模式 A: 单行调试模式
    # -------------------------------------------------------------------------
    if PROCESS_MODE == 'SINGLE':
        if TARGET_ROW_INDEX >= total_lines:
            print(f"错误: 目标索引 {TARGET_ROW_INDEX} 超出范围 (最大 {total_lines-1})")
            exit(1)

        target_line = data_lines[TARGET_ROW_INDEX].strip()
        print(f"\n[单行模式] 正在处理第 {start_idx + TARGET_ROW_INDEX} 行数据...")
        
        # 解析目标行以获取验证所需的 total_pop 等
        parts = target_line.split(separator)
        target_year = int(parts[0])
        target_pop = int(float(parts[3]))
        target_outflow = int(float(parts[5]))

        # 处理
        target_line_tab = target_line.replace(',', '\t')
        results, normalized_targets = process_row(target_line_tab)

        if results:
            save_to_duckdb(results, DB_FILE, TABLE_NAME)
            
            # 详细验证打印
            validate_results(
                results, target_pop, target_outflow, normalized_targets,
                # 注意：这里如果之前的函数签名没改，可能需要手动适配参数，或者直接传参
                # 这里假设 validate_results 内部逻辑已适配
            )
            
            if SHOW_DUCKDB_SAMPLE:
                con = duckdb.connect(DB_FILE)
                sample = con.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 10").fetch_df()
                con.close()
                print("\n" + "="*50)
                print("DuckDB 最近写入采样")
                print("="*50)
                print(sample.to_string(index=False))

    # -------------------------------------------------------------------------
    # 模式 B: 全表批量生成 (支持断点续传)
    # -------------------------------------------------------------------------
    elif PROCESS_MODE == 'ALL':
        print(f"\n[全表模式] 开始处理... (Batch Size: {BATCH_SIZE})")
        
        # 获取已处理列表用于断点续传 (Year, From_City_ID)
        processed_keys = get_processed_keys(DB_FILE, TABLE_NAME)
        
        buffer_results = []
        batch_count = 0
        start_time = time.time()
        skipped_count = 0
        processed_count = 0

        # 连接数据库 (保持连接打开以提高性能)
        con = duckdb.connect(DB_FILE)

        try:
            for i, line_str in enumerate(data_lines):
                line_str = line_str.strip()
                if not line_str: continue

                parts = line_str.split(separator)
                
                try:
                    # 获取年份和城市ID用于断点判断
                    curr_year = str(int(parts[0]))
                    
                    # 提取 CityID: "城市(ID)" -> ID
                    city_match = re.search(r'\((\d{4})\)', parts[2])
                    raw_city_id = city_match.group(1) if city_match else '0000'
                    curr_city_id = CITY_ID_MAPPING.get(raw_city_id, raw_city_id)
                    
                    # 断点检查
                    if (curr_year, curr_city_id) in processed_keys:
                        skipped_count += 1
                        continue

                    # 处理行
                    line_tab = line_str.replace(',', '\t')
                    row_results, _ = process_row(line_tab) 
                    
                    if row_results:
                        buffer_results.extend(row_results)
                        processed_count += 1

                    # 缓冲区满或最后一行，执行写入
                    if len(buffer_results) >= (BATCH_SIZE * 1200) or (i == total_lines - 1 and buffer_results):
                        # 批量写入
                        if buffer_results:
                            df_batch = pd.DataFrame(buffer_results)
                            df_batch = df_batch[FINAL_SCHEMA] # 确保列顺序
                            
                            # 使用当前的连接执行写入
                            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM df_batch LIMIT 0")
                            con.execute(f"INSERT INTO {TABLE_NAME} SELECT * FROM df_batch")
                            
                            batch_count += 1
                            elapsed = time.time() - start_time
                            avg_speed = processed_count / elapsed if elapsed > 0 else 0
                            
                            print(f"进度: {i+1}/{total_lines} | 跳过: {skipped_count} | 本批写入: {len(df_batch)} 条 | 耗时: {elapsed:.1f}s | 速度: {avg_speed:.2f} 行/秒")
                            
                            buffer_results = [] # 清空缓冲

                except Exception as e:
                    print(f"行 {i+start_idx} 数据异常: {e} -> 内容: {line_str[:50]}...")
                    continue
        finally:
            con.close()

        print(f"\n[完成] 全表处理结束。总共跳过 {skipped_count} 行 (已存在)，新处理 {processed_count} 行。")