"""
人口分布生成器：生成Type的Total_Count（考虑长尾分布和城市规模）
"""

import numpy as np
import random  # 新增：用于打乱Type顺序
from typing import Dict, List, Optional
from config import (
    MIN_TYPE_COUNT, TOTAL_POPULATION_BASE, CITY_POPULATION_WEIGHTS, 
    TIER1_CITIES, TIER2_CITIES,
    POWER_LAW_ALPHA, CITY_POP_NOISE_STABLE, CITY_POP_NOISE_TIME, CITY_POP_NOISE_EXTRA
)


def generate_type_counts(types: List[Dict[str, str]], type_to_id_func) -> Dict[str, int]:
    """
    为每个Type生成全局Total_Count（不考虑城市），考虑长尾分布
    
    策略：
    1. 打乱Type顺序，避免字典序偏差
    2. 使用优化的幂律分布（更平缓）
    3. 某些Type组合更常见（如：中等年龄、中等学历、中等收入）
    4. 某些Type组合很少见（如：高年龄+低学历+高收入）
    
    Args:
        types: Type列表
        type_to_id_func: Type_ID生成函数
        
    Returns:
        Dict[str, int]: {Type_ID: Total_Count} - 全局Type人口数
    """
    type_counts = {}
    
    # 1. 复制并打乱 types 列表，打破 itertools 的字典序
    # 这样男性/女性、年轻/年老就会随机分布在列表的不同位置
    shuffled_types = types.copy()
    # 使用固定的种子确保每次运行结果一致，但内部是乱序的
    random.Random(42).shuffle(shuffled_types)
    
    # 生成幂律分布的权重
    n = len(shuffled_types)
    # 使用优化后的ALPHA值：从2.5降低到更合理的值
    weights = np.array([(i+1)**(-POWER_LAW_ALPHA) for i in range(n)])
    weights = weights / weights.sum()  # 归一化
    
    # 使用配置的总人口基数
    total_population = TOTAL_POPULATION_BASE
    
    # 为每个Type分配人口
    # 注意：这里遍历的是 shuffled_types（打乱后的）
    for i, type_dict in enumerate(shuffled_types):
        # 基础权重 （现在高权重随机分配给了不同特征的人）
        base_weight = weights[i]
        
        # 根据Type特征调整权重（某些组合更常见）
        multiplier = _calculate_type_multiplier(type_dict)
        
        # 计算最终人口数（全局）
        count = int(total_population * base_weight * multiplier)
        
        # 应用长尾控制：小于阈值的设为0
        if count < MIN_TYPE_COUNT:
            count = 0
        
        # 只记录有效的Type，避免存储大量空数据
        if count > 0:
            # 生成Type_ID
            type_id = type_to_id_func(type_dict)
            type_counts[type_id] = count
    
    return type_counts


def calculate_city_type_count(
    type_id: str, 
    city_code: str, 
    global_type_count: int,
    year: Optional[int] = None,
    month: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None
) -> int:
    """
    计算特定城市×Type的人口数
    
    策略：
    1. 基于城市规模权重分配
    2. 考虑Type特征与城市的匹配度
    3. 添加多层随机噪声（确保不同行都有差异）
    4. 考虑时间因素（不同月份可能有细微变化）
    
    Args:
        type_id: Type_ID
        city_code: 城市代码
        global_type_count: 该Type的全局人口数
        year: 年份（用于时间维度的随机性）
        month: 月份（用于时间维度的随机性）
        rng: 随机数生成器（可选）
        
    Returns:
        int: 该Type在该城市的人口数
    """
    if rng is None:
        rng = np.random.RandomState()
    
    # 获取城市规模权重
    city_weight = CITY_POPULATION_WEIGHTS.get(city_code, None)
    
    if city_weight is None:
        # 如果没有配置，根据城市等级估算
        if city_code in TIER1_CITIES:
            city_weight = 0.8
        elif city_code in TIER2_CITIES:
            city_weight = 0.3
        else:
            # 其他城市：基于城市代码生成稳定的随机权重（确保同一城市权重稳定）
            city_seed = hash(city_code) % (2**31)
            city_rng = np.random.RandomState(city_seed)
            city_weight = 0.05 + city_rng.uniform(0, 0.15)
    
    # 计算基础人口数（按城市规模分配）
    # 策略：将全局Type人口数按城市权重分配到各城市
    # 假设所有城市权重总和约为20（一线4个×1.0 + 新一线10个×0.5 + 其他300个×0.1 ≈ 50）
    # 使用归一化权重：city_weight / 平均权重
    # 简化：直接按权重比例分配，使用调整因子避免人口过多
    avg_city_weight = 0.2  # 平均城市权重（估算）
    base_count = int(global_type_count * city_weight / avg_city_weight * 0.01)  # 0.01是调整因子
    
    # 根据Type特征调整（某些Type在某些城市更常见）
    type_multiplier = _calculate_city_type_match(type_id, city_code)
    base_count = int(base_count * type_multiplier)
    
    # 第一层随机噪声：基于Type×城市的稳定噪声（可配置）
    # 这确保不同城市×Type组合有足够差异
    stable_seed = hash(f"{type_id}_{city_code}") % (2**31)
    stable_rng = np.random.RandomState(stable_seed)
    stable_noise = 1.0 + stable_rng.uniform(-CITY_POP_NOISE_STABLE, CITY_POP_NOISE_STABLE)
    base_count = int(base_count * stable_noise)
    
    # 第二层随机噪声：基于时间的动态噪声（可配置）
    # 这确保不同月份有细微差异，避免完全相同的Total_Count
    if year is not None and month is not None:
        time_seed = hash(f"{type_id}_{city_code}_{year}_{month}") % (2**31)
        time_rng = np.random.RandomState(time_seed)
        time_noise = 1.0 + time_rng.uniform(-CITY_POP_NOISE_TIME, CITY_POP_NOISE_TIME)
        base_count = int(base_count * time_noise)
    else:
        # 如果没有提供时间，使用通用随机噪声
        time_noise = 1.0 + rng.uniform(-CITY_POP_NOISE_TIME, CITY_POP_NOISE_TIME)
        base_count = int(base_count * time_noise)
    
    # 第三层随机噪声：额外的随机扰动（可配置）
    # 进一步确保每行数据都有差异
    extra_noise = 1.0 + rng.uniform(-CITY_POP_NOISE_EXTRA, CITY_POP_NOISE_EXTRA)
    final_count = int(base_count * extra_noise)
    
    # 确保不小于最小值
    final_count = max(final_count, MIN_TYPE_COUNT // 10)  # 城市级别的最小值更小
    
    return final_count


def _calculate_city_type_match(type_id: str, city_code: str) -> float:
    """
    计算Type与城市的匹配度
    
    例如：
    - 高学历Type在一线城市更常见
    - 蓝领制造Type在制造业城市更常见
    - 高收入Type在一线城市更常见
    
    Args:
        type_id: Type_ID
        city_code: 城市代码
        
    Returns:
        float: 匹配度乘数
    """
    multiplier = 1.0
    
    # 解析Type_ID
    parts = type_id.split('_')
    if len(parts) < 6:
        return 1.0
    
    edu = parts[2]  # 学历
    industry = parts[3]  # 行业
    income = parts[4]  # 收入
    
    # 一线城市：高学历、高收入Type更常见
    if city_code in TIER1_CITIES:
        if edu == 'EduHi':
            multiplier *= 1.3
        elif edu == 'EduLo':
            multiplier *= 0.8
        
        if income in ['IncMH', 'IncH']:
            multiplier *= 1.2
        elif income in ['IncL', 'IncML']:
            multiplier *= 0.9
    
    # 制造业城市：蓝领制造Type更常见
    from city_geo_econ import get_city_industry_type
    city_industry = get_city_industry_type(city_code)
    if city_industry == 1 and industry == 'Mfg':  # 制造业城市 + 制造业Type
        multiplier *= 1.3
    elif city_industry == 2 and industry == 'Service':  # 服务业城市 + 服务业Type
        multiplier *= 1.2
    
    return multiplier


def _calculate_type_multiplier(type_dict: Dict[str, str]) -> float:
    """
    计算Type的常见度乘数
    
    某些组合更常见：
    - 中等年龄（25-49）更常见
    - 中等学历更常见
    - 中等收入更常见
    - 团聚状态更常见（在稳定城市）
    
    Args:
        type_dict: Type字典
        
    Returns:
        float: 乘数
    """
    multiplier = 1.0
    
    # 年龄：中等年龄更常见
    age_multiplier = {
        '16-24': 0.8,
        '25-34': 1.5,  # 最活跃
        '35-49': 1.3,
        '50-60': 0.6,
        '60+': 0.3
    }
    multiplier *= age_multiplier.get(type_dict['D2'], 1.0)
    
    # 学历：中等学历更常见
    edu_multiplier = {
        'EduLo': 1.2,   # 低学历人数多
        'EduMid': 1.5,  # 中等学历最多
        'EduHi': 0.8    # 高学历相对少
    }
    multiplier *= edu_multiplier.get(type_dict['D3'], 1.0)
    
    # 收入：中等收入更常见
    income_multiplier = {
        'IncL': 1.1,
        'IncML': 1.3,
        'IncM': 1.5,    # 中等收入最多
        'IncMH': 1.0,
        'IncH': 0.5     # 高收入少
    }
    multiplier *= income_multiplier.get(type_dict['D5'], 1.0)
    
    # 行业：传统服务和蓝领更常见
    industry_multiplier = {
        'Mfg': 1.2,
        'Service': 1.4,  # 传统服务最多
        'Wht': 0.8
    }
    multiplier *= industry_multiplier.get(type_dict['D4'], 1.0)
    
    # 家庭状态：团聚更常见（在稳定城市）
    family_multiplier = {
        'Split': 0.9,
        'Unit': 1.1
    }
    multiplier *= family_multiplier.get(type_dict['D6'], 1.0)
    
    return multiplier


def filter_valid_types(type_counts: Dict[str, int]) -> Dict[str, int]:
    """
    过滤掉Count为0的Type（长尾控制）
    
    Args:
        type_counts: Type计数字典
        
    Returns:
        Dict[str, int]: 过滤后的字典
    """
    return {tid: count for tid, count in type_counts.items() if count > 0}


if __name__ == '__main__':
    # 测试
    from type_generator import generate_all_types, type_to_id
    
    types = generate_all_types()
    print(f"总Type数: {len(types)}")
    
    type_counts = generate_type_counts(types, type_to_id)
    valid_counts = filter_valid_types(type_counts)
    
    print(f"有效Type数: {len(valid_counts)}")
    print(f"总人口: {sum(valid_counts.values()):,}")
    
    # 显示前10个最多的Type
    sorted_types = sorted(valid_counts.items(), key=lambda x: x[1], reverse=True)
    print("\n前10个最多的Type:")
    for i, (tid, count) in enumerate(sorted_types[:10]):
        print(f"{i+1}. {tid}: {count:,}")
