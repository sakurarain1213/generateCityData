"""
Type生成器：生成所有可能的Type组合
"""

import itertools
import random
import numpy as np
from typing import List, Tuple, Dict
from config import DIMENSIONS, REGION_CODES, CITIES


def generate_all_types() -> List[Dict[str, str]]:
    """
    生成所有可能的Type组合（不包含Region）

    Returns:
        List[Dict]: 每个Type是一个字典，包含D1-D6维度的值
    """
    # 获取所有维度的值列表（不包含D7/Region）
    dimension_keys = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    dimension_values = [DIMENSIONS[key]['values'] for key in dimension_keys]

    # 生成所有组合
    all_combinations = list(itertools.product(*dimension_values))

    # 转换为字典列表
    types = []
    for combo in all_combinations:
        type_dict = {}
        for i, key in enumerate(dimension_keys):
            type_dict[key] = combo[i]
        types.append(type_dict)

    return types


def generate_city_types() -> List[Dict[str, str]]:
    """
    生成所有城市×Type的组合（D7作为与其他维度并列）

    总数量 = 1200 (基础Type) × 300+ (城市数量)

    Returns:
        List[Dict]: 每个Type是一个字典，包含D1-D7所有维度的值
    """
    # 先生成基础Type（D1-D6）
    base_types = generate_all_types()

    # 为每个城市生成Type组合
    city_types = []
    for city_code, city_name in CITIES:
        for base_type in base_types:
            # 复制基础Type并添加D7 (Region)
            type_with_region = base_type.copy()
            type_with_region['D7'] = city_code
            city_types.append(type_with_region)

    return city_types


def type_to_id(type_dict: Dict[str, str]) -> str:
    """
    将Type字典转换为Type_ID字符串
    注意：D7 (Region) 现在已经包含在 type_dict 中

    Args:
        type_dict: Type字典（包含D1-D7所有维度）

    Returns:
        str: Type_ID，格式如 "M_25_EduLo_Mfg_IncL_Split_1100"
    """
    # 映射规则
    parts = []

    # D1: 性别
    parts.append(type_dict['D1'])

    # D2: 生命周期（取年龄范围的中点或代表值）
    age_map = {
        '16-24': '20',
        '25-34': '30',
        '35-49': '40',
        '50-60': '55',
        '60+': '65'
    }
    parts.append(age_map.get(type_dict['D2'], type_dict['D2']))

    # D3: 学历
    parts.append(type_dict['D3'])

    # D4: 行业
    parts.append(type_dict['D4'])

    # D5: 收入
    parts.append(type_dict['D5'])

    # D6: 家庭状态
    parts.append(type_dict['D6'])

    # D7: Region（城市编码，已在type_dict中）
    parts.append(type_dict['D7'])

    return '_'.join(parts)


def get_type_label(type_dict: Dict[str, str]) -> str:
    """
    获取Type的中文标签（用于显示）

    Args:
        type_dict: Type字典（包含D1-D7）

    Returns:
        str: 中文标签
    """
    labels = []

    # 处理D1-D6（在DIMENSIONS中定义）
    for key in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        dim = DIMENSIONS[key]
        value_idx = dim['values'].index(type_dict[key])
        labels.append(dim['labels'][value_idx])

    # 处理D7（城市）
    if 'D7' in type_dict:
        city_code = type_dict['D7']
        # 从CITIES列表中查找城市名称
        city_name = None
        for code, name in CITIES:
            if code == city_code:
                city_name = name
                break
        if city_name:
            labels.append(f"地区:{city_name}")
        else:
            labels.append(f"地区:{city_code}")

    return ' | '.join(labels)


if __name__ == '__main__':
    # 测试
    print("=== 测试基础Type生成 (D1-D6) ===")
    base_types = generate_all_types()
    print(f"基础Type数量: {len(base_types)}")
    print(f"理论组合数: {2 * 5 * 3 * 3 * 5 * 2} = {2 * 5 * 3 * 3 * 5 * 2}")

    # 显示前5个基础Type
    print("\n前5个基础Type示例 (不含Region):")
    for i, t in enumerate(base_types[:5]):
        print(f"{i+1}. {t}")

    print("\n=== 测试城市Type生成 (D1-D7) ===")
    city_types = generate_city_types()
    print(f"城市Type数量: {len(city_types)}")
    print(f"理论组合数: {len(base_types)} × {len(CITIES)} = {len(base_types) * len(CITIES)}")

    # 显示前5个城市Type
    print("\n前5个城市Type示例 (含Region):")
    for i, t in enumerate(city_types[:5]):
        print(f"{i+1}. {type_to_id(t)}")
        print(f"   {get_type_label(t)}")
