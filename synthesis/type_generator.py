"""
Type生成器：生成所有可能的Type组合
"""

import itertools
from typing import List, Tuple, Dict
from config import DIMENSIONS


def generate_all_types() -> List[Dict[str, str]]:
    """
    生成所有可能的Type组合
    
    Returns:
        List[Dict]: 每个Type是一个字典，包含所有维度的值
    """
    # 获取所有维度的值列表
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


def type_to_id(type_dict: Dict[str, str]) -> str:
    """
    将Type字典转换为Type_ID字符串
    
    Args:
        type_dict: Type字典
        
    Returns:
        str: Type_ID，格式如 "M_25_EduLo_Mfg_IncL_Split"
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
    
    return '_'.join(parts)


def get_type_label(type_dict: Dict[str, str]) -> str:
    """
    获取Type的中文标签（用于显示）
    
    Args:
        type_dict: Type字典
        
    Returns:
        str: 中文标签
    """
    labels = []
    for key in ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']:
        dim = DIMENSIONS[key]
        value_idx = dim['values'].index(type_dict[key])
        labels.append(dim['labels'][value_idx])
    
    return ' | '.join(labels)


if __name__ == '__main__':
    # 测试
    types = generate_all_types()
    print(f"总Type数量: {len(types)}")
    print(f"理论组合数: {2 * 5 * 3 * 3 * 5 * 2} = {2 * 5 * 3 * 3 * 5 * 2}")
    
    # 显示前5个Type
    print("\n前5个Type示例:")
    for i, t in enumerate(types[:5]):
        print(f"{i+1}. {type_to_id(t)}")
        print(f"   {get_type_label(t)}")
