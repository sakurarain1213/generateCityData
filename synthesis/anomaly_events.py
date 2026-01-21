"""
异常事件定义：影响迁移的特殊事件
如：疫情、自然灾害、重大政策变化等
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 异常事件定义
# 格式：事件名称: {
#     'start_date': (年, 月),
#     'end_date': (年, 月),
#     'affected_cities': [城市代码列表],
#     'migration_multiplier': 迁移概率乘数,
#     'description': 描述
# }
ANOMALY_EVENTS = {
    # 示例：2020年1-3月疫情（武汉及周边）
    'covid_wuhan_2020': {
        'start_date': (2020, 1),
        'end_date': (2020, 3),
        'affected_cities': ['4201', '4202', '4203', '4205', '4206', '4207', '4208', '4209', '4210', '4211', '4212', '4213'],
        'migration_multiplier': 0.3,  # 迁移概率大幅降低（封城）
        'description': '2020年1-3月武汉及周边疫情封城'
    },
    
    # 示例：2022年3-5月上海疫情
    'covid_shanghai_2022': {
        'start_date': (2022, 3),
        'end_date': (2022, 5),
        'affected_cities': ['3100', '3201', '3202', '3203', '3204', '3205', '3206'],
        'migration_multiplier': 0.4,  # 迁移概率降低
        'description': '2022年3-5月上海及周边疫情'
    },
    
    # 示例：2021年7月河南暴雨
    'henan_flood_2021': {
        'start_date': (2021, 7),
        'end_date': (2021, 7),
        'affected_cities': ['4101', '4102', '4103', '4104', '4105', '4106', '4107', '4108', '4109', '4110', '4111', '4112', '4113', '4114', '4115', '4116', '4117'],
        'migration_multiplier': 1.5,  # 迁移概率增加（灾后重建、人员流动）
        'description': '2021年7月河南暴雨'
    },
    
    # 示例：2024年春节（正常季节性，但可以标记为特殊事件）
    'spring_festival_2024': {
        'start_date': (2024, 1),
        'end_date': (2024, 2),
        'affected_cities': [],  # 全国范围
        'migration_multiplier': 0.6,  # 春节期间迁移意愿低
        'description': '2024年春节（1-2月）'
    },
}

def get_anomaly_multiplier(city_code: str, year: int, month: int) -> float:
    """
    获取指定城市在指定时间的异常事件乘数
    
    Args:
        city_code: 城市代码
        year: 年份
        month: 月份
    
    Returns:
        float: 迁移概率乘数（1.0表示无影响）
    """
    multiplier = 1.0
    
    for event_name, event_info in ANOMALY_EVENTS.items():
        start_year, start_month = event_info['start_date']
        end_year, end_month = event_info['end_date']
        
        # 检查时间是否在事件期间
        start_date_num = start_year * 12 + start_month
        end_date_num = end_year * 12 + end_month
        current_date_num = year * 12 + month
        
        if start_date_num <= current_date_num <= end_date_num:
            # 检查城市是否受影响
            affected_cities = event_info['affected_cities']
            if not affected_cities or city_code in affected_cities:
                # 应用乘数（取最小值，表示最严重的影响）
                multiplier = min(multiplier, event_info['migration_multiplier'])
    
    return multiplier

def get_historical_trend(year: int, month: int) -> float:
    """
    获取历史迁移趋势（模拟）
    可以基于历史数据或趋势模型
    
    Args:
        year: 年份
        month: 月份
    
    Returns:
        float: 趋势乘数（1.0表示正常）
    """
    # 模拟历史趋势：
    # 2020-2021: 疫情影响，迁移意愿低
    # 2022-2023: 逐步恢复
    # 2024+: 正常水平
    
    if year < 2020:
        return 1.0  # 正常
    elif year == 2020:
        if month <= 3:
            return 0.5  # 疫情初期，大幅降低
        elif month <= 6:
            return 0.7  # 逐步恢复
        else:
            return 0.9  # 继续恢复
    elif year == 2021:
        return 0.95  # 继续恢复
    elif year == 2022:
        if month <= 5:
            return 0.8  # 上海疫情
        else:
            return 0.95  # 恢复
    elif year == 2023:
        return 1.0  # 恢复正常
    else:  # 2024+
        return 1.0  # 正常水平
