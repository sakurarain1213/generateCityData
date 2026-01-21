"""
城市地理和经济数据：经纬度、GDP、产业特点
"""

import numpy as np
from typing import Dict, Tuple, Optional

# 主要城市经纬度（简化版，包含主要城市）
# 格式：城市代码: (经度, 纬度)
CITY_COORDINATES = {
    # 一线城市
    '1100': (116.4074, 39.9042),  # 北京
    '3100': (121.4737, 31.2304),  # 上海
    '4401': (113.2644, 23.1291),  # 广州
    '4403': (114.0579, 22.5431),  # 深圳
    
    # 新一线城市
    '1200': (117.2008, 39.0842),  # 天津
    '3201': (118.7969, 32.0603),  # 南京
    '3205': (120.5853, 31.2989),  # 苏州
    '3301': (120.1551, 30.2741),  # 杭州
    '3302': (121.5440, 29.8683),  # 宁波
    '3702': (120.3826, 36.0671),  # 青岛
    '4201': (114.3054, 30.5928),  # 武汉
    '4301': (112.9388, 28.2282),  # 长沙
    '5101': (104.0668, 30.5728),  # 成都
    '5000': (106.5516, 29.5630),  # 重庆
    
    # 其他重要城市（示例，可以根据需要扩展）
    '1301': (114.5149, 38.0428),  # 石家庄
    '2101': (123.4315, 41.8057),  # 沈阳
    '2102': (121.6147, 38.9140),  # 大连
    '2201': (125.3235, 43.8171),  # 长春
    '2301': (126.5358, 45.8022),  # 哈尔滨
    '3401': (117.2272, 31.8206),  # 合肥
    '3501': (119.2965, 26.0745),  # 福州
    '3502': (118.1108, 24.4798),  # 厦门
    '3601': (115.8921, 28.6765),  # 南昌
    '3701': (117.0009, 36.6758),  # 济南
    '4101': (113.6654, 34.7579),  # 郑州
    '4501': (108.3200, 22.8240),  # 南宁
    '5201': (106.6302, 26.6477),  # 贵阳
    '5301': (102.7123, 25.0406),  # 昆明
    '6101': (108.9398, 34.3416),  # 西安
    '6201': (103.8236, 36.0580),  # 兰州
}

# 城市GDP数据（2023年，单位：万亿元，简化版）
# 数据来源：公开数据，部分为估算
CITY_GDP = {
    '1100': 4.4,   # 北京
    '3100': 4.7,   # 上海
    '4401': 3.0,   # 广州
    '4403': 3.5,   # 深圳
    '1200': 1.6,   # 天津
    '3201': 1.7,   # 南京
    '3205': 2.4,   # 苏州
    '3301': 1.9,   # 杭州
    '3302': 1.6,   # 宁波
    '3702': 1.5,   # 青岛
    '4201': 1.9,   # 武汉
    '4301': 1.4,   # 长沙
    '5101': 2.1,   # 成都
    '5000': 3.0,   # 重庆
    # 其他城市使用相对规模估算
}

# 城市产业特点（简化分类）
# 1: 制造业发达, 2: 服务业发达, 3: 综合型
CITY_INDUSTRY_TYPE = {
    '1100': 3,  # 北京：综合型
    '3100': 3,  # 上海：综合型
    '4401': 3,  # 广州：综合型
    '4403': 1,  # 深圳：制造业发达
    '3205': 1,  # 苏州：制造业发达
    '3302': 1,  # 宁波：制造业发达
    '4406': 1,  # 佛山：制造业发达
    '3301': 2,  # 杭州：服务业发达（互联网）
    '5101': 2,  # 成都：服务业发达
    '4201': 2,  # 武汉：服务业发达
    # 默认：综合型
}

def get_city_coordinate(city_code: str) -> Optional[Tuple[float, float]]:
    """获取城市经纬度"""
    return CITY_COORDINATES.get(city_code)

def calculate_distance(city_code1: str, city_code2: str) -> Optional[float]:
    """
    计算两个城市间的距离（公里）
    使用Haversine公式计算球面距离
    """
    coord1 = get_city_coordinate(city_code1)
    coord2 = get_city_coordinate(city_code2)
    
    if coord1 is None or coord2 is None:
        # 如果没有坐标数据，返回None（将使用城市圈作为替代）
        return None
    
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    
    # 地球半径（公里）
    R = 6371.0
    
    # 转换为弧度
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    # Haversine公式
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    
    return distance

def get_city_gdp(city_code: str) -> float:
    """获取城市GDP（万亿元）"""
    # 如果有数据，返回；否则根据城市等级估算
    if city_code in CITY_GDP:
        return CITY_GDP[city_code]
    
    # 根据城市等级估算
    from config import TIER1_CITIES, TIER2_CITIES
    if city_code in TIER1_CITIES:
        return 2.5  # 一线城市平均
    elif city_code in TIER2_CITIES:
        return 1.0  # 新一线城市平均
    else:
        return 0.3  # 其他城市平均

def get_city_industry_type(city_code: str) -> int:
    """获取城市产业类型：1=制造业, 2=服务业, 3=综合型"""
    return CITY_INDUSTRY_TYPE.get(city_code, 3)  # 默认综合型

def calculate_industry_match(from_industry_type: int, to_industry_type: int, person_industry: str) -> float:
    """
    计算产业匹配度（使用config.py中的参数）
    
    Args:
        from_industry_type: 源城市产业类型（1=制造业, 2=服务业, 3=综合型）
        to_industry_type: 目标城市产业类型
        person_industry: 个人行业（Mfg/Service/Wht）
    
    Returns:
        float: 匹配度乘数（>1表示匹配，<1表示不匹配）
    """
    from config import INDUSTRY_MATCH_MULTIPLIER
    
    # 制造业人员
    if person_industry == 'Mfg':
        if to_industry_type == 1:  # 制造业城市
            return INDUSTRY_MATCH_MULTIPLIER['Mfg_to_Mfg']
        elif to_industry_type == 2:  # 服务业城市
            return INDUSTRY_MATCH_MULTIPLIER['Mfg_to_Service']
        else:  # 综合型
            return INDUSTRY_MATCH_MULTIPLIER['Mfg_to_Mixed']
    
    # 服务业人员
    elif person_industry == 'Service':
        if to_industry_type == 2:  # 服务业城市
            return INDUSTRY_MATCH_MULTIPLIER['Service_to_Service']
        elif to_industry_type == 1:  # 制造业城市
            return INDUSTRY_MATCH_MULTIPLIER['Service_to_Mfg']
        else:  # 综合型
            return INDUSTRY_MATCH_MULTIPLIER['Service_to_Mixed']
    
    # 白领人员
    elif person_industry == 'Wht':
        if to_industry_type == 3:  # 综合型
            return INDUSTRY_MATCH_MULTIPLIER['Wht_to_Mixed']
        elif to_industry_type == 2:  # 服务业城市
            return INDUSTRY_MATCH_MULTIPLIER['Wht_to_Service']
        else:  # 制造业城市
            return INDUSTRY_MATCH_MULTIPLIER['Wht_to_Mfg']
    
    return 1.0
