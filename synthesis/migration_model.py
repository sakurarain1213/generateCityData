"""
迁移概率模型：基于Type特征计算迁移概率
包含：城市圈、时间因素、随机噪声
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from config import (
    DIMENSIONS, CITIES, TIER1_CITIES, TIER2_CITIES, TOP_N_CITIES,
    # 迁移概率模型参数
    AGE_MIGRATION_BASE, EDU_MIGRATION_MULTIPLIER, INDUSTRY_MIGRATION_MULTIPLIER,
    INCOME_MIGRATION_MULTIPLIER, FAMILY_MIGRATION_MULTIPLIER, GENDER_MIGRATION_MULTIPLIER,
    SEASONAL_FACTORS, MIGRATION_PROB_MIN, MIGRATION_PROB_MAX,
    NOISE_NORMAL_RANGE, NOISE_ANOMALY_RANGE, NOISE_ANOMALY_PROB,
    CITY_ATTRACTIVENESS_NOISE,
    # 城市吸引力参数
    CLUSTER_MULTIPLIER, DISTANCE_FACTORS, GDP_MULTIPLIER_HIGH, GDP_MULTIPLIER_MEDIUM,
    GDP_MULTIPLIER_LOW, TYPE_CITY_ADJUSTMENT,
    # 迁移目标分配参数
    TEMPERATURE_BASE, TEMPERATURE_NOISE, TOP_N_RATIO_BASE, TOP_N_RATIO_NOISE,
    TOP_N_RATIO_MIN, TOP_N_RATIO_MAX, TARGET_PROB_NOISE
)
from city_clusters import are_in_same_cluster, get_cluster_cities
from city_geo_econ import (
    calculate_distance, get_city_gdp, get_city_industry_type, 
    calculate_industry_match
)
from anomaly_events import get_anomaly_multiplier, get_historical_trend


class MigrationModel:
    """基于特征的迁移概率模型"""
    
    def __init__(self, random_seed: Optional[int] = None):
        # 初始化随机数生成器（用于噪声）
        self.rng = np.random.RandomState(random_seed)
        
        # 初始化各种权重参数（基于经济学假设）
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重（从config.py读取）"""
        # 从配置文件读取所有参数
        self.age_migration_base = AGE_MIGRATION_BASE
        self.edu_migration_multiplier = EDU_MIGRATION_MULTIPLIER
        self.industry_migration_multiplier = INDUSTRY_MIGRATION_MULTIPLIER
        self.income_migration_multiplier = INCOME_MIGRATION_MULTIPLIER
        self.family_migration_multiplier = FAMILY_MIGRATION_MULTIPLIER
        self.gender_migration_multiplier = GENDER_MIGRATION_MULTIPLIER
        self.seasonal_factors = SEASONAL_FACTORS
    
    def calculate_base_migration_prob(
        self, 
        type_dict: Dict[str, str], 
        month: Optional[int] = None,
        year: Optional[int] = None,
        from_city_code: Optional[str] = None,
        noise: bool = True
    ) -> float:
        """
        计算基础迁移概率（离开原城市的概率）
        
        Args:
            type_dict: Type字典
            month: 月份（1-12），用于时间因素调整
            year: 年份，用于历史趋势和异常事件
            from_city_code: 源城市代码，用于异常事件检测
            noise: 是否添加随机噪声
            
        Returns:
            float: 基础迁移概率（0-1之间）
        """
        # 从年龄获取基础概率
        base_prob = self.age_migration_base.get(type_dict['D2'], 0.15)
        
        # 应用各维度乘数
        multipliers = [
            self.gender_migration_multiplier.get(type_dict['D1'], 1.0),
            self.edu_migration_multiplier.get(type_dict['D3'], 1.0),
            self.industry_migration_multiplier.get(type_dict['D4'], 1.0),
            self.income_migration_multiplier.get(type_dict['D5'], 1.0),
            self.family_migration_multiplier.get(type_dict['D6'], 1.0)
        ]
        
        # 计算基础概率
        migration_prob = base_prob * np.prod(multipliers)
        
        # 时间因素：季节对迁移意愿的影响
        if month is not None:
            seasonal_factor = self.seasonal_factors.get(month, 1.0)
            migration_prob *= seasonal_factor
        
        # 历史趋势因素
        if year is not None and month is not None:
            trend_multiplier = get_historical_trend(year, month)
            migration_prob *= trend_multiplier
        
        # 异常事件因素
        if year is not None and month is not None and from_city_code:
            anomaly_multiplier = get_anomaly_multiplier(from_city_code, year, month)
            migration_prob *= anomaly_multiplier
        
        # 添加随机噪声（确保每行数据都有差异）
        if noise:
            # 噪声范围：可配置，偶尔有异常噪声
            if self.rng.random() < NOISE_ANOMALY_PROB:
                noise_factor = 1.0 + self.rng.uniform(-NOISE_ANOMALY_RANGE, NOISE_ANOMALY_RANGE)
            else:
                noise_factor = 1.0 + self.rng.uniform(-NOISE_NORMAL_RANGE, NOISE_NORMAL_RANGE)
            migration_prob *= noise_factor
        
        # 限制在合理范围内（可配置）
        migration_prob = np.clip(migration_prob, MIGRATION_PROB_MIN, MIGRATION_PROB_MAX)
        
        return migration_prob
    
    
    def calculate_city_attractiveness(
        self, 
        city_code: str, 
        from_city_code: Optional[str] = None,
        noise: bool = True
    ) -> float:
        """
        计算城市吸引力
        
        Args:
            city_code: 城市代码
            from_city_code: 源城市代码（用于城市圈调整）
            noise: 是否添加随机噪声
            
        Returns:
            float: 吸引力分数
        """
        if city_code in TIER1_CITIES:
            base_attractiveness = 1.0  # 一线城市：最高吸引力
        elif city_code in TIER2_CITIES:
            base_attractiveness = 0.7  # 新一线城市：较高吸引力
        else:
            # 其他城市：基于城市规模（简化处理）
            base_attractiveness = 0.3 + self.rng.uniform(0, 0.2)  # 0.3-0.5之间随机
        
        # 城市圈因素：同一城市群内的城市吸引力更高
        if from_city_code and are_in_same_cluster(from_city_code, city_code):
            base_attractiveness *= CLUSTER_MULTIPLIER
        
        # 添加随机噪声
        if noise:
            noise_factor = 1.0 + self.rng.uniform(-CITY_ATTRACTIVENESS_NOISE, CITY_ATTRACTIVENESS_NOISE)
            base_attractiveness *= noise_factor
        
        return base_attractiveness
    
    def calculate_migration_targets(
        self, 
        from_city_code: str, 
        type_dict: Dict[str, str],
        migration_prob: float,
        month: Optional[int] = None,
        year: Optional[int] = None,
        row_seed: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        计算迁移目标城市及其概率
        
        Args:
            from_city_code: 源城市代码
            type_dict: Type字典
            migration_prob: 总迁移概率
            month: 月份（用于时间因素）
            year: 年份（用于异常事件）
            row_seed: 行级随机种子（确保每行都有不同的噪声）
            
        Returns:
            List[Tuple[str, float]]: [(城市代码, 概率), ...]，按概率降序排列
        """
        # 如果提供了行级种子，创建临时随机数生成器
        if row_seed is not None:
            row_rng = np.random.RandomState(row_seed)
        else:
            row_rng = self.rng
        
        # 获取源城市信息
        from_industry_type = get_city_industry_type(from_city_code)
        from_gdp = get_city_gdp(from_city_code)
        
        # 计算所有城市的吸引力
        city_scores = []
        for city_code, city_name in CITIES:
            if city_code == from_city_code:
                continue  # 排除源城市
            
            # 使用行级随机数生成器计算吸引力（确保每行不同）
            attractiveness = self._calculate_city_attractiveness_with_rng(
                city_code, from_city_code, row_rng
            )
            
            # === 距离因素 ===
            distance = calculate_distance(from_city_code, city_code)
            if distance is not None:
                # 距离越近，吸引力越高（使用配置的距离衰减参数）
                distance_factor = 1.0
                for threshold, factor in [
                    (DISTANCE_FACTORS['very_close'][0], DISTANCE_FACTORS['very_close'][1]),
                    (DISTANCE_FACTORS['close'][0], DISTANCE_FACTORS['close'][1]),
                    (DISTANCE_FACTORS['medium'][0], DISTANCE_FACTORS['medium'][1]),
                    (DISTANCE_FACTORS['far'][0], DISTANCE_FACTORS['far'][1]),
                    (DISTANCE_FACTORS['very_far'][0], DISTANCE_FACTORS['very_far'][1])
                ]:
                    if distance < threshold:
                        distance_factor = factor
                        break
                attractiveness *= distance_factor
            
            # === 经济因素：GDP ===
            to_gdp = get_city_gdp(city_code)
            # GDP越高，吸引力越高（使用配置的参数）
            if to_gdp > from_gdp * 1.5:
                attractiveness *= GDP_MULTIPLIER_HIGH
            elif to_gdp > from_gdp * 1.2:
                attractiveness *= GDP_MULTIPLIER_MEDIUM
            elif to_gdp < from_gdp * 0.7:
                attractiveness *= GDP_MULTIPLIER_LOW
            
            # === 产业匹配度 ===
            to_industry_type = get_city_industry_type(city_code)
            industry_match = calculate_industry_match(
                from_industry_type, to_industry_type, type_dict['D4']
            )
            attractiveness *= industry_match
            
            # === Type特征调整（使用配置的参数） ===
            # 高学历更倾向于一线城市
            if type_dict['D3'] == 'EduHi' and city_code in TIER1_CITIES:
                attractiveness *= TYPE_CITY_ADJUSTMENT['EduHi_Tier1']
            elif type_dict['D3'] == 'EduLo' and city_code in TIER1_CITIES:
                attractiveness *= TYPE_CITY_ADJUSTMENT['EduLo_Tier1']
            
            # 高收入更倾向于一线城市
            if type_dict['D5'] in ['IncMH', 'IncH'] and city_code in TIER1_CITIES:
                attractiveness *= TYPE_CITY_ADJUSTMENT['IncHigh_Tier1']
            
            # 城市圈因素：同一城市群内的城市对某些Type更有吸引力
            if are_in_same_cluster(from_city_code, city_code):
                # 低学历、低收入更倾向于同城市群（文化相近、成本低）
                if type_dict['D3'] == 'EduLo':
                    attractiveness *= TYPE_CITY_ADJUSTMENT['EduLo_Cluster']
                if type_dict['D5'] in ['IncL', 'IncML']:
                    attractiveness *= TYPE_CITY_ADJUSTMENT['IncLow_Cluster']
                # 家庭团聚状态更倾向于同城市群（回流）
                if type_dict['D6'] == 'Unit':
                    attractiveness *= TYPE_CITY_ADJUSTMENT['Unit_Cluster']
            
            # === 异常事件因素（目标城市） ===
            if year is not None and month is not None:
                target_anomaly = get_anomaly_multiplier(city_code, year, month)
                # 如果目标城市有异常事件（如封城），吸引力降低
                if target_anomaly < 1.0:
                    attractiveness *= target_anomaly
            
            city_scores.append((city_code, city_name, attractiveness))
        
        # 按吸引力排序
        city_scores.sort(key=lambda x: x[2], reverse=True)
        
        # 计算概率分布（使用softmax-like方法，但增强Top城市的权重）
        scores = np.array([score[2] for score in city_scores])
        
        # 使用温度参数调整分布：温度越低，Top城市概率越高
        # 温度也加入一些随机性（基于行种子，使用配置的参数）
        temperature = TEMPERATURE_BASE * (1.0 + row_rng.uniform(-TEMPERATURE_NOISE, TEMPERATURE_NOISE))
        
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / exp_scores.sum()
        
        # Top N城市概率分配（使用配置的参数）
        top_n_ratio = TOP_N_RATIO_BASE + row_rng.uniform(-TOP_N_RATIO_NOISE, TOP_N_RATIO_NOISE)
        top_n_ratio = np.clip(top_n_ratio, TOP_N_RATIO_MIN, TOP_N_RATIO_MAX)
        
        top_n_prob = migration_prob * top_n_ratio
        other_prob = migration_prob * (1 - top_n_ratio)
        
        # 将Top N城市的概率归一化并分配
        top_n_scores = scores[:TOP_N_CITIES]
        top_n_exp = np.exp(top_n_scores / temperature)
        top_n_probs = top_n_exp / top_n_exp.sum()
        
        # 对概率添加一些随机扰动（确保每行都不同，使用配置的参数）
        prob_noise = 1.0 + row_rng.uniform(-TARGET_PROB_NOISE, TARGET_PROB_NOISE, size=len(top_n_probs))
        top_n_probs = top_n_probs * prob_noise
        top_n_probs = top_n_probs / top_n_probs.sum()  # 重新归一化
        
        top_n_migration_probs = top_n_probs * top_n_prob
        
        # 返回Top N城市
        result = []
        for i, (city_code, city_name, _) in enumerate(city_scores[:TOP_N_CITIES]):
            result.append((city_code, city_name, top_n_migration_probs[i]))
        
        # 添加Other
        if other_prob > 0:
            result.append(('Other', '其他', other_prob))
        
        return result
    
    def _calculate_city_attractiveness_with_rng(
        self, 
        city_code: str, 
        from_city_code: Optional[str] = None,
        rng: np.random.RandomState = None
    ) -> float:
        """使用指定的随机数生成器计算城市吸引力"""
        if rng is None:
            rng = self.rng
        
        if city_code in TIER1_CITIES:
            base_attractiveness = 1.0
        elif city_code in TIER2_CITIES:
            base_attractiveness = 0.7
        else:
            base_attractiveness = 0.3 + rng.uniform(0, 0.2)
        
        # 城市圈因素
        if from_city_code and are_in_same_cluster(from_city_code, city_code):
            base_attractiveness *= 1.3
        
        # 添加随机噪声（使用配置的参数）
        noise_factor = 1.0 + rng.uniform(-CITY_ATTRACTIVENESS_NOISE, CITY_ATTRACTIVENESS_NOISE)
        base_attractiveness *= noise_factor
        
        return base_attractiveness
    
    def calculate_stay_prob(self, migration_prob: float) -> float:
        """
        计算留在原城市的概率
        
        Args:
            migration_prob: 迁移概率
            
        Returns:
            float: 留在原城市的概率
        """
        return 1.0 - migration_prob


if __name__ == '__main__':
    # 测试
    model = MigrationModel()
    
    # 测试不同类型
    test_types = [
        {'D1': 'M', 'D2': '25-34', 'D3': 'EduHi', 'D4': 'Wht', 'D5': 'IncH', 'D6': 'Unit'},
        {'D1': 'F', 'D2': '16-24', 'D3': 'EduLo', 'D4': 'Mfg', 'D5': 'IncL', 'D6': 'Split'},
    ]
    
    for type_dict in test_types:
        print(f"\nType: {type_dict}")
        migration_prob = model.calculate_base_migration_prob(type_dict)
        stay_prob = model.calculate_stay_prob(migration_prob)
        print(f"迁移概率: {migration_prob:.3f}, 留下概率: {stay_prob:.3f}")
        
        targets = model.calculate_migration_targets('3205', type_dict, migration_prob)
        print(f"Top 5 迁移目标:")
        for city_code, city_name, prob in targets[:5]:
            print(f"  {city_name}({city_code}): {prob:.4f}")
