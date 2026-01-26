import json
import random
import numpy as np

# ==============================================================================
# 1. 常量定义与城市列表
# ==============================================================================

# 省份对应身份证前两位代码 (GB/T 2260)
PROVINCE_PREFIX = {
    'Hebei': '13', 'Shanxi': '14', 'InnerMongolia': '15',
    'Liaoning': '21', 'Jilin': '22', 'Heilongjiang': '23',
    'Jiangsu': '32', 'Zhejiang': '33', 'Anhui': '34', 'Fujian': '35', 'Jiangxi': '36', 'Shandong': '37',
    'Henan': '41', 'Hubei': '42', 'Hunan': '43', 'Guangdong': '44', 'Guangxi': '45', 'Hainan': '46',
    'Sichuan': '51', 'Guizhou': '52', 'Yunnan': '53', 'Tibet': '54',
    'Shaanxi': '61', 'Gansu': '62', 'Qinghai': '63', 'Ningxia': '64', 'Xinjiang': '65'
}

# 省会城市坐标锚点 (用于生成省内其他城市坐标)
CAPITAL_COORDINATES = {
    '11': [116.40, 39.90], '12': [117.20, 39.08], '31': [121.47, 31.23], '50': [106.55, 29.56], # 直辖市
    '13': [114.51, 38.04], '14': [112.55, 37.87], '15': [111.77, 40.82],
    '21': [123.43, 41.80], '22': [125.32, 43.81], '23': [126.53, 45.80],
    '32': [118.79, 32.06], '33': [120.15, 30.27], '34': [117.23, 31.82], '35': [119.30, 26.08], '36': [115.89, 28.68], '37': [117.02, 36.65],
    '41': [113.62, 34.75], '42': [114.30, 30.59], '43': [112.93, 28.23], '44': [113.26, 23.13], '45': [108.33, 22.84], '46': [110.33, 20.02],
    '51': [104.06, 30.57], '52': [106.63, 26.65], '53': [102.71, 25.04], '54': [91.11, 29.97],
    '61': [108.95, 34.27], '62': [103.82, 36.06], '63': [101.78, 36.62], '64': [106.27, 38.47], '65': [87.62, 43.79]
}

# 城市分组数据 (省份, [城市名列表])
CITY_GROUPS = [
    ('Hebei', ["石家庄", "唐山", "秦皇岛", "邯郸", "邢台", "保定", "张家口", "承德", "沧州", "廊坊", "衡水"]),
    ('Shanxi', ["太原", "大同", "阳泉", "长治", "晋城", "朔州", "晋中", "运城", "忻州", "临汾", "吕梁"]),
    ('InnerMongolia', ["呼和浩特", "包头", "乌海", "赤峰", "通辽", "鄂尔多斯", "呼伦贝尔", "巴彦淖尔", "乌兰察布", "兴安盟", "锡林郭勒盟", "阿拉善盟"]),
    ('Liaoning', ["沈阳", "大连", "鞍山", "抚顺", "本溪", "丹东", "锦州", "营口", "阜新", "辽阳", "盘锦", "铁岭", "朝阳", "葫芦岛"]),
    ('Jilin', ["长春", "吉林", "四平", "辽源", "通化", "白山", "松原", "白城", "延边"]),
    ('Heilongjiang', ["哈尔滨", "齐齐哈尔", "鸡西", "鹤岗", "双鸭山", "大庆", "伊春", "佳木斯", "七台河", "牡丹江", "黑河", "绥化", "大兴安岭"]),
    ('Jiangsu', ["南京", "无锡", "徐州", "常州", "苏州", "南通", "连云港", "淮安", "盐城", "扬州", "镇江", "泰州", "宿迁"]),
    ('Zhejiang', ["杭州", "宁波", "温州", "嘉兴", "湖州", "绍兴", "金华", "衢州", "舟山", "台州", "丽水"]),
    ('Anhui', ["合肥", "芜湖", "蚌埠", "淮南", "马鞍山", "淮北", "铜陵", "安庆", "黄山", "滁州", "阜阳", "宿州", "六安", "亳州", "池州", "宣城"]),
    ('Fujian', ["福州", "厦门", "莆田", "三明", "泉州", "漳州", "南平", "龙岩", "宁德"]),
    ('Jiangxi', ["南昌", "景德镇", "萍乡", "九江", "新余", "鹰潭", "赣州", "吉安", "宜春", "抚州", "上饶"]),
    ('Shandong', ["济南", "青岛", "淄博", "枣庄", "东营", "烟台", "潍坊", "济宁", "泰安", "威海", "日照", "临沂", "德州", "聊城", "滨州", "菏泽"]),
    ('Henan', ["郑州", "开封", "洛阳", "平顶山", "安阳", "鹤壁", "新乡", "焦作", "濮阳", "许昌", "漯河", "三门峡", "南阳", "商丘", "信阳", "周口", "驻马店"]),
    ('Hubei', ["武汉", "黄石", "十堰", "宜昌", "襄阳", "鄂州", "荆门", "孝感", "荆州", "黄冈", "咸宁", "随州", "恩施"]),
    ('Hunan', ["长沙", "株洲", "湘潭", "衡阳", "邵阳", "岳阳", "常德", "张家界", "益阳", "郴州", "永州", "怀化", "娄底", "湘西"]),
    ('Guangdong', ["广州", "韶关", "深圳", "珠海", "汕头", "佛山", "江门", "湛江", "茂名", "肇庆", "惠州", "梅州", "汕尾", "河源", "阳江", "清远", "东莞", "中山", "潮州", "揭阳", "云浮"]),
    ('Guangxi', ["南宁", "柳州", "桂林", "梧州", "北海", "防城港", "钦州", "贵港", "玉林", "百色", "贺州", "河池", "来宾", "崇左"]),
    ('Hainan', ["海口", "三亚", "三沙", "儋州"]),
    ('Sichuan', ["成都", "自贡", "攀枝花", "泸州", "德阳", "绵阳", "广元", "遂宁", "内江", "乐山", "南充", "眉山", "宜宾", "广安", "达州", "雅安", "巴中", "资阳", "阿坝", "甘孜", "凉山"]),
    ('Guizhou', ["贵阳", "六盘水", "遵义", "安顺", "毕节", "铜仁", "黔西南", "黔东南", "黔南"]),
    ('Yunnan', ["昆明", "曲靖", "玉溪", "保山", "昭通", "丽江", "普洱", "临沧", "楚雄", "红河", "文山", "西双版纳", "大理", "德宏", "怒江", "迪庆"]),
    ('Tibet', ["拉萨", "日喀则", "昌都", "林芝", "山南", "那曲", "阿里"]),
    ('Shaanxi', ["西安", "铜川", "宝鸡", "咸阳", "渭南", "延安", "汉中", "榆林", "安康", "商洛"]),
    ('Gansu', ["兰州", "嘉峪关", "金昌", "白银", "天水", "武威", "张掖", "平凉", "酒泉", "庆阳", "定西", "陇南", "临夏", "甘南"]),
    ('Qinghai', ["西宁", "海东", "海北", "黄南", "海南", "果洛", "玉树", "海西"]),
    ('Ningxia', ["银川", "石嘴山", "吴忠", "固原", "中卫"]),
    ('Xinjiang', ["乌鲁木齐", "克拉玛依", "吐鲁番", "哈密", "昌吉", "博尔塔拉", "巴音郭楞", "阿克苏", "克孜勒苏", "喀什", "和田", "伊犁", "塔城", "阿勒泰"])
]

# 直辖市 (名称, ID前缀, 固定层级)
MUNICIPALITIES = [
    ("北京", "1100", 1),
    ("天津", "1200", 2),
    ("上海", "3100", 1),
    ("重庆", "5000", 2)
]

# 城市层级定义 (Tier 1-4)
TIER_1 = ["北京", "上海", "广州", "深圳"]
TIER_NEW_1 = ["成都", "重庆", "杭州", "武汉", "西安", "天津", "苏州", "南京", "郑州", "长沙", "东莞", "沈阳", "青岛", "佛山", "合肥"]
TIER_2 = ["宁波", "昆明", "福州", "无锡", "厦门", "济南", "大连", "哈尔滨", "温州", "石家庄", "泉州", "南宁", "长春", "南昌", "贵阳", "金华", "常州", "惠州", "嘉兴", "南通", "徐州", "太原", "珠海", "中山", "保定", "兰州", "台州", "绍兴", "烟台", "廊坊"]

# ==============================================================================
# 2. 辅助函数
# ==============================================================================

def get_tier(name):
    """根据城市名获取城市层级 (1:一线, 2:新一线, 3:二线, 4:其他)"""
    if name in TIER_1: return 1
    if name in TIER_NEW_1: return 2
    if name in TIER_2: return 3
    return 4

def generate_coords(province_prefix, is_capital=False, rng=None):
    """生成城市经纬度，非省会城市在省会附近添加随机偏移"""
    if rng is None: rng = random.Random()
    base = CAPITAL_COORDINATES.get(province_prefix, [105.0, 35.0])
    if is_capital:
        return base
    else:
        # 添加小范围随机偏移
        return [
            round(base[0] + rng.uniform(-1.5, 1.5), 2),
            round(base[1] + rng.uniform(-1.5, 1.5), 2)
        ]

def normalize_dict(d):
    """归一化字典的值，使其和为 1.0"""
    total = sum(d.values())
    if total == 0: return d
    return {k: round(v / total, 3) for k, v in d.items()}

def force_sum_1(d):
    """强制让字典的值和严格等于 1.0 (将误差加到最大值上)"""
    total = sum(d.values())
    diff = 1.0 - total
    max_key = max(d, key=d.get)
    d[max_key] = round(d[max_key] + diff, 4)
    return d

# ==============================================================================
# 3. 生成逻辑 (经济、人口、服务等)
# ==============================================================================

def generate_basic_info(tier, coords, rng):
    """生成基础信息：层级、坐标、面积"""
    # 一线/新一线城市通常建成区面积更大
    base_area = 2000 if tier > 2 else 5000
    area = base_area + rng.uniform(-500, 3000)
    return {
        "tier": tier,
        "coordinates": coords,
        "area_sqkm": round(area, 1)
    }

def generate_economy_and_jobs(tier, rng):
    """生成经济数据与行业分布 (4类行业)"""
    # 1. 基础经济参数
    if tier == 1:
        gdp_pc = rng.uniform(160000, 220000)
        unemployment = rng.uniform(0.03, 0.045)
        wage_multiplier = 1.0  # 基准工资倍率
    elif tier == 2:
        gdp_pc = rng.uniform(100000, 160000)
        unemployment = rng.uniform(0.035, 0.05)
        wage_multiplier = 0.75
    elif tier == 3:
        gdp_pc = rng.uniform(60000, 100000)
        unemployment = rng.uniform(0.04, 0.06)
        wage_multiplier = 0.55
    else:
        gdp_pc = rng.uniform(30000, 60000)
        unemployment = rng.uniform(0.04, 0.07)
        wage_multiplier = 0.40

    # 2. 行业结构分布 (权重分配逻辑)
    # 顺序: [农业, 制造业, 传统服务业, 现代服务业]
    if tier == 1:
        # 一线城市：现代服务业占比高，农业极少
        weights = [0.05, 0.20, 0.35, 0.40]
    elif tier == 2:
        # 新一线：制造业强，服务业均衡
        weights = [0.10, 0.30, 0.35, 0.25]
    elif tier == 3:
        # 二三线：传统服务业和制造业为主
        weights = [0.20, 0.35, 0.30, 0.15]
    else:
        # 四线及以下：农业占比相对较高
        weights = [0.30, 0.30, 0.30, 0.10]
    
    # 添加随机噪声并归一化
    weights = np.array(weights) * np.random.uniform(0.8, 1.2, 4)
    weights = weights / weights.sum()
    
    shares = {
        "agriculture": weights[0],
        "manufacturing": weights[1],
        "traditional_services": weights[2],
        "modern_services": weights[3]
    }
    shares = force_sum_1({k: round(v, 3) for k, v in shares.items()})

    # 3. 工资与空缺率生成
    # 以上海(Tier 1)为基准工资
    base_wages = {
        "agriculture": 6000,
        "manufacturing": 12000,
        "traditional_services": 7000,
        "modern_services": 18000
    }
    
    sectors = {}
    for sec_name, share in shares.items():
        # 工资 = 基准 * 城市倍率 * 随机波动
        avg_wage = base_wages[sec_name] * wage_multiplier * rng.uniform(0.9, 1.1)
        
        # 空缺率: 现代服务业通常更紧俏(空缺率低)，制造业常年缺人(空缺率高)
        if sec_name == "modern_services":
            base_vacancy = 0.04
        elif sec_name == "manufacturing":
            base_vacancy = 0.08
        else:
            base_vacancy = 0.06
            
        sectors[sec_name] = {
            "share": share,
            "avg_wage": int(avg_wage),
            "vacancy_rate": round(base_vacancy * rng.uniform(0.8, 1.5), 2)
        }

    economy = {
        "gdp_per_capita": round(gdp_pc, 1),
        "cpi_index": round(rng.uniform(100.0, 105.0), 1),
        "unemployment_rate": round(unemployment, 3),
        "industry_sectors": sectors
    }
    return economy

def generate_demographics(tier, rng):
    """生成人口统计特征"""
    # 年龄结构划分 (5类): 16-24, 25-34, 35-49, 50-60, 60+
    # 逻辑: 
    # Tier 1/2: 虹吸效应，年轻人(16-34)占比高
    # Tier 3/4: 人口流出，中老年(50+)占比高
    
    if tier <= 2:
        age_weights = {
            "16_24": 0.15, # 大学生/实习生/职场新人
            "25_34": 0.35, # 核心劳动力/奋斗期
            "35_49": 0.30, # 资深劳动力
            "50_60": 0.10, # 退休过渡期
            "60_plus": 0.10 # 老龄人口
        }
    else:
        age_weights = {
            "16_24": 0.10, # 年轻人流出求学/工作
            "25_34": 0.15, # 严重流失群体
            "35_49": 0.25, # 留守中年
            "50_60": 0.20, 
            "60_plus": 0.30 # 老龄化较重
        }
    
    # 添加噪声并归一化
    age_dist = {k: v * rng.uniform(0.9, 1.1) for k,v in age_weights.items()}
    age_dist = normalize_dict(age_dist)
    age_dist = force_sum_1(age_dist)

    return {
        "age_structure": age_dist,
        "sex_ratio": round(rng.uniform(101.0, 108.0), 1)
    }

def generate_living_cost(tier, rng, economy_data):
    """生成生活成本数据"""
    # 房价与城市层级强相关
    if tier == 1:
        base_price = 65000
    elif tier == 2:
        base_price = 25000
    elif tier == 3:
        base_price = 12000
    else:
        base_price = 6000
        
    housing_price = base_price * rng.uniform(0.8, 1.2)
    rent = housing_price / rng.uniform(18, 25) # 售租比
    
    # 生活成本指数
    cost_index = 1.5 if tier == 1 else (1.2 if tier == 2 else 1.0)
    
    return {
        "housing_price_avg": round(housing_price, 0),
        "rent_avg": round(rent, 0),
        "daily_cost_index": round(cost_index * rng.uniform(0.95, 1.05), 2)
    }

def generate_public_services(tier, rng):
    """生成公共服务评分"""
    # 分数范围 0.0 - 1.0
    if tier == 1:
        base_score = 0.9
        commute = 50
    elif tier == 2:
        base_score = 0.8
        commute = 40
    elif tier == 3:
        base_score = 0.7
        commute = 30
    else:
        base_score = 0.6
        commute = 20

    return {
        "medical_score": round(min(base_score * rng.uniform(0.9, 1.1), 0.99), 2),
        "education_score": round(min(base_score * rng.uniform(0.9, 1.1), 0.99), 2),
        "transport_convenience": round(min(base_score * rng.uniform(0.8, 1.2), 0.99), 2),
        "avg_commute_mins": int(commute * rng.uniform(0.8, 1.2))
    }

def generate_social_context(city_id, tier, rng):
    """生成社会人口与流动背景"""
    # 城市总人口范围设定
    if tier == 1:
        pop = rng.randint(15000000, 25000000)
    elif tier == 2:
        pop = rng.randint(8000000, 15000000)
    elif tier == 3:
        pop = rng.randint(3000000, 8000000)
    else:
        pop = rng.randint(500000, 3000000)

    # 外来人口来源分布 (模拟)
    # 随机选择3个非本省的省份作为主要来源地
    current_prov = city_id[:2]
    possible_origins = [k for k in CAPITAL_COORDINATES.keys() if k != current_prov]
    selected_origins = rng.sample(possible_origins, 3)
    
    # 设定外来人口比例: 一线城市高(30%)，四线城市低(5%)
    dist = {}
    total_share = 0.3 if tier == 1 else 0.05
    
    shares = np.random.dirichlet(np.ones(3), size=1)[0] * total_share
    for i, origin in enumerate(selected_origins):
        dist[f"{origin}00"] = round(float(shares[i]), 3)

    return {
        "population_total": pop,
        "migrant_stock_distribution": dist
    }

def generate_ground_truth(tier, rng):
    """生成用于校验的真实观测值 (模拟)"""
    # 迁徙指数
    if tier == 1:
        idx = rng.uniform(10.0, 15.0)
    elif tier == 2:
        idx = rng.uniform(6.0, 10.0)
    elif tier == 3:
        idx = rng.uniform(3.0, 6.0)
    else:
        idx = rng.uniform(0.5, 3.0)
        
    return {
        "inflow_index_last_year": round(idx, 2)
    }

# ==============================================================================
# 4. 主生成循环
# ==============================================================================

def generate_full_dataset(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    all_city_data = []

    # 1. 处理普通省辖城市
    for province_name, city_list in CITY_GROUPS:
        prefix = PROVINCE_PREFIX[province_name]
        for idx, city_name in enumerate(city_list):
            city_id = f"{prefix}{idx+1:02d}"
            tier = get_tier(city_name)
            is_capital = (idx == 0)
            
            # 生成各个模块数据
            coords = generate_coords(prefix, is_capital, rng)
            basic = generate_basic_info(tier, coords, rng)
            economy = generate_economy_and_jobs(tier, rng)
            demog = generate_demographics(tier, rng)
            cost = generate_living_cost(tier, rng, economy)
            services = generate_public_services(tier, rng)
            social = generate_social_context(city_id, tier, rng)
            gt = generate_ground_truth(tier, rng)

            # 组装节点
            city_node = {
                "city_id": city_id,
                "city_name": city_name,
                "basic_info": basic,
                "economy": economy,
                "demographics": demog,
                "living_cost": cost,
                "public_services": services,
                "social_context": social,
                "ground_truth_cache": gt
            }
            all_city_data.append(city_node)

    # 2. 处理直辖市 (北京/上海/天津/重庆)
    for name, cid, fixed_tier in MUNICIPALITIES:
        prefix = cid[:2]
        
        coords = generate_coords(prefix, is_capital=True, rng=rng)
        basic = generate_basic_info(fixed_tier, coords, rng)
        economy = generate_economy_and_jobs(fixed_tier, rng)
        demog = generate_demographics(fixed_tier, rng)
        cost = generate_living_cost(fixed_tier, rng, economy)
        services = generate_public_services(fixed_tier, rng)
        social = generate_social_context(cid, fixed_tier, rng)
        gt = generate_ground_truth(fixed_tier, rng)

        city_node = {
            "city_id": cid,
            "city_name": name,
            "basic_info": basic,
            "economy": economy,
            "demographics": demog,
            "living_cost": cost,
            "public_services": services,
            "social_context": social,
            "ground_truth_cache": gt
        }
        all_city_data.append(city_node)

    # 3. 写入文件
    output_file = "city_node.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for city in all_city_data:
            f.write(json.dumps(city, ensure_ascii=False) + '\n')

    print(f"成功生成 {len(all_city_data)} 个城市的数据，已保存至 {output_file}")

if __name__ == "__main__":
    generate_full_dataset()