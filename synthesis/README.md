# 人口迁徙预测数据合成系统

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
python main.py
```

### 输出文件

默认输出到 `output/` 目录：
- `output/migration_data.csv` - CSV格式数据
- `output/migration_data.xlsx` - Excel格式数据（如果数据量<100万行）

**输出路径可在 `config.py` 中配置**：
```python
OUTPUT_DIR = 'output'  # 输出目录（相对于项目根目录）
OUTPUT_FILENAME = 'migration_data.csv'  # 输出文件名
```

---

## 参数配置

**所有超参数都在 `config.py` 中统一管理**，包括：

### 基础配置

#### 时间范围
```python
OUTPUT_YEARS = [2024]        # 输出年份列表
OUTPUT_MONTHS = list(range(1, 13))  # 输出月份（1-12月）
```

#### 城市列表
```python
CITIES = [('1100', '北京'), ('1200', '天津'), ...]  # 城市代码和名称列表
```

#### 长尾控制
```python
MIN_TYPE_COUNT = 10  # 最小Type样本数阈值
```

#### 迁移目标数量
```python
TOP_N_CITIES = 20  # Top N迁移目标城市数量
```

#### 输出文件路径
```python
OUTPUT_DIR = 'output'  # 输出目录（相对于项目根目录）
OUTPUT_FILENAME = 'migration_data.csv'  # 输出文件名
```

### 人口分布参数

#### 人口基数
```python
TOTAL_POPULATION_BASE = 100_000_000  # 总人口基数（单位：人）
```

#### 城市规模权重
```python
CITY_POPULATION_WEIGHTS = {
    '1100': 1.0,  # 北京（一线城市）
    '3100': 1.0,  # 上海（一线城市）
    # ... 更多城市
}
```

#### 幂律分布参数
```python
POWER_LAW_ALPHA = 1.8  # 幂律分布指数（越大，长尾越明显）
```

#### 城市人口噪声
```python
CITY_POP_NOISE_STABLE = 0.15  # 第一层噪声：±15%
CITY_POP_NOISE_TIME = 0.05    # 第二层噪声：±5%（基于时间）
CITY_POP_NOISE_EXTRA = 0.03   # 第三层噪声：±3%
```

### 迁移概率模型参数

#### 个人特征因素
```python
# 年龄迁移基础概率
AGE_MIGRATION_BASE = {
    '16-24': 0.35,  # 试错期：高迁移
    '25-34': 0.25,  # 成家期：中等迁移
    # ...
}

# 各维度乘数
EDU_MIGRATION_MULTIPLIER = {'EduLo': 0.8, 'EduMid': 1.0, 'EduHi': 1.3}
INDUSTRY_MIGRATION_MULTIPLIER = {'Mfg': 1.4, 'Service': 1.2, 'Wht': 0.9}
INCOME_MIGRATION_MULTIPLIER = {'IncL': 1.3, ..., 'IncH': 0.7}
FAMILY_MIGRATION_MULTIPLIER = {'Split': 1.2, 'Unit': 0.8}
GENDER_MIGRATION_MULTIPLIER = {'M': 1.05, 'F': 0.95}
```

#### 时间因素
```python
# 季节因子（不同月份对迁移意愿的影响）
SEASONAL_FACTORS = {
    1: 0.7,   # 1月：春节前，迁移意愿低
    2: 0.6,   # 2月：春节期间，迁移意愿最低
    3: 1.2,   # 3月：春季求职旺季，迁移意愿高
    # ...
}
```

#### 随机噪声
```python
NOISE_NORMAL_RANGE = 0.05      # 正常噪声：±5%
NOISE_ANOMALY_RANGE = 0.15     # 异常噪声：±15%
NOISE_ANOMALY_PROB = 0.05      # 异常噪声出现概率：5%
MIGRATION_PROB_MIN = 0.05      # 最小迁移概率
MIGRATION_PROB_MAX = 0.50      # 最大迁移概率
```

### 城市吸引力模型参数

#### 地理因素
```python
CLUSTER_MULTIPLIER = 1.3  # 同一城市群内的城市互相迁移概率提升30%

# 距离衰减参数
DISTANCE_FACTORS = {
    'very_close': (100, 1.3),    # <100km: ×1.3
    'close': (300, 1.2),         # <300km: ×1.2
    'medium': (500, 1.1),        # <500km: ×1.1
    'far': (1000, 1.0),          # <1000km: ×1.0
    'very_far': (float('inf'), 0.9)  # >=1000km: ×0.9
}
```

#### 经济因素
```python
GDP_MULTIPLIER_HIGH = 1.15   # 目标城市GDP > 源城市×1.5
GDP_MULTIPLIER_MEDIUM = 1.1  # 目标城市GDP > 源城市×1.2
GDP_MULTIPLIER_LOW = 0.95    # 目标城市GDP < 源城市×0.7

# 产业匹配度乘数
INDUSTRY_MATCH_MULTIPLIER = {
    'Mfg_to_Mfg': 1.3,        # 制造业人员 → 制造业城市
    'Service_to_Service': 1.2, # 服务业人员 → 服务业城市
    # ...
}
```

#### Type特征调整
```python
TYPE_CITY_ADJUSTMENT = {
    'EduHi_Tier1': 1.3,        # 高学历 + 一线城市
    'EduLo_Tier1': 0.8,        # 低学历 + 一线城市（落户难）
    'IncHigh_Tier1': 1.2,      # 高收入 + 一线城市
    'EduLo_Cluster': 1.2,      # 低学历 + 同城市群
    # ...
}
```

### 迁移目标分配参数

```python
TEMPERATURE_BASE = 0.3         # Softmax温度参数（越低，Top城市概率越高）
TEMPERATURE_NOISE = 0.1        # 温度随机变化：±10%
TOP_N_RATIO_BASE = 0.8         # Top N城市分配80%的迁移概率
TOP_N_RATIO_NOISE = 0.05       # 比例随机变化：±5%
TOP_N_RATIO_MIN = 0.7          # 最小比例
TOP_N_RATIO_MAX = 0.9          # 最大比例
TARGET_PROB_NOISE = 0.1        # 迁移目标概率噪声：±10%
CITY_ATTRACTIVENESS_NOISE = 0.08  # 城市吸引力噪声：±8%
```

---

## 输出文件格式

### 字段说明

| 字段名 | 类型 | 说明 |
|--------|------|------|
| Year | int | 年份 |
| Month | int | 月份（1-12） |
| Type_ID | str | Type标识，格式：`M_25_EduLo_Mfg_IncL_Split`<br>含义：性别_年龄_学历_行业_收入_家庭状态 |
| From_City | str | 源城市，格式：`城市名(城市代码)` |
| Total_Count | int | 该Type在该源城市的总人口数<br>**注意**：不同城市的同一Type人口数不同，考虑城市规模和Type特征匹配度 |
| Stay_Prob | float | 留在原城市的概率（0-1） |
| To_Top1 | str | 迁移到第1热门城市的名称 |
| To_Top1_Prob | float | 迁移到第1热门城市的概率 |
| To_Top2, To_Top2_Prob | ... | 迁移到第2热门城市 |
| ... | ... | ... |
| To_Top20, To_Top20_Prob | ... | 迁移到第20热门城市 |
| Other | float | 迁移到其他所有城市的概率总和 |

### 概率约束

**所有概率总和 = 1.0**

```
Stay_Prob + To_Top1_Prob + ... + To_Top20_Prob + Other = 1.0
```

### Type_ID格式说明

Type_ID由6个维度组成，用下划线（`_`）分隔，格式为：
```
{性别}_{年龄代表值}_{学历}_{行业}_{收入}_{家庭状态}
```

#### 维度说明

| 位置 | 维度 | 可能值 | 含义 |
|------|------|--------|------|
| 1 | 性别 | `M`, `F` | M=男, F=女 |
| 2 | 年龄代表值 | `25`, `30`, `40`, `55`, `65` | 对应年龄段：16-24, 25-34, 35-49, 50-60, 60+ |
| 3 | 学历 | `EduLo`, `EduMid`, `EduHi` | 初中及以下, 高中或专科, 本科及以上 |
| 4 | 行业 | `Mfg`, `Service`, `Wht` | 蓝领制造, 传统服务, 现代专业服务 |
| 5 | 收入 | `IncL`, `IncML`, `IncM`, `IncMH`, `IncH` | Q1-Q5分位（低到高） |
| 6 | 家庭状态 | `Split`, `Unit` | 分离, 团聚 |

#### 示例

**示例1**：`M_25_EduLo_Mfg_IncL_Split`
- 男性（M）
- 25岁代表值（对应16-24年龄段）
- 初中及以下学历（EduLo）
- 蓝领制造行业（Mfg）
- 低收入（IncL，Q1分位）
- 家庭分离（Split）

**示例2**：`F_30_EduMid_Service_IncM_Unit`
- 女性（F）
- 30岁代表值（对应25-34年龄段）
- 高中或专科学历（EduMid）
- 传统服务行业（Service）
- 中等收入（IncM，Q3分位）
- 家庭团聚（Unit）

**示例3**：`F_65_EduHi_Wht_IncH_Unit`
- 女性（F）
- 65岁代表值（对应60+年龄段）
- 本科及以上学历（EduHi）
- 现代专业服务行业（Wht）
- 高收入（IncH，Q5分位）
- 家庭团聚（Unit）

#### 如何解析Type_ID

**Python解析代码**：

```python
def parse_type_id(type_id: str) -> dict:
    """
    解析Type_ID字符串，返回维度字典
    
    Args:
        type_id: Type_ID字符串，如 "M_25_EduLo_Mfg_IncL_Split"
        
    Returns:
        dict: 维度字典，如 {'D1': 'M', 'D2': '16-24', ...}
    """
    parts = type_id.split('_')
    if len(parts) != 6:
        raise ValueError(f"Type_ID格式错误: {type_id}")
    
    # 年龄映射（反向）
    age_reverse_map = {
        '25': '16-24',
        '30': '25-34',
        '40': '35-49',
        '55': '50-60',
        '65': '60+'
    }
    
    return {
        'D1': parts[0],  # 性别
        'D2': age_reverse_map.get(parts[1], parts[1]),  # 生命周期
        'D3': parts[2],  # 学历
        'D4': parts[3],  # 行业
        'D5': parts[4],  # 收入
        'D6': parts[5]   # 家庭状态
    }

# 使用示例
type_id = "M_25_EduLo_Mfg_IncL_Split"
parsed = parse_type_id(type_id)
print(parsed)
# 输出: {'D1': 'M', 'D2': '16-24', 'D3': 'EduLo', 'D4': 'Mfg', 'D5': 'IncL', 'D6': 'Split'}
```

**Excel/Pandas解析**：

```python
import pandas as pd

# 读取CSV
df = pd.read_csv('output/migration_data.csv')

# 解析Type_ID
df[['Gender', 'Age_Rep', 'Education', 'Industry', 'Income', 'Family']] = \
    df['Type_ID'].str.split('_', expand=True)

# 年龄映射
age_map = {'25': '16-24', '30': '25-34', '40': '35-49', '55': '50-60', '65': '60+'}
df['Age_Range'] = df['Age_Rep'].map(age_map)
```

**SQL解析**（如果导入数据库）：

```sql
-- 使用字符串函数拆分
SELECT 
    Type_ID,
    SPLIT_PART(Type_ID, '_', 1) AS Gender,
    SPLIT_PART(Type_ID, '_', 2) AS Age_Rep,
    SPLIT_PART(Type_ID, '_', 3) AS Education,
    SPLIT_PART(Type_ID, '_', 4) AS Industry,
    SPLIT_PART(Type_ID, '_', 5) AS Income,
    SPLIT_PART(Type_ID, '_', 6) AS Family
FROM migration_data;
```

---

## 数据量估算

```
总行数 = Type数量 × 城市数量 × 年份数量 × 月份数量
```

**默认配置**：
- Type数量：约582（有效Type，经过长尾过滤）
- 城市数量：约300+（所有地级市）
- 年份数量：1（2024年）
- 月份数量：12（1-12月）

**预计数据量**：582 × 300 × 1 × 12 ≈ **210万行**

**注意**：数据量较大，生成时间可能需要几分钟到十几分钟。

---

## 实现方法

### 1. Type维度设计

基于7个维度定义"特征人"（Type）：

| 维度 | 名称 | 类别数 | 说明 |
|------|------|--------|------|
| D1 | 性别 | 2 | 男/女 |
| D2 | 生命周期 | 5 | 16-24, 25-34, 35-49, 50-60, 60+ |
| D3 | 学历/技能 | 3 | 初中及以下/高中或专科/本科及以上 |
| D4 | 行业赛道 | 3 | 蓝领制造/传统服务/现代专业服务 |
| D5 | 相对收入 | 5 | Q1-Q5分位 |
| D6 | 家庭状态 | 2 | 分离/团聚 |
| D7 | 文化地缘 | 300+ | 城市级别（通过From_City体现） |

理论Type总数：2 × 5 × 3 × 3 × 5 × 2 = **900种**

### 2. 迁移概率模型

基于经济学假设设计迁移概率函数，综合考虑以下因素。**所有参数都在 `config.py` 中配置**。

#### 个人特征因素（代码位置：`migration_model.py`）
- **年龄**：`AGE_MIGRATION_BASE` - 年轻人（16-34）迁移倾向高，老年人（50+）迁移倾向低
- **学历**：`EDU_MIGRATION_MULTIPLIER` - 高学历迁移倾向高，但低学历对一线城市吸引力较低（落户难）
- **行业**：`INDUSTRY_MIGRATION_MULTIPLIER` - 蓝领制造和传统服务迁移倾向高，现代专业服务相对稳定
- **收入**：`INCOME_MIGRATION_MULTIPLIER` - 低收入迁移倾向高（寻求机会），高收入迁移倾向低（已稳定）
- **家庭**：`FAMILY_MIGRATION_MULTIPLIER` - 分离状态迁移倾向高，团聚状态迁移倾向低（回流阻尼）
- **性别**：`GENDER_MIGRATION_MULTIPLIER` - 男性略高于女性

#### 地理因素（代码位置：`migration_model.py` + `city_clusters.py`）
- **城市圈**：`CLUSTER_MULTIPLIER = 1.3` - 同一城市群内的城市互相迁移概率提升30%
- **距离**：`DISTANCE_FACTORS` - 距离越近，迁移概率越高（100km内×1.3，1000km以上×0.9）

#### 经济因素（代码位置：`migration_model.py` + `city_geo_econ.py`）
- **GDP**：`GDP_MULTIPLIER_HIGH/MEDIUM/LOW` - 目标城市GDP明显更高时，吸引力提升
- **产业匹配**：`INDUSTRY_MATCH_MULTIPLIER` - 制造业人员更倾向于制造业城市（×1.3），服务业人员更倾向于服务业城市（×1.2）

#### 时间因素（代码位置：`migration_model.py` + `anomaly_events.py`）
- **季节**：`SEASONAL_FACTORS` - 春季（3-5月）迁移意愿高，春节（1-2月）迁移意愿低
- **历史趋势**：`anomaly_events.py` 中的 `get_historical_trend()` - 反映历史迁移趋势（如2020-2022年疫情影响）

#### 异常事件（代码位置：`anomaly_events.py`）
- **疫情**：封城期间迁移概率大幅降低（×0.3-0.4）
- **自然灾害**：灾后重建期间迁移概率增加（×1.5）
- **其他事件**：可在 `anomaly_events.py` 的 `ANOMALY_EVENTS` 字典中定义

#### 随机噪声（代码位置：`migration_model.py`）
- 确保每行数据都有差异，避免完全重复
- `NOISE_NORMAL_RANGE = 0.05` - 正常噪声：±5%
- `NOISE_ANOMALY_RANGE = 0.15` - 异常噪声：±15%
- `NOISE_ANOMALY_PROB = 0.05` - 异常噪声出现概率：5%

### 3. 城市吸引力模型

- **一线城市**（北京、上海、广州、深圳）：最高吸引力
- **新一线城市**：较高吸引力
- **其他城市**：基于城市规模、GDP、产业类型分配吸引力

### 4. 人口分布计算

#### 4.1 全局Type人口数
- 使用幂律分布生成每个Type的全局人口数
- 考虑Type特征：中等年龄、中等学历、中等收入更常见
- 只保留样本数≥10的Type（可配置）

#### 4.2 城市×Type人口数
每个城市×Type组合的人口数计算方式：

1. **基础分配**：基于城市规模权重
   - 一线城市（北京、上海等）：权重1.0
   - 新一线城市（天津、南京等）：权重0.3-0.6
   - 其他城市：权重0.05-0.2

2. **Type匹配度调整**：
   - 高学历Type在一线城市更常见（×1.3）
   - 蓝领制造Type在制造业城市更常见（×1.3）
   - 高收入Type在一线城市更常见（×1.2）

3. **多层随机噪声**：
   - 第一层：基于Type×城市的稳定噪声（±15%）- 确保不同城市有差异
   - 第二层：基于时间的动态噪声（±5%）- 确保不同月份有差异
   - 第三层：额外随机扰动（±3%）- 进一步确保每行数据都不同，确保数据多样性

4. **人口基数**：可在`config.py`中配置`TOTAL_POPULATION_BASE`

**示例**（基于测试数据）：
- 某Type全局人口：85,017,652人
- 北京（一线城市，权重1.0）：约290万人
- 上海（一线城市，权重1.0）：约290万人
- 天津（新一线，权重0.5）：约200万人
- 石家庄（其他城市，权重约0.1）：约53万人

**说明**：
- ✅ 不同城市的同一Type人口数不同，符合真实分布
- ✅ 同一Type×城市，不同月份的人口数有细微差异（±5%），避免完全重复
- ✅ 所有行的Total_Count都不同（测试验证：84个样本，0%重复率）
- ✅ 一线城市人口明显多于其他城市
- ✅ 可通过`config.py`中的`TOTAL_POPULATION_BASE`调整总人口基数
- ✅ 可通过`CITY_POPULATION_WEIGHTS`调整各城市的人口权重

---

## 文件结构

```
synthesis/
├── main.py                    # 主程序（入口）
├── config.py                  # 配置文件（时间、城市、参数）
├── type_generator.py          # Type生成器
├── migration_model.py         # 迁移概率模型（核心算法）
├── population_distribution.py # 人口分布生成器
├── city_clusters.py           # 城市圈定义
├── city_geo_econ.py           # 城市地理和经济数据
├── anomaly_events.py          # 异常事件和历史趋势
├── requirements.txt           # 依赖包列表
└── README.md                  # 本文档
```

---

## 依赖包

- `pandas >= 1.5.0` - 数据处理
- `numpy >= 1.23.0` - 数值计算
- `tqdm >= 4.64.0` - 进度条
- `openpyxl >= 3.0.0` - Excel文件支持

---

## 注意事项

1. **数据量**：完整数据量约210万行，生成时间较长，建议先用小规模测试
2. **内存占用**：生成完整数据时，内存占用可能较大（建议8GB+）
3. **Excel限制**：如果数据量超过100万行，不会生成Excel文件（只生成CSV）
4. **参数调整**：迁移概率模型参数可在 `migration_model.py` 中调整
5. **城市数据**：如需更多城市的经纬度和GDP数据，可在 `city_geo_econ.py` 中扩展

---

## 参数配置

**所有超参数都在 `config.py` 中统一管理**，包括：

- 迁移概率模型参数（年龄、学历、行业、收入、家庭、性别等）
- 时间因素参数（季节因子）
- 地理因素参数（城市圈、距离衰减）
- 经济因素参数（GDP、产业匹配度）
- 随机噪声参数
- 人口分布参数
- 迁移目标分配参数

**详细参数说明请查看**：`参数配置说明.md`

### 快速调整示例

#### 提高整体迁移率
```python
# 在 config.py 中
AGE_MIGRATION_BASE = {
    '16-24': 0.40,  # 从0.35提高到0.40
    '25-34': 0.30,  # 从0.25提高到0.30
    # ...
}
```

#### 增强距离因素
```python
# 在 config.py 中
DISTANCE_FACTORS = {
    'very_close': (100, 1.5),  # 从1.3提高到1.5
    # ...
}
```

#### 调整人口基数
```python
# 在 config.py 中
TOTAL_POPULATION_BASE = 150_000_000  # 从1亿提高到1.5亿
```

## 扩展说明

### 添加新异常事件

在 `anomaly_events.py` 的 `ANOMALY_EVENTS` 字典中添加：

```python
'event_name': {
    'start_date': (年, 月),
    'end_date': (年, 月),
    'affected_cities': [城市代码列表],  # 空列表表示全国
    'migration_multiplier': 乘数,  # <1降低，>1增加
    'description': '描述'
}
```

### 调整迁移概率参数

**直接在 `config.py` 中修改**：
- `AGE_MIGRATION_BASE` - 年龄迁移基础概率
- `EDU_MIGRATION_MULTIPLIER` - 学历迁移倾向
- `INDUSTRY_MIGRATION_MULTIPLIER` - 行业迁移倾向
- `INCOME_MIGRATION_MULTIPLIER` - 收入迁移倾向
- `FAMILY_MIGRATION_MULTIPLIER` - 家庭状态迁移倾向
- `GENDER_MIGRATION_MULTIPLIER` - 性别迁移倾向
- `SEASONAL_FACTORS` - 季节因子

### 扩展城市数据

在 `city_geo_econ.py` 中添加更多城市的经纬度和GDP数据：
- `CITY_COORDINATES` - 城市经纬度
- `CITY_GDP` - 城市GDP数据
- `CITY_INDUSTRY_TYPE` - 城市产业类型

---

## 技术支持

如有问题或建议，请查看代码注释或联系开发团队。
