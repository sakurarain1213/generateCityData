# 人口迁徙数据生成与查询系统

## 📊 项目简介
基于经济学模型的大规模人口迁移数据合成系统，生成500万人口基数的城市间迁移预测数据。

## 🚀 快速开始

### 方法一：一键运行（推荐）
```bash
# 1. 生成完整数据（数据生成 + 数据库构建）
python run_optimized.py

# 2. 查询和分析数据  
python local_db/query_test.py
```

### 方法二：分步执行
```bash
# 步骤1: 生成CSV数据（约3-5分钟）
python synthesis/main_optimized.py

# 步骤2: 构建DuckDB数据库（约1-2分钟）
python local_db/optimized_data_generator.py

# 步骤3: 查询和测试【和生成jsonl】（几秒钟）
python local_db/query_test.py
```

## 📁 输出文件
运行完成后会生成：
- `output/migration_data.csv` - 完整迁移数据（~80MB，18万+行）
- `output/local_migration_data_optimized.db` - DuckDB数据库
- `output/migration_sample_optimized.csv` - 100行样本数据

## 📊 数据规模
- **总人口**: 1.8亿人
- **覆盖城市**: 339个中国地级市
- **人群类型**: 149种有效Type
- **数据记录**: 18万+行
- **时间维度**: 2024年全年（12个月）

## 🔍 查询功能
`local_db/query_test.py` 提供：
- 数据库总体统计信息
- 按城市/年月/人群类型条件查询
- Top20迁移目标城市分析
- DuckDB查询性能测试

## 🏗️ 系统架构
```
├── run_optimized.py          # 一键运行脚本
├── synthesis/                # 数据生成模块
│   ├── main_optimized.py    # 优化版数据生成
│   ├── config.py            # 参数配置
│   ├── type_generator.py    # 人群类型生成
│   ├── migration_model.py   # 迁移模型
│   └── population_distribution.py # 人口分布
└── local_db/                # 数据库模块
    ├── optimized_data_generator.py # 数据库构建
    └── query_test.py        # 查询测试工具
```

## ⚡ 性能特点
- **多进程并行**: 12核并行生成，速度提升10倍+
- **内存优化**: 分批处理，支持大规模数据
- **高效存储**: DuckDB列式存储，查询速度快
- **索引优化**: 自动创建查询索引

---

## 📋 技术文档

### Type人群分类维度
人口特征维度（Type）用于刻画人的结构性特征：

| 维度 | 名称 | 分类 | 说明 |
|------|------|------|------|
| D1 | 性别 | 男/女 | 劳动供给弹性差异 |
| D2 | 生命周期 | 5个年龄段 | 16-24(试错)、25-34(成家)、35-49(稳固)、50-60(回流)、60+(养老) |
| D3 | 学历水平 | 3个层次 | 初中及以下/高中专科/本科及以上 |
| D4 | 行业赛道 | 3个类型 | 蓝领制造/传统服务/现代专业服务 |
| D5 | 相对收入 | 5个分位 | Q1-Q5收入分位数 |
| D6 | 家庭状态 | 分离/团聚 | 回流阻尼系数 |

### 数据模式（Schema）
```sql
Year           INTEGER    -- 年份
Month          INTEGER    -- 月份  
Type_ID        VARCHAR    -- 人群类型ID（如：M_30_EduLo_Mfg_IncM_Unit）
From_City      VARCHAR    -- 出发城市（如：北京(1100)）
Total_Count    INTEGER    -- 该类型人口总数
Stay_Prob      REAL       -- 留守概率
To_Top1        VARCHAR    -- 第1目标城市
To_Top1_Prob   REAL       -- 第1目标概率
...            ...        -- Top2-20目标城市及概率
To_Other_Prob  REAL       -- 其他城市概率
```

### 依赖环境
- Python 3.8+
- pandas, numpy, duckdb, tqdm
- 内存建议：8GB+
- 存储空间：500MB+