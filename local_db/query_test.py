"""
DuckDB 极简查询演示 & 特征提取工具
包含：
1. 总体统计查询
2. 单个示例查询
3. [新增] Type_ID 特征解析并导出为 JSONL
"""

import duckdb
import os
import time
import pandas as pd
import json

# 数据库路径 (请确保此前已生成 .db 文件)
OUTPUT_DIR = 'output'
DB_FILE = os.path.join(OUTPUT_DIR, 'local_migration_data.db')
JSONL_FILE = os.path.join(OUTPUT_DIR, 'type_features.jsonl')

def query_total_stats(conn):
    """查询数据库总体统计信息"""
    print("\n=== 📊 数据库总体概览 ===")
    start_time = time.time()
    
    # 聚合查询：总行数和总人口
    query = "SELECT COUNT(*) as total_rows, SUM(Total_Count) as total_pop FROM migration_data"
    result = conn.execute(query).fetchone()
    
    elapsed = time.time() - start_time
    
    print(f"总行数:   {result[0]:,}")
    print(f"总人口数: {result[1]:,}")
    print(f"查询耗时: {elapsed:.4f} 秒")

def query_sample_city(conn, city_name="北京", year=2024, month=3, limit=500):
    """示例查询：获取指定城市指定年月的不同Type数据"""
    print(f"\n=== 🔍 示例查询: {year}年{month}月来源城市包含 '{city_name}' (前{limit}条) ===")
    start_time = time.time()
    
    # 参数化查询，显示Type、总数、目标城市和概率
    sql = """
    SELECT Year, Month, Type_ID, From_City, Total_Count, Stay_Prob,
           To_Top1, To_Top1_Prob, To_Top2, To_Top2_Prob, To_Top3, To_Top3_Prob
    FROM migration_data 
    WHERE From_City LIKE ? AND Year = ? AND Month = ?
    ORDER BY Total_Count DESC
    LIMIT ?
    """
    
    # DuckDB 可以直接返回 Pandas DataFrame，打印非常美观
    df = conn.execute(sql, [f'%{city_name}%', year, month, limit]).df()
    
    elapsed = time.time() - start_time
    
    if df.empty:
        print("未找到匹配数据。")
    else:
        print(f"找到 {len(df)} 条记录:")
        print(f"总人口数: {df['Total_Count'].sum():,}")
        print("-" * 120)
        # to_string(index=False) 隐藏 pandas 的索引列，使输出更干净
        print(df.to_string(index=False))
    
    print(f"\n查询耗时: {elapsed:.4f} 秒")

def extract_type_features(conn):
    """
    [新增功能] 提取所有唯一的 Type_ID，解析为 6 个维度，并保存为 JSONL
    """
    print(f"\n=== 🧬 正在提取 Type_ID 特征到 {JSONL_FILE} ===")
    start_time = time.time()

    # 1. 获取所有唯一的 Type_ID，按字母升序排列
    # 使用 DISTINCT 确保唯一性
    types_list = conn.execute("SELECT DISTINCT Type_ID FROM migration_data ORDER BY Type_ID ASC").fetchall()
    
    print(f"发现 {len(types_list)} 个唯一的 Type ID。开始解析...")

    # 维度定义（仅作参考，用于代码逻辑对照）
    # D1: gender (M/F)
    # D2: lifecycle (16-24/25-34...)
    # D3: education (EduLo/EduMid...)
    # D4: industry (Mfg/Service...)
    # D5: income (IncL/IncM...)
    # D6: family_status (Split/Unit)

    count = 0
    with open(JSONL_FILE, 'w', encoding='utf-8') as f:
        for t in types_list:
            type_id = t[0]
            parts = type_id.split('_')

            # 确保切分出 6 个部分，防止数据异常导致报错
            if len(parts) == 6:
                feature_dict = {
                    "type": type_id,          # 原始 ID
                    "gender": parts[0],          # D1: 性别
                    "age": parts[1],       # D2: 生命周期 (年龄段)
                    "edu": parts[2],       # D3: 学历
                    "job": parts[3],        # D4: 行业赛道
                    "income": parts[4],          # D5: 相对收入
                    "family": parts[5]    # D6: 家庭状态
                }
                
                # 写入 JSONL (一行一个 JSON 对象)
                f.write(json.dumps(feature_dict, ensure_ascii=False) + '\n')
                count += 1
            else:
                print(f"[WARN] 跳过格式异常的 ID: {type_id}")

    elapsed = time.time() - start_time
    print(f"✅ 成功导出 {count} 条特征记录。")
    print(f"耗时: {elapsed:.4f} 秒")

if __name__ == "__main__":
    if not os.path.exists(DB_FILE):
        print(f"错误: 数据库文件 {DB_FILE} 不存在，请先运行生成脚本。")
    else:
        try:
            # 建立连接 (read_only=True 更安全且支持并发读取)
            with duckdb.connect(DB_FILE, read_only=True) as conn:
                # 1. 查询总数
                query_total_stats(conn)
                
                # 2. 执行一个示例查询
                query_sample_city(conn, "宁波")

                # 3. [新增] 提取并解析 Type_ID
                extract_type_features(conn)
                
        except Exception as e:
            print(f"发生错误: {e}")