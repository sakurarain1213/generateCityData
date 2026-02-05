# -*- coding: utf-8 -*-
"""
从数据库导出人口迁移统计数据
生成字段：Year, Month, Type_ID, Region, From_City, Total_Count, Outflow_Count, Outflow_Ratio
排序：按年份升序 → 城市代码升序 → Type_ID升序
"""

import os
import duckdb
import pandas as pd
from tqdm import tqdm

# 配置
DB_PATH = r'output/local_migration_data.db'
OUTPUT_PATH = r'output/migration_stats_with_to_cities.csv'
TOP_N_CITIES = 20
LIMIT_ROWS = 10000  # 导出前10000条

def export_migration_statistics():
    """
    从数据库导出人口和迁出统计
    """
    print("="*80)
    print("人口迁移统计数据导出工具")
    print("="*80)

    # 检查数据库
    if not os.path.exists(DB_PATH):
        print(f"\n错误: 数据库不存在: {DB_PATH}")
        return

    # 连接数据库
    print(f"\n连接数据库: {DB_PATH}")
    conn = duckdb.connect(DB_PATH)

    # 检查表是否存在
    table_check = conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'migration_data'"
    ).fetchone()[0]

    if table_check == 0:
        print("\n错误: migration_data 表不存在")
        conn.close()
        return

    # 统计总行数
    total_rows = conn.execute("SELECT COUNT(*) FROM migration_data").fetchone()[0]
    print(f"数据库总行数: {total_rows:,}")

    # 导出数据
    print(f"\n开始导出前 {LIMIT_ROWS:,} 条记录...")
    print(f"排序规则: 年份(ASC) → 城市代码(ASC) → Type_ID(ASC)")

    # 构建查询：计算每个Type的迁出总人口以及具体的To_City分布（Top20）
    # 迁出总人口 = Total_Count × (1 - Stay_Prob)
    top_columns = []
    select_columns = ["Year", "Month", "Type_ID", "Region", "From_City", "Total_Count", "Stay_Prob", "Outflow_Count", "Outflow_Ratio_Pct"]

    # 动态生成Top1到Top20的列
    for i in range(1, 21):
        top_columns.append(f"To_Top{i}, To_Top{i}_Prob")
        select_columns.append(f"To_Top{i}")
        select_columns.append(f"CAST(Total_Count * To_Top{i}_Prob AS BIGINT) as To_Top{i}_Count")

    query = f"""
        WITH base_data AS (
            SELECT
                Year,
                Month,
                Type_ID,
                Region,
                From_City,
                Total_Count,
                Stay_Prob,
                CAST(Total_Count * (1 - Stay_Prob) AS BIGINT) as Outflow_Count,
                ROUND((1 - Stay_Prob) * 100, 2) as Outflow_Ratio_Pct,
                {', '.join(top_columns)}
            FROM migration_data
            ORDER BY Year ASC, Region ASC, Type_ID ASC
            LIMIT {LIMIT_ROWS}
        )
        SELECT
            {', '.join(select_columns)}
        FROM base_data
    """

    print("执行查询...")
    df = conn.execute(query).df()

    print(f"\n查询完成，共 {len(df):,} 条记录")

    # 统计分析
    print("\n统计摘要:")
    print(f"  年份范围: {df['Year'].min()} - {df['Year'].max()}")
    print(f"  城市数量: {df['Region'].nunique()}")
    print(f"  Type数量: {df['Type_ID'].nunique()}")
    print(f"  总人口: {df['Total_Count'].sum():,}")
    print(f"  总迁出人口: {df['Outflow_Count'].sum():,}")
    print(f"  平均迁出比例: {df['Outflow_Ratio_Pct'].mean():.2f}%")
    print(f"  迁出比例范围: {df['Outflow_Ratio_Pct'].min():.2f}% - {df['Outflow_Ratio_Pct'].max():.2f}%")

    # 按年份统计
    print("\n按年份统计:")
    year_stats = df.groupby('Year').agg({
        'Total_Count': 'sum',
        'Outflow_Count': 'sum',
        'Outflow_Ratio_Pct': 'mean'
    }).round(2)
    year_stats.columns = ['总人口', '迁出人口', '平均迁出比例%']
    print(year_stats.to_string())

    # 保存CSV
    print(f"\n保存到CSV: {OUTPUT_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')

    print(f"[OK] 导出完成: {OUTPUT_PATH}")
    print(f"  文件大小: {os.path.getsize(OUTPUT_PATH) / 1024:.2f} KB")

    # 打印前10行预览
    print("\n前10行数据预览:")
    print(df.head(10).to_string())

    conn.close()
    print("\n" + "="*80)
    print("导出完成！")
    print("="*80)


if __name__ == '__main__':
    export_migration_statistics()
