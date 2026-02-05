# -*- coding: utf-8 -*-
"""
人口约束验证工具

验证数据库中每年每城市的 Total_Count 总和是否严格等于 CSV 中的常住人口约束
"""

import os
import sys
import duckdb
import pandas as pd

# 配置
OUTPUT_DIR = r"C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN"
DB_FILENAME = 'local_migration_data_full.db'
CSV_PATH = os.path.join(OUTPUT_DIR, '2.csv')


def verify_population_constraints(db_path=None, csv_path=None):
    """
    验证数据库中的人口总和是否等于CSV约束

    参数:
        db_path: 数据库路径
        csv_path: CSV文件路径
    """
    if db_path is None:
        db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)
    if csv_path is None:
        csv_path = CSV_PATH

    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return

    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return

    print("=" * 100)
    print("人口约束验证 - 检查每年每城市的 Total_Count 总和是否等于 CSV 常住人口约束")
    print("=" * 100)
    print(f"数据库: {db_path}")
    print(f"CSV文件: {csv_path}\n")

    # 1. 读取CSV约束
    print("正在读取CSV约束...")
    df_csv = pd.read_csv(csv_path, engine='python')
    df_csv.columns = df_csv.columns.str.strip()

    # 列名映射
    col_mapping = {
        '年份': 'year',
        '城市代码': 'city_code',
        '常住人口数(人)': 'total_pop'
    }
    df_csv = df_csv.rename(columns=col_mapping)

    # 数据清洗
    df_csv['city_code'] = df_csv['city_code'].astype(str).str.split('.').str[0].str.strip()
    df_csv = df_csv[df_csv['city_code'].str.len() == 4]
    df_csv['year'] = pd.to_numeric(df_csv['year'], errors='coerce')
    df_csv['total_pop'] = pd.to_numeric(df_csv['total_pop'], errors='coerce')
    df_csv = df_csv.dropna(subset=['year', 'city_code', 'total_pop'])
    df_csv['year'] = df_csv['year'].astype(int)
    df_csv['total_pop'] = df_csv['total_pop'].astype(int)

    # 过滤年份范围
    df_csv = df_csv[(df_csv['year'] >= 2000) & (df_csv['year'] <= 2020)]

    print(f"CSV约束记录数: {len(df_csv):,}\n")

    # 2. 查询数据库
    print("正在查询数据库...")
    conn = duckdb.connect(db_path, read_only=True)

    query = """
        SELECT
            Year,
            Birth_Region as city_code,
            SUM(Total_Count) as db_total
        FROM migration_data
        GROUP BY Year, Birth_Region
        ORDER BY Year, Birth_Region
    """
    df_db = conn.execute(query).df()
    df_db['Year'] = df_db['Year'].astype(int)
    df_db['db_total'] = df_db['db_total'].astype(int)

    print(f"数据库记录数: {len(df_db):,}\n")

    # 3. 合并对比
    print("正在对比数据...")
    df_merged = pd.merge(
        df_csv,
        df_db,
        left_on=['year', 'city_code'],
        right_on=['Year', 'city_code'],
        how='outer',
        indicator=True
    )

    # 计算差异
    df_merged['csv_pop'] = df_merged['total_pop'].fillna(0).astype(int)
    df_merged['db_pop'] = df_merged['db_total'].fillna(0).astype(int)
    df_merged['diff'] = df_merged['db_pop'] - df_merged['csv_pop']
    df_merged['diff_abs'] = df_merged['diff'].abs()
    df_merged['diff_pct'] = (df_merged['diff_abs'] / df_merged['csv_pop'] * 100).fillna(0)
    df_merged['match'] = (df_merged['diff'] == 0)

    # 使用year列（优先使用非空的）
    df_merged['year_final'] = df_merged['year'].fillna(df_merged['Year']).astype(int)

    # 4. 统计结果
    print("=" * 100)
    print("总体统计")
    print("=" * 100)

    total_records = len(df_merged)
    matched_records = df_merged['match'].sum()
    match_rate = (matched_records / total_records * 100) if total_records > 0 else 0

    print(f"总记录数: {total_records:,}")
    print(f"完全匹配记录数: {matched_records:,}")
    print(f"匹配率: {match_rate:.2f}%")
    print(f"不匹配记录数: {total_records - matched_records:,}")

    # 只在CSV中存在的记录
    only_csv = df_merged[df_merged['_merge'] == 'left_only']
    print(f"\n只在CSV中存在（数据库缺失）: {len(only_csv):,}")

    # 只在数据库中存在的记录
    only_db = df_merged[df_merged['_merge'] == 'right_only']
    print(f"只在数据库中存在（CSV缺失）: {len(only_db):,}")

    # 5. 显示不匹配的记录
    mismatch_df = df_merged[~df_merged['match'] & (df_merged['_merge'] == 'both')].copy()

    if len(mismatch_df) > 0:
        print("\n" + "=" * 100)
        print(f"不匹配记录详情（按差异绝对值倒序，前20条）")
        print("=" * 100)

        mismatch_df = mismatch_df.sort_values('diff_abs', ascending=False)

        print(f"\n{'序号':<6} {'年份':<6} {'城市代码':<10} {'CSV人口':<15} {'DB人口':<15} "
              f"{'差异':<15} {'差异%':<10}")
        print("-" * 90)

        for idx, (_, row) in enumerate(mismatch_df.head(20).iterrows(), 1):
            print(f"{idx:<6} {int(row['year_final']):<6} {row['city_code']:<10} "
                  f"{row['csv_pop']:<15,} {row['db_pop']:<15,} "
                  f"{row['diff']:<+15,} {row['diff_pct']:<10.2f}%")

        # 按年份统计
        print("\n" + "=" * 100)
        print("按年份统计不匹配情况")
        print("=" * 100)

        print(f"\n{'年份':<8} {'不匹配数':<12} {'平均差异':<15} {'最大差异':<15} "
              f"{'总差异':<15}")
        print("-" * 75)

        for year in sorted(mismatch_df['year_final'].unique()):
            year_data = mismatch_df[mismatch_df['year_final'] == year]
            count = len(year_data)
            mean_diff = year_data['diff_abs'].mean()
            max_diff = year_data['diff_abs'].max()
            total_diff = year_data['diff_abs'].sum()

            print(f"{int(year):<8} {count:<12,} {mean_diff:<15,.0f} {max_diff:<15,.0f} "
                  f"{total_diff:<15,.0f}")

        # 按城市统计
        print("\n" + "=" * 100)
        print("不匹配最多的前10个城市")
        print("=" * 100)

        city_stats = mismatch_df.groupby('city_code').agg({
            'diff_abs': ['count', 'mean', 'sum']
        }).round(2)
        city_stats.columns = ['不匹配次数', '平均差异', '总差异']
        city_stats = city_stats.sort_values('不匹配次数', ascending=False).head(10)

        print(f"\n{'城市代码':<10} {'不匹配次数':<15} {'平均差异':<15} {'总差异':<15}")
        print("-" * 60)

        for city_code, row in city_stats.iterrows():
            print(f"{city_code:<10} {int(row['不匹配次数']):<15,} "
                  f"{row['平均差异']:<15,.0f} {row['总差异']:<15,.0f}")

    else:
        print("\n✓ 所有记录的数据库人口总和都等于CSV约束！")

    # 6. 全国总人口对比
    print("\n" + "=" * 100)
    print("全国总人口对比（按年份）")
    print("=" * 100)

    print(f"\n{'年份':<8} {'CSV总人口':<20} {'DB总人口':<20} {'差异':<15} {'差异%':<10}")
    print("-" * 80)

    for year in range(2000, 2021):
        csv_total = df_csv[df_csv['year'] == year]['total_pop'].sum()
        db_total = df_db[df_db['Year'] == year]['db_total'].sum()
        diff = db_total - csv_total
        diff_pct = (abs(diff) / csv_total * 100) if csv_total > 0 else 0

        print(f"{year:<8} {csv_total:<20,} {db_total:<20,} {diff:<+15,} {diff_pct:<10.4f}%")

    print("\n" + "=" * 100)
    print("验证完成")
    print("=" * 100)

    conn.close()


if __name__ == '__main__':
    # 支持命令行参数
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        csv_path = sys.argv[2] if len(sys.argv) > 2 else None
        verify_population_constraints(db_path, csv_path)
    else:
        verify_population_constraints()
