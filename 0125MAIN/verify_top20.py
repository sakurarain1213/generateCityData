# -*- coding: utf-8 -*-
"""
Top20 分布验证工具（独立脚本）

用法:
    python verify_top20.py
    或
    python verify_top20.py <数据库路径>
"""

import os
import sys
import duckdb
import pandas as pd

# 配置
OUTPUT_DIR = r"C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN"
DB_FILENAME = 'local_migration_data_full.db'
TOP_N_CITIES = 20


def verify_top20_distribution(db_path=None):
    """
    独立验证函数：检查 Top20 去向总和是否等于 Outflow_Count
    可以在数据库生成后单独运行

    参数:
        db_path: 数据库路径，如果为None则使用默认路径
    """
    if db_path is None:
        db_path = os.path.join(OUTPUT_DIR, DB_FILENAME)

    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return

    print("=" * 80)
    print("Top20 去向分布验证")
    print("=" * 80)
    print(f"数据库路径: {db_path}\n")

    conn = duckdb.connect(db_path, read_only=True)

    try:
        # 查询所有数据
        print("正在加载数据...")
        query = f"""
            SELECT
                Year, Month, Type_ID, Birth_Region, From_City,
                Total_Count, Stay_Prob, Outflow_Count,
                {', '.join([f'To_Top{i}_Count' for i in range(1, TOP_N_CITIES + 1)])}
            FROM migration_data
        """
        df = conn.execute(query).df()

        print(f"总记录数: {len(df):,}\n")

        # 计算 Top20 去向总和
        top_cols = [f'To_Top{i}_Count' for i in range(1, TOP_N_CITIES + 1)]
        df['Top20_Sum'] = df[top_cols].sum(axis=1)

        # 计算差异
        df['Diff'] = df['Top20_Sum'] - df['Outflow_Count']
        df['Diff_Abs'] = df['Diff'].abs()
        df['Match'] = (df['Diff'] == 0)

        # 统计匹配情况
        total_rows = len(df)
        matched_rows = df['Match'].sum()
        match_rate = (matched_rows / total_rows * 100) if total_rows > 0 else 0

        print("=" * 80)
        print("总体统计")
        print("=" * 80)
        print(f"总记录数: {total_rows:,}")
        print(f"完全匹配记录数: {matched_rows:,}")
        print(f"匹配率: {match_rate:.2f}%")
        print(f"不匹配记录数: {total_rows - matched_rows:,}")

        # 如果有不匹配的记录
        if matched_rows < total_rows:
            print("\n" + "=" * 80)
            print("不匹配样本分析（按 Outflow_Count 和 Top20_Sum 绝对值倒序）")
            print("=" * 80)

            # 筛选不匹配的记录
            mismatch_df = df[~df['Match']].copy()

            # 按 Outflow_Count 和 Top20_Sum 的绝对值倒序排序
            mismatch_df['Total_Flow'] = mismatch_df['Outflow_Count'] + mismatch_df['Top20_Sum']
            mismatch_df = mismatch_df.sort_values('Total_Flow', ascending=False)

            # 打印前20条
            print(f"\n前20条不匹配记录:\n")

            print(f"{'序号':<6} {'年份':<6} {'城市':<10} {'Type_ID':<45} "
                  f"{'总人数':<12} {'流出数':<12} {'Top20和':<12} {'差异':<10}")
            print("-" * 130)

            for idx, (_, row) in enumerate(mismatch_df.head(20).iterrows(), 1):
                # 截断Type_ID以适应显示
                type_id = row['Type_ID'][:43] if len(row['Type_ID']) > 43 else row['Type_ID']
                print(f"{idx:<6} {row['Year']:<6} {row['Birth_Region']:<10} {type_id:<45} "
                      f"{row['Total_Count']:<12,} {row['Outflow_Count']:<12,} "
                      f"{row['Top20_Sum']:<12,} {row['Diff']:<+10,}")

            # 按年份统计
            print("\n" + "=" * 80)
            print("按年份统计不匹配情况")
            print("=" * 80)

            print(f"\n{'年份':<8} {'不匹配数':<12} {'平均差异':<12} {'最大差异':<12} "
                  f"{'总流出':<15} {'Top20总和':<15}")
            print("-" * 90)

            for year in sorted(mismatch_df['Year'].unique()):
                year_data = mismatch_df[mismatch_df['Year'] == year]
                count = len(year_data)
                mean_diff = year_data['Diff_Abs'].mean()
                max_diff = year_data['Diff_Abs'].max()
                total_outflow = year_data['Outflow_Count'].sum()
                total_top20 = year_data['Top20_Sum'].sum()

                print(f"{year:<8} {count:<12,} {mean_diff:<12,.0f} {max_diff:<12,.0f} "
                      f"{total_outflow:<15,} {total_top20:<15,}")

            # 按Type_ID统计（显示不匹配最多的前10个Type）
            print("\n" + "=" * 80)
            print("不匹配最多的前10个Type_ID")
            print("=" * 80)

            type_stats = mismatch_df.groupby('Type_ID').agg({
                'Diff_Abs': ['count', 'mean', 'sum']
            }).round(2)
            type_stats.columns = ['不匹配次数', '平均差异', '总差异']
            type_stats = type_stats.sort_values('不匹配次数', ascending=False).head(10)

            print(f"\n{'Type_ID':<50} {'不匹配次数':<15} {'平均差异':<15} {'总差异':<15}")
            print("-" * 95)

            for type_id, row in type_stats.iterrows():
                # 截断Type_ID
                type_id_display = type_id[:48] if len(type_id) > 48 else type_id
                print(f"{type_id_display:<50} {int(row['不匹配次数']):<15,} "
                      f"{row['平均差异']:<15,.0f} {row['总差异']:<15,.0f}")

            # 按城市统计（只显示不匹配最多的前10个城市）
            print("\n" + "=" * 80)
            print("不匹配最多的前10个城市")
            print("=" * 80)

            city_stats = mismatch_df.groupby('Birth_Region').agg({
                'Diff_Abs': ['count', 'mean', 'sum']
            }).round(2)
            city_stats.columns = ['不匹配次数', '平均差异', '总差异']
            city_stats = city_stats.sort_values('不匹配次数', ascending=False).head(10)

            print(f"\n{'城市代码':<10} {'不匹配次数':<15} {'平均差异':<15} {'总差异':<15}")
            print("-" * 60)

            for city_code, row in city_stats.iterrows():
                print(f"{city_code:<10} {int(row['不匹配次数']):<15,} "
                      f"{row['平均差异']:<15,.0f} {row['总差异']:<15,.0f}")

        else:
            print("\n✓ 所有记录的 Top20 去向总和都等于 Outflow_Count！")

        print("\n" + "=" * 80)
        print("验证完成")
        print("=" * 80)

    except Exception as e:
        print(f"验证过程出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == '__main__':
    # 支持命令行参数指定数据库路径
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
        verify_top20_distribution(db_path)
    else:
        # 使用默认路径
        verify_top20_distribution()
