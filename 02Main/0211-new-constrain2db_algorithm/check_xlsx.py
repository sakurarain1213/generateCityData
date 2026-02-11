"""
分析 constrain2.xlsx 文件
按年份汇总 Total_Count 和 Outflow_Count
计算迁徙比率 = (Outflow_Count / Total_Count) * 100
"""

import pandas as pd
from pathlib import Path

# 文件路径
xlsx_file = Path(r"C:\Users\w1625\Desktop\CityDBGenerate\0211constrain2\constrain2.xlsx")
output_file = Path(__file__).parent / "migration_summary.txt"

def process_migration_data():
    # 读取 Excel 文件
    print(f"正在读取文件: {xlsx_file}")
    df = pd.read_excel(xlsx_file)

    print(f"数据行数: {len(df)}")
    print(f"列名: {df.columns.tolist()}")

    # 按年份分组汇总
    summary = df.groupby('Year')[['Total_Count', 'Outflow_Count']].sum().reset_index()

    # 计算迁徙比率
    summary['Migration_Rate'] = (summary['Outflow_Count'] / summary['Total_Count']) * 100

    # 格式化数字（添加千位分隔符）
    summary['Total_Count_Formatted'] = summary['Total_Count'].apply(lambda x: f"{x:,}")
    summary['Outflow_Count_Formatted'] = summary['Outflow_Count'].apply(lambda x: f"{x:,}")
    summary['Migration_Rate_Formatted'] = summary['Migration_Rate'].apply(lambda x: f"{x:.4f}%")

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write("年份 | 全国总人口 | 总迁徙人口 | 迁徙比率(%)\n")
        f.write("-" * 70 + "\n")

        # 写入数据
        for _, row in summary.iterrows():
            line = f"{int(row['Year'])} | {row['Total_Count_Formatted']} | {row['Outflow_Count_Formatted']} | {row['Migration_Rate_Formatted']}"
            f.write(line + "\n")

    # 同时打印到控制台
    print("\n" + "=" * 70)
    print("汇总结果:")
    print("=" * 70)
    print("年份 | 全国总人口 | 总迁徙人口 | 迁徙比率(%)")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"{int(row['Year'])} | {row['Total_Count_Formatted']} | {row['Outflow_Count_Formatted']} | {row['Migration_Rate_Formatted']}")

    print(f"\n结果已保存到: {output_file}")

    # 显示详细数据
    print("\n详细数据:")
    print("-" * 70)
    for _, row in summary.iterrows():
        print(f"年份 {int(row['Year'])}:")
        print(f"  全国总人口: {row['Total_Count']:,.0f}")
        print(f"  总迁徙人口: {row['Outflow_Count']:,.0f}")
        print(f"  迁徙比率: {row['Migration_Rate']:.4f}%")
        print()

if __name__ == "__main__":
    process_migration_data()
