"""
读取0211处理后城市数据文件夹下所有xlsx文件
打印每个表的表名、schema和样例行
"""

import os
from pathlib import Path
import pandas as pd

# 配置路径
SOURCE_DIR = Path(r"C:\Users\w1625\Desktop\CityDBGenerate\0211处理后城市数据")
OUTPUT_FILE = Path(__file__).parent / "print_xlsx_schema.txt"


def print_dataframe_info(df, file_name, f):
    """打印DataFrame的schema和样例信息"""
    f.write("=" * 80 + "\n")
    f.write(f"文件名: {file_name}\n")
    f.write("=" * 80 + "\n\n")

    # 基本信息
    f.write(f"行数: {len(df)}\n")
    f.write(f"列数: {len(df.columns)}\n\n")

    # Schema（列名和数据类型）
    f.write("Schema (列名 | 数据类型):\n")
    f.write("-" * 80 + "\n")
    for col in df.columns:
        dtype = str(df[col].dtype)
        # 统计非空值数量
        non_null = df[col].notna().sum()
        f.write(f"  {col:<40} | {dtype:<15} | 非空值: {non_null}/{len(df)}\n")
    f.write("\n")

    # 第一行样例数据
    f.write("第一行样例:\n")
    f.write("-" * 80 + "\n")
    first_row = df.iloc[0]
    for col in df.columns:
        value = first_row[col]
        if pd.isna(value):
            value_str = "NaN"
        else:
            # 限制显示长度
            value_str = str(value)
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
        f.write(f"  {col:<40} | {value_str}\n")
    f.write("\n")

    # 打印前5行的县市名列和年份列
    if len(df.columns) >= 2:
        f.write("前5行的县市名和年份:\n")
        f.write("-" * 80 + "\n")
        city_col = df.columns[0]
        year_col = df.columns[1]
        for idx in range(min(5, len(df))):
            city_name = df.iloc[idx][city_col]
            year = df.iloc[idx][year_col]
            f.write(f"  行{idx}: {city_name} | {year}\n")
    f.write("\n\n")


def main():
    """主函数"""
    print("=" * 60)
    print("xlsx文件Schema分析工具")
    print("=" * 60)

    # 确保输出目录存在
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 获取所有xlsx文件
    xlsx_files = sorted(list(SOURCE_DIR.glob("*.xlsx")))

    if not xlsx_files:
        print(f"未找到xlsx文件，目录: {SOURCE_DIR}")
        return

    print(f"\n找到 {len(xlsx_files)} 个xlsx文件")
    print(f"输出文件: {OUTPUT_FILE}\n")

    # 处理每个文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("0211处理后城市数据 - Excel文件Schema分析\n")
        f.write("=" * 80 + "\n\n")

        for xlsx_file in xlsx_files:
            try:
                print(f"正在处理: {xlsx_file.name}")
                df = pd.read_excel(xlsx_file)
                print_dataframe_info(df, xlsx_file.name, f)
                print(f"  完成 ({len(df)}行 x {len(df.columns)}列)")
            except Exception as e:
                print(f"  错误: {e}")
                f.write(f"错误: {xlsx_file.name}\n")
                f.write(f"  {e}\n\n")

    print("\n" + "=" * 60)
    print(f"分析完成！结果已保存到: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
