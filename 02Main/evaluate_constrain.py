
import pandas as pd
import json
import re

def load_city_ids(jsonl_path):
    """从jsonl文件加载所有城市ID"""
    city_ids = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            city_ids.add(data['city_id'])
    return city_ids

def extract_city_code(cell_value):
    """从单元格值中提取四位编码"""
    if pd.isna(cell_value) or cell_value == '':
        return None

    cell_str = str(cell_value).strip()
    # 使用正则表达式提取括号中的四位数字
    match = re.search(r'\((\d{4})\)', cell_str)
    if match:
        return match.group(1)
    return None

def extract_city_name(cell_value):
    """从单元格值中提取城市名称"""
    if pd.isna(cell_value) or cell_value == '':
        return None

    cell_str = str(cell_value).strip()
    # 使用正则表达式提取括号前的城市名
    match = re.match(r'([^\(]+)\(\d{4}\)', cell_str)
    if match:
        return match.group(1).strip()
    return None

def main():
    # 文件路径
    excel_path = r'C:\Users\w1625\Desktop\数据\city_outflow_complete_2000_2020_v15c_final.xlsx'
    jsonl_path = r'c:\Users\w1625\Desktop\db-generate\02Main\city.jsonl'

    # 加载jsonl中的所有城市ID
    print("正在加载city.jsonl中的城市ID...")
    city_ids_in_jsonl = load_city_ids(jsonl_path)
    print(f"city.jsonl中共有 {len(city_ids_in_jsonl)} 个城市\n")

    # 读取Excel文件
    print("正在读取Excel文件...")
    df = pd.read_excel(excel_path)
    print(f"Excel文件共有 {len(df)} 行, {len(df.columns)} 列\n")

    # 要检查的列名
    columns_to_check = ['From_City'] + [f'To_Top{i}' for i in range(1, 21)]

    # 存储未在jsonl中找到的城市 (编码 -> 名称)
    missing_cities = {}

    print("正在检查所有单元格...")
    checked_cells = 0

    for col in columns_to_check:
        if col not in df.columns:
            print(f"警告: 列 '{col}' 不存在于Excel中")
            continue

        for cell_value in df[col]:
            checked_cells += 1
            city_code = extract_city_code(cell_value)
            city_name = extract_city_name(cell_value)

            if city_code and city_code not in city_ids_in_jsonl:
                # 如果这个编码还没记录过，或者当前名称更完整（非空）
                if city_code not in missing_cities or (city_name and city_name != ''):
                    missing_cities[city_code] = city_name if city_name else f"未知城市({city_code})"

    print(f"共检查了 {checked_cells} 个单元格\n")
    print(f"=" * 60)
    print(f"未在city.jsonl中找到的城市（共 {len(missing_cities)} 个）:")
    print(f"=" * 60)

    # 按编码排序输出
    for city_code in sorted(missing_cities.keys()):
        city_name = missing_cities[city_code]
        print(f"编码: {city_code}, 名称: {city_name}")

    # 保存结果到文件
    output_file = r'c:\Users\w1625\Desktop\DBgenerate\missing_cities.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"未在city.jsonl中找到的城市（共 {len(missing_cities)} 个）:\n")
        f.write("=" * 60 + "\n\n")
        for city_code in sorted(missing_cities.keys()):
            city_name = missing_cities[city_code]
            f.write(f"编码: {city_code}, 名称: {city_name}\n")

    print(f"\n=" * 60)
    print(f"检查完成！共发现 {len(missing_cities)} 个缺失的城市")
    print(f"结果已保存到: {output_file}")

if __name__ == '__main__':
    main()
