"""
城市人口普查数据处理脚本
功能：读取city.jsonl，处理0211清洗后-00-10-20年的宏观人口普查城市文件夹中的所有xlsx文件
"""

import json
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

# 配置路径
CITY_JSONL_PATH = Path(__file__).parent.parent / "city.jsonl"
SOURCE_DIR = Path(__file__).parent.parent / "0211清洗后-00-10-20年的宏观人口普查城市"
OUTPUT_DIR = Path(__file__).parent.parent / "0211处理后城市数据"

# 四个直辖市及其代码
MUNICIPALITIES = {
    "北京": "1100",
    "天津": "1200",
    "上海": "3100",
    "重庆": "5000"
}

# 县级市到地级市的映射规则
# 格式: {"原始县市名": {"city_id": "xxxx", "name": "目标城市名"}}
COUNTY_TO_PREFECTURE_MAPPING = {
    "开远": {"city_id": "5325", "name": "红河"},
    "景洪": {"city_id": "5328", "name": "西双版纳"},
    "潞西": {"city_id": "5331", "name": "德宏"},
    "锡林浩特": {"city_id": "1525", "name": "锡林郭勒盟"},
    "阿尔山": {"city_id": "1522", "name": "兴安盟"},
    "和龙": {"city_id": "2224", "name": "延边"},
    "大安": {"city_id": "2208", "name": "白城"},
    "简阳": {"city_id": "5101", "name": "成都"},
    "西昌": {"city_id": "5134", "name": "凉山"},
    "巢湖": {"city_id": "3401", "name": "合肥"},
    "莱芜": {"city_id": "3701", "name": "济南"},
    "乌苏": {"city_id": "6542", "name": "塔城"},
    "伊宁": {"city_id": "6540", "name": "伊犁"},
    "博乐": {"city_id": "6527", "name": "博尔塔拉"},
    "库尔勒": {"city_id": "6528", "name": "巴音郭楞"},
    "阿图什": {"city_id": "6530", "name": "克孜勒苏"},
    "利川": {"city_id": "4228", "name": "恩施"},
    "广水": {"city_id": "4213", "name": "随州"},
    "襄樊": {"city_id": "4206", "name": "襄阳"},
    "吉首": {"city_id": "4331", "name": "湘西"},
    "涟源": {"city_id": "4313", "name": "娄底"},
    "合作": {"city_id": "6230", "name": "甘南"},
    "兴义": {"city_id": "5223", "name": "黔西南"},
    "凯里": {"city_id": "5226", "name": "黔东南"},
    "福泉": {"city_id": "5227", "name": "黔南"},
    "德令哈": {"city_id": "6328", "name": "海西"},
    "海伦": {"city_id": "2312", "name": "绥化"},
}


def load_city_mapping(jsonl_path):
    """
    加载城市映射数据
    返回: name_to_id dict 和 id_to_name dict
    """
    name_to_id = {}
    id_to_name = {}

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            city_id = data['city_id']
            name = data['name']
            name_to_id[name] = city_id
            id_to_name[city_id] = name

    print(f"已加载 {len(name_to_id)} 个城市映射")
    return name_to_id, id_to_name


def parse_city_name(raw_name):
    """
    解析原始县市名，返回 (省份_城市, 城市名, 是否直辖市)
    例如：
    - 北京市_东城区 -> (北京市, 东城区, True)
    - 山东省_滨州市 -> (山东省_滨州市, 滨州市, False)
    """
    if '_' not in raw_name:
        return raw_name, raw_name, False

    parts = raw_name.split('_')
    if len(parts) != 2:
        return raw_name, raw_name, False

    province_part, city_part = parts

    # 判断是否是直辖市
    for muni in MUNICIPALITIES:
        if province_part == muni + "市":
            return province_part, city_part, True

    return raw_name, city_part.replace('市', ''), False


def merge_rows(rows_to_merge, city_col, year_col, target_city_name, target_city_id, cols):
    """
    合并多行数据，数值字段加和，比例字段置空
    rows_to_merge: [(row_data, raw_name), ...] 其中 row_data 是 Series
    """
    # 创建合并后的行
    merged_row = {col: None for col in cols}
    merged_row[city_col] = f"{target_city_name}({target_city_id})"
    merged_row[year_col] = rows_to_merge[0][0][year_col]  # rows_to_merge[0][0] 是第一个 row_data

    # 处理每一列
    for col in cols:
        if col in [city_col, year_col]:
            continue

        # 判断是否是比例字段
        if '%' in str(col):
            merged_row[col] = None  # 比例字段置空
        else:
            # 数值字段加和
            total = sum(row_data[col] for row_data, _ in rows_to_merge if pd.notna(row_data[col]))
            merged_row[col] = total if total != 0 else None

    return pd.Series(merged_row)


def process_excel_file(file_path, name_to_id, id_to_name):
    """
    处理单个xlsx文件
    返回: 处理后的DataFrame 和 未匹配的城市列表
    """
    print(f"\n正在处理: {os.path.basename(file_path)}")

    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 获取列名
    cols = df.columns.tolist()
    city_col = cols[0]  # 第一列：县市名
    year_col = cols[1]   # 第二列：年份

    print(f"  原始行数: {len(df)}")
    print(f"  列数: {len(cols)}")

    unmatched_cities = set()

    # 第一遍：解析所有行并分类
    # 使用 (原始名称, 解析后名称, 映射后ID, 映射后名称) 来标记每一行
    parsed_rows = []  # [(原始行数据, 原始县市名, 解析后城市名, 目标city_id, 目标城市名, 是否匹配)]

    for idx, row in df.iterrows():
        raw_name = row[city_col]
        province_city, city_name, is_municipality = parse_city_name(raw_name)

        target_city_id = None
        target_city_name = None
        is_matched = False

        if is_municipality:
            # 直辖市
            muni_name = province_city.replace('市', '')
            target_city_id = MUNICIPALITIES[muni_name]
            target_city_name = muni_name
            is_matched = True
        else:
            # 非直辖市，先尝试在city.jsonl中精确匹配
            if city_name in name_to_id:
                target_city_id = name_to_id[city_name]
                target_city_name = city_name
                is_matched = True
            elif city_name + '市' in name_to_id:
                target_city_id = name_to_id[city_name + '市']
                target_city_name = city_name + '市'
                is_matched = True
            elif city_name.endswith('市') and city_name[:-1] in name_to_id:
                target_city_id = name_to_id[city_name[:-1]]
                target_city_name = city_name[:-1]
                is_matched = True
            elif city_name + '盟' in name_to_id:
                target_city_id = name_to_id[city_name + '盟']
                target_city_name = city_name + '盟'
                is_matched = True
            elif city_name.endswith('盟') and city_name[:-1] in name_to_id:
                target_city_id = name_to_id[city_name[:-1]]
                target_city_name = city_name[:-1]
                is_matched = True
            # 尝试县级市映射规则
            elif city_name in COUNTY_TO_PREFECTURE_MAPPING:
                mapping = COUNTY_TO_PREFECTURE_MAPPING[city_name]
                target_city_id = mapping["city_id"]
                target_city_name = mapping["name"]
                is_matched = True

        parsed_rows.append((row, raw_name, city_name, target_city_id, target_city_name, is_matched))

        if not is_matched:
            unmatched_cities.add(raw_name)

    # 第二遍：按年份和目标城市ID分组，处理需要合并的记录
    # year_city_groups: {年份: {目标城市ID: [(行数据, 原始名称), ...]}}
    year_city_groups = defaultdict(lambda: defaultdict(list))

    for row_data in parsed_rows:
        row, raw_name, city_name, target_id, target_name, is_matched = row_data
        if is_matched and target_id is not None:
            year = row[year_col]
            year_city_groups[year][target_id].append((row, raw_name))
        else:
            # 未匹配的记录单独处理
            year = row[year_col]
            year_city_groups[year][f"UNMATCHED_{raw_name}"].append((row, raw_name))

    # 第三遍：合并并生成最终结果
    final_results = []
    merged_info = []  # 记录合并信息用于打印

    for year, city_groups in year_city_groups.items():
        for target_id, rows_list in city_groups.items():
            if target_id.startswith("UNMATCHED_"):
                # 未匹配的记录，保留原始名称
                for row, raw_name in rows_list:
                    final_results.append(row)
            elif len(rows_list) == 1:
                # 只有一条记录，直接转换名称
                row, raw_name = rows_list[0]
                new_row = row.copy()
                # 找到对应的目标城市名
                target_name = None
                for rd in parsed_rows:
                    if rd[0].equals(row):
                        target_name = rd[4]
                        break
                if target_name:
                    new_row[city_col] = f"{target_name}({target_id})"
                final_results.append(new_row)
            else:
                # 多条记录需要加和
                original_names = [raw_name for _, raw_name in rows_list]
                # 找到目标城市名
                target_name = None
                for row, _ in rows_list:
                    for rd in parsed_rows:
                        if rd[0].equals(row):
                            target_name = rd[4]
                            break
                    if target_name:
                        break

                print(f"    {year}年 {target_name}({target_id}) 有 {len(rows_list)} 条记录，正在合并...")
                print(f"      原始名称: {', '.join(original_names)}")
                merged_info.append(f"      {year}年 {target_name}({target_id}): {len(rows_list)}条 -> {', '.join(original_names)}")

                # 准备合并数据
                rows_to_merge = [(row, raw_name) for row, raw_name in rows_list]
                merged_row = merge_rows(rows_to_merge, city_col, year_col, target_name, target_id, cols)
                final_results.append(merged_row)

    # 创建最终DataFrame
    final_df = pd.DataFrame(final_results)
    if len(final_df) > 0:
        final_df.columns = cols  # 确保列名正确

    print(f"  处理后行数: {len(final_df)}")

    if merged_info:
        print(f"  合并记录汇总:")
        for info in merged_info:
            print(f"    {info}")

    if unmatched_cities:
        print(f"  未匹配的城市 ({len(unmatched_cities)} 个):")
        for city in sorted(unmatched_cities):
            print(f"    - {city}")

    return final_df, list(unmatched_cities)


def main():
    """主函数"""
    print("=" * 60)
    print("城市人口普查数据处理脚本")
    print("=" * 60)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载城市映射
    name_to_id, id_to_name = load_city_mapping(CITY_JSONL_PATH)

    print(f"\n县级市到地级市映射规则: {len(COUNTY_TO_PREFECTURE_MAPPING)} 条")
    for county, mapping in sorted(COUNTY_TO_PREFECTURE_MAPPING.items()):
        print(f"  {county} -> {mapping['name']}({mapping['city_id']})")

    # 获取所有xlsx文件
    xlsx_files = list(Path(SOURCE_DIR).glob("*.xlsx"))
    print(f"\n找到 {len(xlsx_files)} 个xlsx文件")

    # 处理每个文件
    all_unmatched = defaultdict(set)  # {文件名: 未匹配城市集合}

    for xlsx_file in xlsx_files:
        try:
            processed_df, unmatched = process_excel_file(xlsx_file, name_to_id, id_to_name)

            # 保存处理后的文件
            output_path = os.path.join(OUTPUT_DIR, xlsx_file.name)
            processed_df.to_excel(output_path, index=False)
            print(f"  [OK] 已保存到: {output_path}")

            if unmatched:
                all_unmatched[xlsx_file.name] = set(unmatched)

        except Exception as e:
            print(f"  [ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()

    # 输出未匹配汇总
    print("\n" + "=" * 60)
    print("未匹配城市汇总")
    print("=" * 60)

    for file_name, cities in all_unmatched.items():
        if cities:
            print(f"\n{file_name}:")
            for city in sorted(cities):
                print(f"  - {city}")

    print("\n" + "=" * 60)
    print(f"处理完成！结果已保存到: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
