import duckdb
import os

def step1_print_schema(db_path):
    """
    第一步：打印表schema和样例数据，保存到schema.txt
    """
    print("=" * 60)
    print("第一步：打印表结构和样例数据")
    print("=" * 60)

    con = duckdb.connect(db_path)

    # 获取表结构
    schema_query = "DESCRIBE SELECT * FROM migration_data"
    schema_result = con.execute(schema_query).fetchall()

    print("\n表名：migration_data")
    print("\n列名和类型：")
    print("-" * 60)
    for col in schema_result:
        print(f"{col[0]:<30} {col[1]}")

    # 随机抽取两行样例
    sample_query = "SELECT * FROM migration_data ORDER BY RANDOM() LIMIT 2"
    samples = con.execute(sample_query).fetchall()
    columns = [col[0] for col in schema_result]

    print("\n随机样例数据（2行）：")
    print("-" * 60)
    for i, row in enumerate(samples, 1):
        print(f"\n样例 {i}:")
        for col, val in zip(columns, row):
            print(f"  {col}: {val}")

    # 保存到schema.txt
    output_path = os.path.join(os.path.dirname(db_path), "schema.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("表名：migration_data\n\n")
        f.write("列名和类型：\n")
        f.write("-" * 60 + "\n")
        for col in schema_result:
            f.write(f"{col[0]:<30} {col[1]}\n")

        f.write("\n随机样例数据（2行）：\n")
        f.write("-" * 60 + "\n")
        for i, row in enumerate(samples, 1):
            f.write(f"\n样例 {i}:\n")
            for col, val in zip(columns, row):
                f.write(f"  {col}: {val}\n")

    print(f"\n[OK] Schema信息已保存到: {output_path}")

    con.close()
    print()


def step2_generate_city_summary_csv(db_path):
    """
    第二步：生成城市级别的汇总数据，保存为CSV
    """
    print("=" * 60)
    print("第二步：生成城市汇总数据CSV")
    print("=" * 60)

    con = duckdb.connect(db_path)

    # 按城市代码升序、同一城市内按年份升序排列
    # migration_population = 总人口 - 流出人口（留守人口）
    # ratio = 留守人口 / 总人口（百分比格式 xx.xx）
    query = """
    SELECT
        year,
        from_city AS city,
        SUM(total_count) AS total_population,
        SUM(total_count) - SUM(outflow_count) AS migration_population,
        ROUND((SUM(total_count) - SUM(outflow_count)) * 100.0 / SUM(total_count), 2) AS ratio
    FROM migration_data
    GROUP BY year, from_city
    ORDER BY from_city ASC, year ASC
    """

    result = con.execute(query).fetchall()

    # 保存为CSV
    output_path = os.path.join(os.path.dirname(db_path), "origin_sum_data.csv")
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        # 写入表头
        f.write("year,city,total_population,migration_population,ratio\n")
        # 写入数据
        for row in result:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

    print(f"[OK] 城市汇总数据已保存到: {output_path}")
    print(f"  共 {len(result)} 行数据")

    con.close()
    print()


def step3_calculate_national_totals(db_path):
    """
    第三步：计算每年全国总人口、总迁徙人口和比率
    """
    print("=" * 60)
    print("第三步：计算全国总人口和迁徙人口")
    print("=" * 60)

    con = duckdb.connect(db_path)

    # 计算每年全国总人口和总迁徙人口
    query = """
    SELECT
        year,
        SUM(total_count) AS national_total_population,
        SUM(outflow_count) AS national_migration_population,
        ROUND(SUM(outflow_count) * 100.0 / SUM(total_count), 4) AS migration_rate
    FROM migration_data
    GROUP BY year
    ORDER BY year ASC
    """

    result = con.execute(query).fetchall()

    # 保存到all_city_cnt.txt
    output_path = os.path.join(os.path.dirname(db_path), "all_city_cnt.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("年份 | 全国总人口 | 总迁徙人口 | 迁徙比率(%)")
        f.write("\n" + "-" * 70 + "\n")
        for row in result:
            year = int(row[0])
            total_pop = int(row[1])
            migration_pop = int(row[2])
            rate = float(row[3])
            f.write(f"{year} | {total_pop:,} | {migration_pop:,} | {rate}%\n")

    print(f"[OK] 全国统计数据已保存到: {output_path}")
    print(f"\n数据预览：")
    print("-" * 70)
    print("年份 | 全国总人口 | 总迁徙人口 | 迁徙比率(%)")
    print("-" * 70)
    for row in result:
        year = int(row[0])
        total_pop = int(row[1])
        migration_pop = int(row[2])
        rate = float(row[3])
        print(f"{year} | {total_pop:,} | {migration_pop:,} | {rate}%")

    con.close()
    print()


def step4_sample_random_records(db_path, sample_size=100):
    """
    第四步：从数据库中随机采样指定数量的记录，保存为CSV
    """
    print("=" * 60)
    print(f"第四步：随机采样 {sample_size} 条记录")
    print("=" * 60)

    con = duckdb.connect(db_path)

    # 随机采样指定数量的记录
    query = f"""
    SELECT * FROM migration_data
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """

    result = con.execute(query).fetchall()

    # 获取列名
    columns_query = "DESCRIBE SELECT * FROM migration_data"
    columns_result = con.execute(columns_query).fetchall()
    columns = [col[0] for col in columns_result]

    # 保存为CSV
    output_path = os.path.join(os.path.dirname(db_path), "sample_data.csv")
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        # 写入表头
        f.write(",".join(columns) + "\n")
        # 写入数据
        for row in result:
            # 将每个值转换为字符串并处理可能包含逗号的情况
            row_str = ",".join(str(val) if val is not None else "" for val in row)
            f.write(row_str + "\n")

    print(f"[OK] 采样数据已保存到: {output_path}")
    print(f"  共 {len(result)} 行数据")
    print(f"  列: {', '.join(columns)}")

    con.close()
    print()


def step5_analyze_constrain_data():
    """
    第五步：读取constrain.csv，统计每年全国迁徙数据
    """
    print("=" * 60)
    print("第五步：分析constrain.csv数据")
    print("=" * 60)

    import csv

    # constrain.csv文件路径
    csv_path = os.path.join(os.path.dirname(__file__), "constrain.csv")

    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"[错误] 文件不存在: {csv_path}")
        return

    # 读取CSV文件
    yearly_stats = {}  # {year: {'total_population': sum, 'outflow_population': sum}}

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            year = row['Year']

            # 初始化该年份的统计数据
            if year not in yearly_stats:
                yearly_stats[year] = {
                    'total_population': 0,
                    'outflow_population': 0
                }

            # 累加总人口和流出人口
            yearly_stats[year]['total_population'] += int(row['Total_Count'])
            yearly_stats[year]['outflow_population'] += int(row['Outflow_Count'])

    # 按年份排序
    sorted_years = sorted(yearly_stats.keys())

    # 保存到constrain.txt
    output_path = os.path.join(os.path.dirname(__file__), "constrain.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("年份 | 全国总人口 | 全国迁徙人口 | 人口流动率(%)\n")
        f.write("-" * 70 + "\n")

        for year in sorted_years:
            stats = yearly_stats[year]
            total_pop = stats['total_population']
            outflow_pop = stats['outflow_population']
            flow_rate = round(outflow_pop * 100.0 / total_pop, 2) if total_pop > 0 else 0

            f.write(f"{year} | {total_pop:,} | {outflow_pop:,} | {flow_rate}%\n")

    print(f"[OK] 统计结果已保存到: {output_path}")
    print(f"\n数据预览：")
    print("-" * 70)
    print("年份 | 全国总人口 | 全国迁徙人口 | 人口流动率(%)")
    print("-" * 70)

    for year in sorted_years:
        stats = yearly_stats[year]
        total_pop = stats['total_population']
        outflow_pop = stats['outflow_population']
        flow_rate = round(outflow_pop * 100.0 / total_pop, 2) if total_pop > 0 else 0

        print(f"{year} | {total_pop:,} | {outflow_pop:,} | {flow_rate}%")

    print()


def step6_sample_output_db(db_path="output.db", sample_size=100):
    """
    第六步：从output.db中随机采样指定数量的记录，保存为sample.csv
    """
    print("=" * 60)
    print(f"第六步：从output.db随机采样 {sample_size} 条记录")
    print("=" * 60)

    # output.db文件路径
    output_db_path = os.path.join(os.path.dirname(__file__), db_path)

    # 检查文件是否存在
    if not os.path.exists(output_db_path):
        print(f"[错误] 文件不存在: {output_db_path}")
        return

    con = duckdb.connect(output_db_path)

    # 获取所有表名
    tables_query = "SHOW TABLES"
    tables_result = con.execute(tables_query).fetchall()

    if not tables_result:
        print("[错误] 数据库中没有表")
        con.close()
        return

    # 获取第一个表名（假设主要数据在第一个表中）
    table_name = tables_result[0][0]
    print(f"使用表: {table_name}")

    # 随机采样指定数量的记录
    query = f"""
    SELECT * FROM {table_name}
    ORDER BY RANDOM()
    LIMIT {sample_size}
    """

    result = con.execute(query).fetchall()

    # 获取列名
    columns_query = f"DESCRIBE SELECT * FROM {table_name}"
    columns_result = con.execute(columns_query).fetchall()
    columns = [col[0] for col in columns_result]

    # 保存为CSV
    output_path = os.path.join(os.path.dirname(__file__), "sample.csv")
    with open(output_path, 'w', encoding='utf-8-sig') as f:
        # 写入表头
        f.write(",".join(columns) + "\n")
        # 写入数据
        for row in result:
            # 将每个值转换为字符串并处理可能包含逗号的情况
            row_str = ",".join(str(val) if val is not None else "" for val in row)
            f.write(row_str + "\n")

    print(f"[OK] 采样数据已保存到: {output_path}")
    print(f"  共 {len(result)} 行数据")
    print(f"  列: {', '.join(columns)}")

    con.close()
    print()


def step7_replace_unknown_city_names(db_path="output.db"):
    """
    第七步：处理output.db，将"未知xxxx(xxxx)"格式的城市名替换为"城市名(xxxx)"格式
    直接在原.db文件上修改，逐个城市代码处理并显示进度条
    """
    print("=" * 60)
    print("第七步：替换未知城市名称")
    print("=" * 60)

    import json
    from tqdm import tqdm

    # 数据库文件路径
    db_file = os.path.join(os.path.dirname(__file__), db_path)

    # 检查输入文件是否存在
    if not os.path.exists(db_file):
        print(f"[错误] 文件不存在: {db_file}")
        return

    # 加载city.jsonl建立代码到城市名的映射
    city_jsonl_path = os.path.join(os.path.dirname(__file__), "city.jsonl")
    if not os.path.exists(city_jsonl_path):
        print(f"[错误] 文件不存在: {city_jsonl_path}")
        return

    city_code_to_name = {}
    with open(city_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            city_code_to_name[data['city_id']] = data['name']

    print(f"[OK] 已加载 {len(city_code_to_name)} 个城市代码映射\n")

    # 连接数据库
    con = duckdb.connect(db_file)

    # 获取所有表名
    tables_query = "SHOW TABLES"
    tables_result = con.execute(tables_query).fetchall()

    if not tables_result:
        print("[错误] 数据库中没有表")
        con.close()
        return

    # 使用第一个表名
    table_name = tables_result[0][0]
    print(f"使用表: {table_name}")

    # 获取表结构
    schema_query = f"DESCRIBE SELECT * FROM {table_name}"
    schema_result = con.execute(schema_query).fetchall()
    columns = [col[0] for col in schema_result]

    # 找到所有To_Top开头的文本列（排除Count列）
    to_top_columns = [col for col in columns if col.startswith('To_Top') and not col.endswith('_Count')]

    # 获取总行数
    count_query = f"SELECT COUNT(*) FROM {table_name}"
    total_rows = con.execute(count_query).fetchone()[0]
    print(f"总行数: {total_rows}")
    print(f"处理的列: from_city, {', '.join(to_top_columns)}\n")

    # 统计总替换次数
    total_replace_count = 0

    # 遍历每个城市代码
    print("开始处理城市代码...")
    for city_code, city_name in tqdm(city_code_to_name.items(), total=len(city_code_to_name), desc="处理进度"):
        # 构建需要更新的列列表
        columns_to_update = ['from_city'] + to_top_columns

        # 对每一列执行UPDATE操作
        for col in columns_to_update:
            # 先查询符合条件的行数
            count_query = f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE {col} LIKE '%未知%({city_code})'
            """
            affected_rows = con.execute(count_query).fetchone()[0]

            # 如果有需要更新的行，执行UPDATE
            if affected_rows > 0:
                update_query = f"""
                UPDATE {table_name}
                SET {col} = '{city_name}({city_code})'
                WHERE {col} LIKE '%未知%({city_code})'
                """
                con.execute(update_query)
                total_replace_count += affected_rows

    con.close()

    print(f"\n[OK] 替换完成！")
    print(f"  共替换 {total_replace_count} 个未知城市名称")
    print(f"  数据库已更新: {db_file}")
    print()


def main():
    # 数据库文件路径
    db_path = os.path.join(os.path.dirname(__file__), "local_migration_data.db")

    print("\n" + "=" * 60)
    print("开始处理迁移数据")
    print("=" * 60 + "\n")

    # 执行各个步骤
    # step1_print_schema(db_path)
    # step2_generate_city_summary_csv(db_path)
    # step3_calculate_national_totals(db_path)
    # step4_sample_random_records(db_path, sample_size=100)
    # step5_analyze_constrain_data()
    step6_sample_output_db(db_path="output.db", sample_size=100)
    # step7_replace_unknown_city_names(db_path="output.db")

    print("=" * 60)
    print("[OK] 所有步骤完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
