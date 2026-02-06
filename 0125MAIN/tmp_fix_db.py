import duckdb
import pandas as pd
import shutil
#  在db222之后进行校准和后处理 得到更新的.db文件
def main():
    # 设置DuckDB使用所有可用核心
    db_path = "local_migration_data.db"
    excel_path = "5.xlsx"
    output_db_path = "local_migration_data_adjusted.db"

    print("步骤1: 读取5.xlsx获取每年的真实inflow_count总数...")
    df_real = pd.read_excel(excel_path)
    # 获取每年的真实总数
    yearly_real_counts = df_real.groupby('year')['inflow_count'].sum().to_dict()
    print(f"读取到 {len(yearly_real_counts)} 年的真实数据")
    print(f"年份范围: {min(yearly_real_counts.keys())} - {max(yearly_real_counts.keys())}")

    print("\n步骤2: 连接数据库并计算每年的Outflow_Count总数...")
    # 连接到数据库，设置线程数
    conn = duckdb.connect(db_path)
    conn.execute("SET threads TO 14")

    # 计算数据库中每年的Outflow_Count总数
    query_yearly_sum = """
    SELECT Year, SUM(Outflow_Count) as db_total
    FROM migration_data
    GROUP BY Year
    ORDER BY Year
    """
    df_db_yearly = conn.execute(query_yearly_sum).fetchdf()
    yearly_db_counts = dict(zip(df_db_yearly['Year'], df_db_yearly['db_total']))

    print(f"数据库中有 {len(yearly_db_counts)} 年的数据")

    print("\n步骤3: 计算每年的调整比率...")
    # 计算每年的调整比率
    yearly_ratios = {}
    for year in yearly_db_counts.keys():
        if year in yearly_real_counts:
            db_total = yearly_db_counts[year]
            real_total = yearly_real_counts[year]
            ratio = real_total / db_total if db_total > 0 else 1.0
            yearly_ratios[year] = ratio
            print(f"  年份 {year}: 数据库总数={db_total}, 真实总数={real_total}, 比率={ratio:.6f}")
        else:
            print(f"  警告: 年份 {year} 在5.xlsx中没有对应数据，使用比率=1.0")
            yearly_ratios[year] = 1.0

    print(f"\n步骤4: 复制数据库到新文件...")
    conn.close()
    # 复制数据库文件
    import shutil
    shutil.copy2(db_path, output_db_path)
    print(f"已复制到 {output_db_path}")

    print("\n步骤5: 在新数据库中批量更新所有记录...")
    # 连接到新数据库
    conn = duckdb.connect(output_db_path)
    conn.execute("SET threads TO 14")

    # 创建年份比率的临时表
    ratio_df = pd.DataFrame(list(yearly_ratios.items()), columns=['Year', 'Ratio'])
    conn.register('yearly_ratios', ratio_df)

    print("  正在更新Outflow_Count和To_TopX_Count列...")
    # 构建更新语句 - 一次性更新所有To_TopX_Count列
    top_columns = []
    for i in range(1, 21):  # Top1 到 Top20
        top_columns.append(f"To_Top{i}_Count")

    # 构建SET子句
    set_clauses = ["Outflow_Count = CAST(ROUND(Outflow_Count * r.Ratio) AS INTEGER)"]
    for col in top_columns:
        set_clauses.append(f"{col} = CAST(ROUND({col} * r.Ratio) AS INTEGER)")

    update_query = f"""
    UPDATE migration_data
    SET {', '.join(set_clauses)}
    FROM yearly_ratios r
    WHERE migration_data.Year = r.Year
    """

    conn.execute(update_query)
    print("  已完成Outflow_Count和To_TopX_Count的调整")

    print("\n步骤6: 根据Stay_Prob重新计算Total_Count...")
    # 更新Total_Count = Outflow_Count / Stay_Prob (取整)
    # 注意：Stay_Prob可能为0或接近0，需要处理
    update_total_query = """
    UPDATE migration_data
    SET Total_Count = CASE
        WHEN Stay_Prob > 0 THEN CAST(ROUND(Outflow_Count / Stay_Prob) AS INTEGER)
        ELSE Outflow_Count
    END
    """
    conn.execute(update_total_query)
    print("  已完成Total_Count的重新计算")

    print("\n步骤7: 验证更新结果...")
    # 验证：检查更新后的每年总数
    df_new_yearly = conn.execute(query_yearly_sum).fetchdf()
    print("\n更新后各年份的Outflow_Count总数:")
    for _, row in df_new_yearly.iterrows():
        year = row['Year']
        new_total = row['db_total']
        real_total = yearly_real_counts.get(year, 0)
        diff_pct = abs(new_total - real_total) / real_total * 100 if real_total > 0 else 0
        print(f"  年份 {year}: 更新后={new_total}, 真实值={real_total}, 差异={diff_pct:.2f}%")

    # 关闭连接
    conn.close()

    print(f"\n✓ 处理完成！新数据库已保存为: {output_db_path}")
    print(f"✓ 使用了14个线程进行并行处理")


if __name__ == "__main__":
    main()


'''
import duckdb
import pandas as pd

def compare_databases():
    """对比新老数据库的2000年和2020年的记录"""

    old_db = "local_migration_data.db"
    new_db = "local_migration_data_adjusted.db"

    print("=" * 100)
    print("数据库对比工具 - 查看2000年和2020年的记录变化")
    print("=" * 100)

    # 连接到两个数据库
    conn_old = duckdb.connect(old_db, read_only=True)
    conn_new = duckdb.connect(new_db, read_only=True)

    # 要对比的年份
    years = [2000, 2020]

    for year in years:
        print(f"\n{'=' * 100}")
        print(f"年份: {year}")
        print(f"{'=' * 100}")

        # 从旧数据库读取第一条记录
        query = f"SELECT * FROM migration_data WHERE Year = {year} LIMIT 1"
        df_old = conn_old.execute(query).fetchdf()
        df_new = conn_new.execute(query).fetchdf()

        if len(df_old) == 0:
            print(f"警告: {year}年在数据库中没有记录")
            continue

        # 获取第一条记录
        old_row = df_old.iloc[0]
        new_row = df_new.iloc[0]

        print(f"\n【基本信息】")
        print(f"  Type_ID: {old_row['Type_ID']}")
        print(f"  From_City: {old_row['From_City']}")
        print(f"  Birth_Region: {old_row['Birth_Region']}")

        print(f"\n【核心字段对比】")
        print(f"  {'字段':<20} {'旧值':>15} {'新值':>15} {'变化':>15}")
        print(f"  {'-' * 70}")

        # Total_Count
        old_total = old_row['Total_Count']
        new_total = new_row['Total_Count']
        change = new_total - old_total
        print(f"  {'Total_Count':<20} {old_total:>15,} {new_total:>15,} {change:>+15,}")

        # Stay_Prob (不变)
        stay_prob = old_row['Stay_Prob']
        print(f"  {'Stay_Prob':<20} {stay_prob:>15.6f} {new_row['Stay_Prob']:>15.6f} {'(不变)':>15}")

        # Outflow_Count
        old_outflow = old_row['Outflow_Count']
        new_outflow = new_row['Outflow_Count']
        change = new_outflow - old_outflow
        ratio = new_outflow / old_outflow if old_outflow > 0 else 0
        print(f"  {'Outflow_Count':<20} {old_outflow:>15,} {new_outflow:>15,} {change:>+15,} (×{ratio:.4f})")

        print(f"\n【Top目的地对比 (前10个)】")
        print(f"  {'目的地':<15} {'旧Count':>12} {'新Count':>12} {'变化':>12} {'比率':>10}")
        print(f"  {'-' * 70}")

        for i in range(1, 11):
            city_col = f'To_Top{i}'
            count_col = f'To_Top{i}_Count'

            city = old_row[city_col]
            old_count = old_row[count_col]
            new_count = new_row[count_col]
            change = new_count - old_count
            ratio = new_count / old_count if old_count > 0 else 0

            print(f"  {city:<15} {old_count:>12,} {new_count:>12,} {change:>+12,} {ratio:>10.4f}x")

        # 验证计算
        print(f"\n【验证计算】")
        calculated_total = round(new_outflow / stay_prob) if stay_prob > 0 else new_outflow
        print(f"  新Total_Count: {new_total:,}")
        print(f"  计算值 (Outflow/Stay_Prob): {calculated_total:,}")
        print(f"  是否匹配: {'✓' if calculated_total == new_total else '✗'}")

        # 计算Top目的地总和
        old_top_sum = sum(old_row[f'To_Top{i}_Count'] for i in range(1, 21))
        new_top_sum = sum(new_row[f'To_Top{i}_Count'] for i in range(1, 21))
        print(f"\n  Top1-20总和(旧): {old_top_sum:,}")
        print(f"  Top1-20总和(新): {new_top_sum:,}")
        print(f"  Outflow_Count(新): {new_outflow:,}")
        print(f"  差异: {new_outflow - new_top_sum:,}")

    # 关闭连接
    conn_old.close()
    conn_new.close()

    print(f"\n{'=' * 100}")
    print("对比完成！")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    compare_databases()



'''