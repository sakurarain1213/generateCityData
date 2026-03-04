import duckdb
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 用于显示负号

# cd "C:\Users\w1625\Desktop\db-generate\0302GenerateAgentJSON" ; python generate_jsonl.py

# 配置参数
THRESHOLD_K = 0  # 阈值，可以根据需要调整
TARGET_YEAR = 2000
DB_PATH = "local_migration_data.db"
OUTPUT_FILE1 = "type_features.jsonl"
OUTPUT_DIR2 = "migration_states"  # 输出文件夹


def parse_type_id(type_id):
    """
    解析 Type_ID，提取各个维度的特征
    格式示例: M_20_EduLo_Agri_IncL_Split_4403
    """
    parts = type_id.split('_')

    if len(parts) < 7:
        raise ValueError(f"Invalid Type_ID format: {type_id}")

    return {
        "id": type_id,
        "gender": parts[0],      # M/F
        "age": parts[1],         # 20, 30, etc.
        "edu": parts[2],         # EduHi, EduLo, etc.
        "job": parts[3],         # Agri, etc.
        "income": parts[4],      # IncH, IncL, IncM, etc.
        "family": parts[5],      # Split, Unit
        "dialect": parts[6]      # 地区代码，如 1100, 4403
    }


def generate_jsonl_files():
    """生成 type_features.jsonl 和逐年的 migration_states_YYYY.jsonl 文件"""

    # 连接数据库
    conn = duckdb.connect(DB_PATH)

    # 查询所有年份
    years_query = "SELECT DISTINCT Year FROM migration_data ORDER BY Year ASC"
    years = [row[0] for row in conn.execute(years_query).fetchall()]
    print(f"找到 {len(years)} 个年份: {years}")

    # 查询 2000 年且 Total_Count >= THRESHOLD_K 的唯一 Type_ID（升序排列）
    query = f"""
    SELECT DISTINCT Type_ID
    FROM migration_data
    WHERE Year = {TARGET_YEAR} AND Total_Count >= {THRESHOLD_K}
    ORDER BY Type_ID ASC
    """

    print(f"\n正在查询 {TARGET_YEAR} 年 Total_Count >= {THRESHOLD_K} 的数据...")
    type_ids = [row[0] for row in conn.execute(query).fetchall()]

    if not type_ids:
        print(f"警告：没有找到符合条件的数据（Year={TARGET_YEAR}, Total_Count>={THRESHOLD_K}）")
        conn.close()
        return

    print(f"找到 {len(type_ids)} 条唯一的 Type_ID 记录")

    # 创建输出文件夹
    output_dir = Path(OUTPUT_DIR2)
    output_dir.mkdir(exist_ok=True)
    print(f"\n创建输出文件夹: {output_dir}")

    # 生成文件1: type_features.jsonl
    print(f"\n正在生成 {OUTPUT_FILE1}...")
    with open(OUTPUT_FILE1, 'w', encoding='utf-8') as f1:
        for type_id in type_ids:
            try:
                features = parse_type_id(type_id)
                f1.write(json.dumps(features, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"处理 Type_ID {type_id} 时出错: {e}")
                continue

    # 为每个年份生成 migration_states_YYYY.jsonl
    for year in years:
        output_file = output_dir / f"migration_states_{year}.jsonl"
        print(f"\n正在生成 {output_file}...")

        # 查询该年份的数据
        year_query = f"""
        SELECT Type_ID, Total_Count
        FROM migration_data
        WHERE Year = {year} AND Type_ID IN ({','.join([f"'{tid}'" for tid in type_ids])})
        ORDER BY Type_ID ASC
        """

        year_results = conn.execute(year_query).fetchall()

        with open(output_file, 'w', encoding='utf-8') as f:
            for type_id, total_count in year_results:
                try:
                    features = parse_type_id(type_id)
                    city_code = features["dialect"]
                    migration_state = {
                        "id": type_id,
                        "city_population": {city_code: int(total_count)}
                    }
                    f.write(json.dumps(migration_state, ensure_ascii=False) + '\n')
                except Exception as e:
                    print(f"  处理 Type_ID {type_id} 时出错: {e}")
                    continue

        print(f"  完成，共 {len(year_results)} 条记录")

    conn.close()

    # 验证生成的文件
    print(f"\n{'='*60}")
    print(f"生成完成！")
    print(f"{'='*60}")
    print(f"\n文件1: {OUTPUT_FILE1}")
    with open(OUTPUT_FILE1, 'r', encoding='utf-8') as f:
        count1 = sum(1 for _ in f)
    print(f"  唯一 Type 数量: {count1}")

    print(f"\n文件夹: {OUTPUT_DIR2}/")
    for year in years:
        output_file = output_dir / f"migration_states_{year}.jsonl"
        with open(output_file, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        print(f"  migration_states_{year}.jsonl: {count} 条记录")

    # 显示前3条样例
    print(f"\n{OUTPUT_FILE1} 前3条样例:")
    with open(OUTPUT_FILE1, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  {line.strip()}")

    print(f"\nmigration_states_{years[0]}.jsonl 前3条样例:")
    first_year_file = output_dir / f"migration_states_{years[0]}.jsonl"
    with open(first_year_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(f"  {line.strip()}")


def plot_population_distribution():
    """绘制 2000 年人口分布的长尾图（降序柱状图）"""

    print(f"\n正在查询数据...")
    conn = duckdb.connect(DB_PATH)

    # 查询 2000 年且 Total_Count >= THRESHOLD_K 的数据，按人口降序排列
    query = f"""
    SELECT Type_ID, Total_Count
    FROM migration_data
    WHERE Year = {TARGET_YEAR} AND Total_Count >= {THRESHOLD_K}
    ORDER BY Total_Count DESC
    """

    results = conn.execute(query).fetchall()
    conn.close()

    if not results:
        print("没有数据可绘制")
        return

    print(f"查询完成，共 {len(results):,} 条数据")

    type_ids = [r[0] for r in results]
    populations = [r[1] for r in results]

    print(f"正在生成图形...")

    # 创建图形
    fig, ax = plt.subplots(figsize=(16, 8))

    # 使用 plot 而不是 bar，速度更快
    x_positions = range(len(populations))
    ax.fill_between(x_positions, populations, color='steelblue', alpha=0.7, linewidth=0)

    # 标注 Top 10
    top10_types = type_ids[:10]
    top10_pops = populations[:10]

    # 在图上标注 Top 10
    for i in range(min(10, len(top10_types))):
        # 简化 Type_ID 显示（只显示前30个字符）
        label = top10_types[i][:30] + '...' if len(top10_types[i]) > 30 else top10_types[i]
        ax.text(i, top10_pops[i], f'{i+1}',
                ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')

    # 设置标题和标签
    ax.set_title(f'{TARGET_YEAR}年人口分布（阈值>={THRESHOLD_K}，共{len(populations):,}个Type）',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Type排名（降序）', fontsize=12)
    ax.set_ylabel('人口数量', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 格式化 y 轴
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))

    # 添加 Top 10 信息文本框
    top10_text = "Top 10 Type:\n"
    for i in range(min(10, len(top10_types))):
        top10_text += f"{i+1}. {top10_types[i]}: {top10_pops[i]:,}\n"

    # 在图的右侧添加文本框
    ax.text(1.02, 0.98, top10_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            family='monospace')

    plt.tight_layout()

    # 保存图片
    output_image = f"population_distribution_{TARGET_YEAR}_k{THRESHOLD_K}.png"
    print(f"正在保存图片...")
    plt.savefig(output_image, dpi=150, bbox_inches='tight')
    print(f"[OK] 图片已保存: {output_image}")

    # 打印 Top 10 到控制台
    print(f"\n[OK] Top 10 人口数量的 Type:")
    for i in range(min(10, len(top10_types))):
        print(f"  {i+1}. {top10_types[i]}: {top10_pops[i]:,}")

    plt.close()
    print(f"[OK] 绘图完成！")


if __name__ == "__main__":
    generate_jsonl_files()
    plot_population_distribution()
