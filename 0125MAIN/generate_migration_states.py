"""
Migration State 生成器 - 强约束平衡修复版 (Fix: Overwrite Bug)
功能：
1. 读取2000年数据作为总目标，生成初始人口分布。
2. 硬约束：每个Type至少60%人口锁死在来源城市。
3. 软约束：剩余人口根据各城市的"人口缺口"进行分配。
4. 修复：解决字典更新覆盖导致流动人口丢失的问题。
5. [新增] 遍历DB中所有年份，将2000年之后出现的Type追加到JSON中（人口设为0）。

数据来源说明：
- 2000年的Type：使用真实人口数据进行初始化分配
- 2001-2014年的Type：仅记录Type定义，城市人口全部设为0（模拟过程中后期激活）
"""

import os
import json
import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================================
# 配置区域
# ============================================================================

OUTPUT_DIR = 'output'
DB_FILE = os.path.join(OUTPUT_DIR, 'local_migration_data.db')
MIGRATION_STATES_FILE = os.path.join(OUTPUT_DIR, 'migration_states.jsonl')
TYPE_FEATURES_FILE = os.path.join(OUTPUT_DIR, 'type_features.jsonl')

INITIAL_YEAR = 2000

# 【核心参数】本地留存率
# 0.6 表示：强制60%留在本地，剩余40%参与全国流动
LOCAL_RETENTION_RATE = 0.6 

# ============================================================================
# 主函数
# ============================================================================

def generate_migration_states_from_db():
    print("="*80)
    print(f"Migration State 生成器 (修复版)")
    print(f"策略: 本地锁死 {LOCAL_RETENTION_RATE*100}% + 缺口权重回填")
    print("="*80)

    if not os.path.exists(DB_FILE):
        print(f"[错误] 数据库不存在: {DB_FILE}")
        return

    # 1. 读取数据库目标
    print(f"[1/6] 读取数据库数据...")
    conn = duckdb.connect(DB_FILE)

    # 2000年的 Type 总数 (作为源头，用于初始化人口)
    df_types_2000 = conn.execute(f"""
        SELECT Type_ID, SUM(Total_Count) as total_population
        FROM migration_data WHERE Year = {INITIAL_YEAR}
        GROUP BY Type_ID
    """).df()

    # [新增] 获取所有年份的所有唯一 Type_ID
    df_all_types = conn.execute("""
        SELECT DISTINCT Type_ID FROM migration_data ORDER BY Type_ID ASC
    """).df()
    all_type_ids_in_db = set(df_all_types['Type_ID'].tolist())
    print(f"    发现 {len(all_type_ids_in_db)} 个唯一的 Type_ID (所有年份)")

    # 城市 总数 (作为目标靶子)
    df_cities = conn.execute(f"""
        SELECT Region as city_code, SUM(Total_Count) as target_pop
        FROM migration_data WHERE Year = {INITIAL_YEAR}
        GROUP BY Region
    """).df()
    conn.close()

    if df_types_2000.empty:
        print("[错误] 未找到数据。")
        return

    # 2. 初始化数据结构
    print(f"[2/6] 初始化计算矩阵...")

    # 建立数据库Type的查找表 (只包含2000年的Type及其人口数据)
    db_type_map = dict(zip(df_types_2000['Type_ID'], df_types_2000['total_population']))
    
    # 城市目标字典
    city_targets = dict(zip(df_cities['city_code'].astype(str), df_cities['target_pop']))
    all_cities_list = list(city_targets.keys())
    n_cities = len(all_cities_list)
    
    # 城市索引映射 (用于快速查找)
    city_to_idx = {code: i for i, code in enumerate(all_cities_list)}

    # 3. 构建最终的 Type 列表
    # 合并：2000年有数据的Type + 2000年之后出现的Type
    print(f"[3/6] 构建完整的 Type 列表...")
    final_type_list = sorted(all_type_ids_in_db)

    # 统计信息
    types_2000 = set(db_type_map.keys())
    types_after_2000 = all_type_ids_in_db - types_2000
    print(f"    2000年有数据的Type: {len(types_2000)} 个")
    print(f"    2000年之后新增的Type: {len(types_after_2000)} 个")
    print(f"    总计Type数量: {len(final_type_list)} 个")

    # 4. 第一轮：计算本地留存 & 确定各城市的人口缺口
    print(f"[4/6] 第一轮分配：执行本地留存 ({LOCAL_RETENTION_RATE*100}%) ...")
    
    type_local_allocations = {} # {type_id: {city_code: count}}
    current_city_fill = {c: 0.0 for c in all_cities_list}
    type_floating_pop = {} # {type_id: count}

    for type_id in final_type_list:
        if type_id in db_type_map:
            total_count = int(db_type_map[type_id])
            
            # 解析来源城市
            try:
                home_city = type_id.split('_')[-1]
            except:
                home_city = None
            
            # 计算留存
            local_count = 0
            if home_city and home_city in city_targets:
                local_count = int(total_count * LOCAL_RETENTION_RATE)
                # 记录分配
                type_local_allocations[type_id] = {home_city: local_count}
                # 更新城市当前填充量
                current_city_fill[home_city] += local_count
            else:
                type_local_allocations[type_id] = {}
            
            # 剩余的人去流浪
            type_floating_pop[type_id] = total_count - local_count

    # 5. 计算缺口权重 (Deficit Weights)
    print(f"[5/6] 计算城市人口缺口权重...")
    
    city_deficits = []
    total_deficit = 0
    
    for city in all_cities_list:
        target = city_targets[city]
        current = current_city_fill.get(city, 0)
        deficit = max(0, target - current) # 缺口
        city_deficits.append(deficit)
        total_deficit += deficit
        
    city_deficits = np.array(city_deficits)
    
    # 归一化权重
    if total_deficit > 0:
        fill_weights = city_deficits / total_deficit
    else:
        fill_weights = np.ones(n_cities) / n_cities

    # 6. 生成最终文件
    print(f"[6/6] 生成最终文件并写入...")
    
    os.makedirs(os.path.dirname(MIGRATION_STATES_FILE), exist_ok=True)
    rng = np.random.RandomState(42)
    
    with open(MIGRATION_STATES_FILE, 'w', encoding='utf-8') as f:
        for type_id in tqdm(final_type_list, desc="Generating"):
            
            final_city_pop = {}
            
            if type_id in db_type_map:
                # 1. 分配流动人口 (Scatter)
                # 先做 scatter，这样后续加 local 时不会被覆盖
                floating_count = type_floating_pop.get(type_id, 0)
                
                if floating_count > 0:
                    scatter_counts = rng.multinomial(floating_count, fill_weights)
                    nonzero_indices = np.nonzero(scatter_counts)[0]
                    for idx in nonzero_indices:
                        city_code = all_cities_list[idx]
                        final_city_pop[city_code] = int(scatter_counts[idx])
                
                # 2. 累加本地锁死的人口 (Local)
                # [关键修复] 使用累加逻辑，而不是 update 覆盖
                if type_id in type_local_allocations:
                    for city_code, count in type_local_allocations[type_id].items():
                        if count > 0:
                            final_city_pop[city_code] = final_city_pop.get(city_code, 0) + count

                # 3. 兜底校验（修正舍入误差）
                # 确保 sum(final) == db_total
                actual_total = sum(final_city_pop.values())
                target_total = int(db_type_map[type_id])
                
                if actual_total != target_total:
                    diff = target_total - actual_total
                    # 把误差补给来源城市，或者最大城市
                    target_city = type_id.split('_')[-1]
                    if target_city in final_city_pop:
                        final_city_pop[target_city] += diff
                    elif final_city_pop:
                        max_city = max(final_city_pop, key=final_city_pop.get)
                        final_city_pop[max_city] += diff
            
            else:
                # 没数据的Type
                final_city_pop = {c: 0 for c in all_cities_list}

            # 写入
            record = {
                "id": type_id,
                "city_population": final_city_pop
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n[完成] 文件已生成: {MIGRATION_STATES_FILE}")
    
    # ========================================================================
    # 快速自检
    # ========================================================================
    print("\n[自检] 验证前10个重点城市人口吻合度...")
    
    file_pop_check = {}
    with open(MIGRATION_STATES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            for c, p in rec['city_population'].items():
                file_pop_check[c] = file_pop_check.get(c, 0) + p
    
    check_cities = ['1100', '3100', '4401', '4403', '5000', '1200', '3301']
    print(f"{'City':<10} | {'Target (DB)':<15} | {'Generated':<15} | {'Diff %':<10}")
    print("-" * 60)
    
    for c in check_cities:
        target = int(city_targets.get(c, 0))
        generated = int(file_pop_check.get(c, 0))
        if target > 0:
            diff = (generated - target) / target * 100
        else:
            diff = 0
        
        tag = ""
        if abs(diff) < 2: tag = "[OK]"
        elif abs(diff) < 5: tag = "[GOOD]"
        else: tag = "[WARN]"
            
        print(f"{c:<10} | {target:<15,} | {generated:<15,} | {diff:+.2f}% {tag}")

if __name__ == "__main__":
    generate_migration_states_from_db()