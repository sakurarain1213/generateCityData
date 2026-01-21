"""
优化版本的主程序 - 修复多进程卡住问题
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback
import warnings
warnings.filterwarnings('ignore')

from config import CITIES, OUTPUT_YEARS, OUTPUT_MONTHS, TOP_N_CITIES, OUTPUT_DIR, OUTPUT_FILENAME
from type_generator import generate_all_types, type_to_id
from migration_model import MigrationModel
from population_distribution import generate_type_counts, filter_valid_types
from population_distribution import calculate_city_type_count


def process_type_batch_optimized(
    type_ids_batch: List[str], 
    type_id_to_dict: Dict, 
    valid_type_counts: Dict
) -> List[Dict]:
    """
    优化版子进程工作函数：处理一批Type的数据生成
    """
    try:
        # 每个进程独立实例化模型，避免跨进程共享状态问题
        local_migration_model = MigrationModel()
        batch_rows = []
        
        # 提前计算一些常量，避免重复计算
        cities_list = list(CITIES)
        
        for type_id in type_ids_batch:
            if type_id not in type_id_to_dict:
                continue
            
            type_dict = type_id_to_dict[type_id]
            global_type_count = valid_type_counts[type_id]
            
            # 为每个城市生成数据
            for city_code, city_name in cities_list:
                # 为每个年月生成一行
                for year in OUTPUT_YEARS:
                    for month in OUTPUT_MONTHS:
                        try:
                            # 1. 计算人口数
                            # 使用确定性Hash作为种子，保证并行计算结果与单线程一致
                            row_count_seed = hash(f"{type_id}_{city_code}_{year}_{month}") % (2**31)
                            count_rng = np.random.RandomState(row_count_seed)
                            
                            total_count = calculate_city_type_count(
                                type_id, city_code, global_type_count, 
                                year=year, month=month, rng=count_rng
                            )

                            # 跳过人口为0的记录，减少数据量
                            if total_count == 0:
                                continue

                            # 2. 计算迁移概率
                            # 同样使用确定性Hash种子
                            row_seed = hash(f"{year}_{month}_{type_id}_{city_code}") % (2**31)
                            
                            # 计算基础迁移概率
                            migration_prob = local_migration_model.calculate_base_migration_prob(
                                type_dict, month=month, year=year, from_city_code=city_code, noise=True
                            )
                            stay_prob = local_migration_model.calculate_stay_prob(migration_prob)
                            
                            # 计算迁移目标
                            migration_targets = local_migration_model.calculate_migration_targets(
                                city_code, type_dict, migration_prob, month=month, year=year, row_seed=row_seed
                            )
                            
                            # 3. 构建行数据
                            row = {
                                'Year': year,
                                'Month': month,
                                'Type_ID': type_id,
                                'From_City': f"{city_name}({city_code})",
                                'Total_Count': total_count,
                                'Stay_Prob': round(stay_prob, 4)
                            }
                            
                            # 处理Top N目标
                            other_prob = 0.0
                            city_targets = []
                            for city_code_target, city_name_target, prob in migration_targets:
                                if city_code_target == 'Other':
                                    other_prob = prob
                                else:
                                    city_targets.append((city_code_target, city_name_target, prob))
                            
                            for i in range(TOP_N_CITIES):
                                if i < len(city_targets):
                                    t_code, t_name, t_prob = city_targets[i]
                                    row[f'To_Top{i+1}'] = f"{t_name}({t_code})"
                                    row[f'To_Top{i+1}_Prob'] = round(t_prob, 4)
                                else:
                                    row[f'To_Top{i+1}'] = ''
                                    row[f'To_Top{i+1}_Prob'] = 0.0
                            
                            row['To_Other_Prob'] = round(other_prob, 4)
                            batch_rows.append(row)
                            
                        except Exception as e:
                            print(f"处理单行数据时出错 (Type: {type_id}, City: {city_code}): {e}")
                            continue
                            
        return batch_rows
        
    except Exception as e:
        print(f"批处理函数出错: {e}")
        traceback.print_exc()
        return []


def generate_synthesis_data_optimized() -> pd.DataFrame:
    """
    优化版本的数据生成函数
    """
    print("开始生成合成数据 (优化多进程版本)...")
    start_time = time.time()
    
    # 1. 生成Type和基础分布
    print("1. 准备Type和基础分布...")
    types = generate_all_types()
    print(f"   总Type数: {len(types)}")
    
    # 生成Type计数（全局）
    type_counts = generate_type_counts(types, type_to_id)
    
    # 过滤出有效的Type
    valid_type_counts = filter_valid_types(type_counts)
    print(f"   有效Type数: {len(valid_type_counts)}")
    
    if len(valid_type_counts) == 0:
        print("警告：没有有效的Type，请检查配置")
        return pd.DataFrame()
    
    # 2. 准备数据字典
    type_id_to_dict = {type_to_id(t): t for t in types}
    
    # 3. 优化的并行生成数据
    valid_type_ids = list(valid_type_counts.keys())
    
    # 计算分块策略：每个进程处理适当数量的Type
    max_workers = max(1, min(cpu_count() - 1, 20))  # 增加到最多20个进程
    chunk_size = max(1, len(valid_type_ids) // (max_workers * 4))  # 创建更多更小的块，提高并行度
    
    print(f"2. 启动进程池 (使用 {max_workers} 个核心，每块 {chunk_size} 个Type)...")
    
    all_rows = []
    chunks = [valid_type_ids[i:i + chunk_size] for i in range(0, len(valid_type_ids), chunk_size)]
    print(f"   共分成 {len(chunks)} 个块")
    
    # 使用上下文管理器和超时控制
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_chunk = {
            executor.submit(
                process_type_batch_optimized, 
                chunk, 
                type_id_to_dict, 
                valid_type_counts
            ): chunk for chunk in chunks
        }
        
        # 进度条改为显示块的进度
        successful_chunks = 0
        with tqdm(total=len(chunks), desc="并行生成中", unit="块") as pbar:
            for future in as_completed(future_to_chunk):
                try:
                    chunk_rows = future.result(timeout=600)  # 10分钟超时
                    
                    if chunk_rows:  # 确保不是空结果
                        all_rows.extend(chunk_rows)
                        successful_chunks += 1
                    
                except Exception as e:
                    chunk = future_to_chunk[future]
                    print(f"\n[ERROR] 处理块失败 ({len(chunk)} 个Type): {e}")
                    # 继续处理其他块，不中断整个流程
                finally:
                    # 无论成功失败，都更新进度条
                    pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\n数据生成完成！")
    print(f"成功处理块数: {successful_chunks}/{len(chunks)}")
    print(f"耗时: {elapsed_time:.2f} 秒")
    print(f"生成记录数: {len(all_rows):,}")
    if all_rows:
        print(f"平均速度: {len(all_rows) / elapsed_time:.0f} 行/秒")

    # 4. 转换为DataFrame
    print("3. 转换为DataFrame...")
    if not all_rows:
        print("警告：没有生成任何数据")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    return df


def save_output_optimized(df: pd.DataFrame, output_file: Optional[str] = None):
    """优化版保存输出数据"""
    if output_file is None:
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = str(output_path / OUTPUT_FILENAME)
    else:
        output_path = Path(output_file).parent
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存数据到 {output_file}...")
    
    try:
        # 分块保存CSV，避免内存问题
        chunk_size = 100000
        if len(df) > chunk_size:
            df.to_csv(output_file, index=False, encoding='utf-8-sig', chunksize=chunk_size)
        else:
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"[SUCCESS] CSV保存完成！共 {len(df):,} 行数据")
        
        # 如果数据量不太大，同时保存Excel
        if len(df) < 300_000:
            excel_file = output_file.replace('.csv', '.xlsx')
            print(f"同时保存Excel格式到 {excel_file}...")
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                # 分工作表保存，避免单个工作表行数限制
                max_rows_per_sheet = 1000000
                if len(df) <= max_rows_per_sheet:
                    df.to_excel(writer, sheet_name='migration_data', index=False)
                else:
                    # 分多个工作表
                    for i in range(0, len(df), max_rows_per_sheet):
                        sheet_name = f'data_part_{i//max_rows_per_sheet + 1}'
                        df.iloc[i:i+max_rows_per_sheet].to_excel(writer, sheet_name=sheet_name, index=False)
            print("[SUCCESS] Excel保存完成！")
        else:
            print("数据量超过30万行，跳过Excel生成以节省时间。")
            
    except Exception as e:
        print(f"[ERROR] 保存文件时出错: {e}")
        raise


def main():
    """主函数"""
    print("=== 优化版人口迁移数据合成系统 ===")
    
    if __name__ != '__main__':
        print("[ERROR] 请直接运行此脚本，不要作为模块导入")
        return
        
    try:
        df = generate_synthesis_data_optimized()
        
        if df.empty:
            print("[ERROR] 没有生成任何数据，请检查配置")
            return
        
        print("\n=== 数据统计 ===")
        print(f"总行数: {len(df):,}")
        print(f"唯一Type数: {df['Type_ID'].nunique()}")
        print(f"唯一城市数: {df['From_City'].nunique()}")
        print(f"总人口数: {df['Total_Count'].sum():,}")
        
        save_output_optimized(df)
        
        print("\n[COMPLETE] 全部完成！")
        
    except Exception as e:
        print(f"[ERROR] 程序执行失败: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()