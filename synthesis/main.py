# """
# 人口迁徙预测数据合成系统 - 主程序 (多进程优化版)
# """

# import pandas as pd
# import numpy as np
# import os
# from pathlib import Path
# from typing import List, Dict, Optional
# from tqdm import tqdm
# import time
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from multiprocessing import cpu_count
# import traceback

# from config import CITIES, OUTPUT_YEARS, OUTPUT_MONTHS, TOP_N_CITIES, OUTPUT_DIR, OUTPUT_FILENAME
# from type_generator import generate_all_types, type_to_id
# from migration_model import MigrationModel
# from population_distribution import generate_type_counts, filter_valid_types
# from population_distribution import calculate_city_type_count


# def process_type_batch(
#     type_ids_batch: List[str], 
#     type_id_to_dict: Dict, 
#     valid_type_counts: Dict
# ) -> List[Dict]:
#     """
#     子进程工作函数：处理一批Type的数据生成
#     """
#     try:
#         # 每个进程独立实例化模型，避免跨进程共享状态问题
#         local_migration_model = MigrationModel()
#         batch_rows = []

#         for type_id in type_ids_batch:
#             if type_id not in type_id_to_dict:
#                 continue
            
#             type_dict = type_id_to_dict[type_id]
#             global_type_count = valid_type_counts[type_id]
            
#             # 为每个城市生成数据
#             for city_code, city_name in CITIES:
#                 # 为每个年月生成一行
#                 for year in OUTPUT_YEARS:
#                     for month in OUTPUT_MONTHS:
#                         # 1. 计算人口数
#                         # 使用确定性Hash作为种子，保证并行计算结果与单线程一致
#                         row_count_seed = hash(f"{type_id}_{city_code}_{year}_{month}") % (2**31)
#                         count_rng = np.random.RandomState(row_count_seed)
                        
#                         total_count = calculate_city_type_count(
#                             type_id, city_code, global_type_count, 
#                             year=year, month=month, rng=count_rng
#                         )

#                         # 2. 计算迁移概率
#                         # 同样使用确定性Hash种子
#                         row_seed = hash(f"{year}_{month}_{type_id}_{city_code}") % (2**31)
                        
#                         # 计算基础迁移概率
#                         migration_prob = local_migration_model.calculate_base_migration_prob(
#                             type_dict, month=month, year=year, from_city_code=city_code, noise=True
#                         )
#                         stay_prob = local_migration_model.calculate_stay_prob(migration_prob)
                        
#                         # 计算迁移目标
#                         migration_targets = local_migration_model.calculate_migration_targets(
#                             city_code, type_dict, migration_prob, month=month, year=year, row_seed=row_seed
#                         )
                        
#                         # 3. 构建行数据
#                         row = {
#                             'Year': year,
#                             'Month': month,
#                             'Type_ID': type_id,
#                             'From_City': f"{city_name}({city_code})",
#                             'Total_Count': total_count,
#                             'Stay_Prob': round(stay_prob, 4)
#                         }
                        
#                         # 处理Top N目标
#                         other_prob = 0.0
#                         city_targets = []
#                         for city_code_target, city_name_target, prob in migration_targets:
#                             if city_code_target == 'Other':
#                                 other_prob = prob
#                             else:
#                                 city_targets.append((city_code_target, city_name_target, prob))
                        
#                         for i in range(TOP_N_CITIES):
#                             if i < len(city_targets):
#                                 t_code, t_name, t_prob = city_targets[i]
#                                 row[f'To_Top{i+1}'] = f"{t_name}({t_code})"
#                                 row[f'To_Top{i+1}_Prob'] = round(t_prob, 4)
#                             else:
#                                 row[f'To_Top{i+1}'] = ''
#                                 row[f'To_Top{i+1}_Prob'] = 0.0
                        
#                         row['Other'] = round(other_prob, 4)
#                         batch_rows.append(row)
        
#         return batch_rows
#     except Exception as e:
#         print(f"子进程处理错误: {e}")
#         traceback.print_exc()
#         return []


# def generate_synthesis_data() -> pd.DataFrame:
#     """
#     生成合成数据 (多进程并行版)
#     """
#     print("开始生成合成数据 (多进程加速模式)...")
    
#     # 1. 基础数据准备
#     print("1. 准备Type和基础分布...")
#     all_types = generate_all_types()
#     type_counts_dict = generate_type_counts(all_types, type_to_id)
#     valid_type_counts = filter_valid_types(type_counts_dict)
    
#     type_id_to_dict = {type_to_id(t): t for t in all_types}
#     valid_type_ids = list(valid_type_counts.keys())
    
#     print(f"   总Type数: {len(all_types)}")
#     print(f"   有效Type数: {len(valid_type_ids)}")

#     # 2. 准备并行任务
#     # 根据CPU核心数决定进程数，保留1个核心给系统
#     max_workers = max(1, cpu_count() - 1)
#     print(f"2. 启动进程池 (使用 {max_workers} 个核心)...")
    
#     # 将Type ID列表切分为多个批次
#     # 使用 numpy.array_split 进行均匀切分
#     if not valid_type_ids:
#         print("警告：没有有效的Type ID")
#         return pd.DataFrame()

#     chunks = np.array_split(valid_type_ids, max_workers) # 根据核心数切分
#     # 将 numpy array 转换回 list，避免序列化问题
#     chunks = [chunk.tolist() for chunk in chunks if len(chunk) > 0]
    
#     all_rows = []
#     start_time = time.time()
    
#     # 3. 执行并行计算
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         # 提交任务
#         futures = {
#             executor.submit(
#                 process_type_batch, 
#                 chunk, 
#                 type_id_to_dict, 
#                 valid_type_counts
#             ): len(chunk) for chunk in chunks
#         }
        
#         # 进度条统计的是Type的数量，而不是行的数量（行数太多了）
#         total_types = len(valid_type_ids)
        
#         with tqdm(total=total_types, desc="并行生成中", unit="type") as pbar:
#             for future in as_completed(futures):
#                 try:
#                     chunk_rows = future.result()
#                     all_rows.extend(chunk_rows)
                    
#                     # 更新进度条（增加当前batch处理的type数量）
#                     processed_count = futures[future]
#                     pbar.update(processed_count)
                    
#                 except Exception as e:
#                     print(f"子进程发生错误: {e}")
#                     traceback.print_exc()
#                 else:
#                     print(f"成功处理了 {futures[future]} 个Type")

#     elapsed_time = time.time() - start_time
#     print(f"\n数据生成完成！耗时: {elapsed_time:.2f} 秒")
#     print(f"平均速度: {len(all_rows) / elapsed_time:.0f} 行/秒")

#     # 4. 转换为DataFrame
#     print("3. 转换为DataFrame...")
#     df = pd.DataFrame(all_rows)
    
#     return df


# def save_output(df: pd.DataFrame, output_file: Optional[str] = None):
#     """保存输出数据"""
#     if output_file is None:
#         output_path = Path(OUTPUT_DIR)
#         output_path.mkdir(parents=True, exist_ok=True)
#         output_file = str(output_path / OUTPUT_FILENAME)
#     else:
#         output_path = Path(output_file).parent
#         output_path.mkdir(parents=True, exist_ok=True)
    
#     print(f"\n保存数据到 {output_file}...")
#     # 使用快一点的保存方式
#     df.to_csv(output_file, index=False, encoding='utf-8-sig', chunksize=100000)
#     print(f"保存完成！共 {len(df)} 行数据")
    
#     # 如果数据量过大，跳过Excel保存
#     if len(df) < 500_000:
#         excel_file = output_file.replace('.csv', '.xlsx')
#         print(f"同时保存Excel格式到 {excel_file}...")
#         df.to_excel(excel_file, index=False, engine='openpyxl')
#         print("Excel保存完成！")
#     else:
#         print("数据量超过50万行，跳过Excel生成以节省时间。")


# def main():
#     # 必须在 if __name__ == '__main__': 下运行，否则多进程在Windows下会报错
#     df = generate_synthesis_data()
    
#     print("\n=== 数据统计 ===")
#     print(f"总行数: {len(df):,}")
#     print(f"唯一Type数: {df['Type_ID'].nunique()}")
#     print(f"唯一城市数: {df['From_City'].nunique()}")
#     print(f"总人口数: {df['Total_Count'].sum():,}")
    
#     save_output(df)
    
#     print("\n全部完成！")


# if __name__ == '__main__':
#     main()
