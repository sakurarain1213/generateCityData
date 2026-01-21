"""
优化版本的数据生成器 - 解决多进程卡住和DuckDB效率问题
(已修复 Windows GBK 编码报错问题)
"""

import os
import duckdb
import pandas as pd
import subprocess
import sys
import traceback
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Queue, Process
import time
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from synthesis.config import OUTPUT_DIR, OUTPUT_FILENAME

# 定义本地数据库文件路径
DB_FILE = os.path.join(OUTPUT_DIR, 'local_migration_data.db')
SAMPLE_CSV_FILE = os.path.join(OUTPUT_DIR, 'migration_sample_optimized.csv')

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_main_program():
    """运行主程序生成CSV数据（增加超时控制）"""
    # 首先检查CSV文件是否已存在
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if os.path.exists(csv_path):
        print(f"发现现有CSV文件: {csv_path}")
        file_size = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"文件大小: {file_size:.1f} MB")
        print("跳过数据生成步骤，直接使用现有数据...")
        return
        
    print("开始运行主程序生成数据...")
    try:
        # 修复路径问题
        result = subprocess.run(
            [sys.executable, "synthesis/main_optimized.py"], 
            check=True, 
            timeout=3600,  # 1小时超时
            capture_output=True, 
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))  # 回到项目根目录
        )
        print("主程序运行完成，CSV数据已生成。")
        if result.stdout:
            print("程序输出:", result.stdout[-1000:])  # 只显示最后1000字符
    except subprocess.TimeoutExpired:
        print("[ERROR] 主程序运行超时（1小时），可能数据量过大或程序卡住")
        raise
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 运行主程序时出错: {e}")
        if e.stderr:
            print("错误输出:", e.stderr[-1000:])
        raise

def load_data():
    """加载生成的CSV数据"""
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到生成的CSV文件: {csv_path}，请先运行主程序生成数据。")
    
    print(f"开始加载数据文件: {csv_path}")
    # 使用分块读取，避免内存不足
    chunks = []
    chunk_size = 100000  # 每次读取10万行
    
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunk_size), desc="加载数据"):
        chunks.append(chunk)
    
    data = pd.concat(chunks, ignore_index=True)
    print(f"数据加载完成，共 {len(data):,} 行")
    return data

def save_to_duckdb_optimized(data):
    """
    优化的DuckDB保存方法（单线程，避免锁竞争）
    """
    print(f"开始保存数据到DuckDB: {DB_FILE}")
    
    # 删除旧数据库文件
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    
    # 创建新连接并一次性保存所有数据
    conn = duckdb.connect(DB_FILE)
    
    try:
        # 直接从pandas DataFrame创建表（最高效的方式）
        conn.execute("CREATE TABLE migration_data AS SELECT * FROM data")
        conn.register('data', data)
        
        # 验证数据
        count = conn.execute("SELECT COUNT(*) FROM migration_data").fetchone()[0]
        print(f"[SUCCESS] 数据保存成功，共 {count:,} 行")
        
        # 创建索引优化查询性能
        print("创建索引...")
        conn.execute("CREATE INDEX idx_type_id ON migration_data(Type_ID)")
        conn.execute("CREATE INDEX idx_city ON migration_data(From_City)")
        conn.execute("CREATE INDEX idx_year_month ON migration_data(Year, Month)")
        
        # --- 修复点：替换 Emoji ---
        print("[OK] 索引创建完成")
        
    except Exception as e:
        print(f"[ERROR] DuckDB保存失败: {e}")
        raise
    finally:
        conn.close()

def save_to_duckdb_batch(data, batch_size=200000):
    """
    分批保存到DuckDB（适用于超大数据集）
    """
    print(f"使用分批方式保存数据到DuckDB，批大小: {batch_size:,}")
    
    # 删除旧数据库文件
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    
    conn = duckdb.connect(DB_FILE)
    total_saved = 0
    
    try:
        # 第一批：创建表
        first_batch = data.iloc[:batch_size]
        conn.register('data', first_batch)
        conn.execute("CREATE TABLE migration_data AS SELECT * FROM data")
        total_saved += len(first_batch)
        print(f"创建表并插入第一批数据: {len(first_batch):,} 行")
        
        # 后续批次：插入数据
        for i in tqdm(range(batch_size, len(data), batch_size), desc="批量插入"):
            batch = data.iloc[i:i+batch_size]
            conn.register('data_batch', batch)
            conn.execute("INSERT INTO migration_data SELECT * FROM data_batch")
            total_saved += len(batch)
        
        # 验证数据
        count = conn.execute("SELECT COUNT(*) FROM migration_data").fetchone()[0]
        print(f"[SUCCESS] 批量保存完成，共 {count:,} 行")
        
        # 创建索引
        print("创建索引...")
        conn.execute("CREATE INDEX idx_type_id ON migration_data(Type_ID)")
        conn.execute("CREATE INDEX idx_city ON migration_data(From_City)")
        conn.execute("CREATE INDEX idx_year_month ON migration_data(Year, Month)")
        
        # --- 修复点：替换 Emoji ---
        print("[OK] 索引创建完成")
        
    except Exception as e:
        print(f"[ERROR] 批量保存失败: {e}")
        raise
    finally:
        conn.close()

def generate_sample_csv():
    """从DuckDB中采样100条数据并保存为CSV"""
    try:
        conn = duckdb.connect(DB_FILE)
        
        # 随机采样100行
        sample_data = conn.execute(
            "SELECT * FROM migration_data USING SAMPLE 100"
        ).fetchdf()
        
        sample_data.to_csv(SAMPLE_CSV_FILE, index=False, encoding='utf-8-sig')
        conn.close()
        
        print(f"[SUCCESS] 采样数据已保存到CSV文件: {SAMPLE_CSV_FILE} (共 {len(sample_data)} 行)")
        
    except Exception as e:
        print(f"[ERROR] 生成采样CSV失败: {e}")
        raise

def show_statistics():
    """显示数据库统计信息"""
    try:
        conn = duckdb.connect(DB_FILE)
        
        # 基本统计
        total_count = conn.execute("SELECT COUNT(*) FROM migration_data").fetchone()[0]
        unique_types = conn.execute("SELECT COUNT(DISTINCT Type_ID) FROM migration_data").fetchone()[0]
        unique_cities = conn.execute("SELECT COUNT(DISTINCT From_City) FROM migration_data").fetchone()[0]
        total_population = conn.execute("SELECT SUM(Total_Count) FROM migration_data").fetchone()[0]
        
        print("\n=== 数据库统计信息 ===")
        print(f"总记录数: {total_count:,}")
        print(f"唯一Type数: {unique_types}")
        print(f"唯一城市数: {unique_cities}")
        print(f"总人口数: {total_population:,}")
        
        # Top 10 城市（按人口）
        top_cities = conn.execute("""
            SELECT From_City, SUM(Total_Count) as total_pop 
            FROM migration_data 
            GROUP BY From_City 
            ORDER BY total_pop DESC 
            LIMIT 10
        """).fetchdf()
        
        print("\n=== Top 10 城市（按总人口） ===")
        print(top_cities.to_string(index=False))
        
        conn.close()
        
    except Exception as e:
        print(f"[ERROR] 显示统计信息失败: {e}")

def main():
    """主函数"""
    print("=== 优化版人口迁移数据生成和数据库构建工具 ===")
    start_time = time.time()
    
    try:
        # 步骤1：运行主程序生成CSV数据
        print("\n步骤1: 运行主程序生成CSV数据")
        run_main_program()

        # 步骤2：加载数据
        print("\n步骤2: 加载生成的数据")
        data = load_data()
        
        # 步骤3：保存到DuckDB（根据数据大小选择策略）
        print("\n步骤3: 保存数据到DuckDB")
        if len(data) > 500000:  # 大于50万行使用分批保存
            save_to_duckdb_batch(data)
        else:
            save_to_duckdb_optimized(data)

        # 步骤4：生成采样CSV
        print("\n步骤4: 生成采样数据")
        generate_sample_csv()
        
        # 步骤5：显示统计信息
        print("\n步骤5: 显示统计信息")
        show_statistics()
        
        elapsed_time = time.time() - start_time
        print(f"\n[COMPLETE] 全部任务完成！总耗时: {elapsed_time:.2f} 秒")
        
        # 清理内存
        del data
        
    except Exception as e:
        print(f"\n[ERROR] 程序执行失败: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()