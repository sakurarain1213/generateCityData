"""
一键运行脚本 - 解决多进程卡住问题，快速生成大规模人口迁移数据
"""

import os
import sys
import time
import subprocess

def run_command(command, description, timeout=3600):
    """
    运行命令 - 直接将输出流向控制台，确保进度条可见
    """
    print(f"\n🚀 {description}...")
    print(f"命令: {' '.join(command)}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # 【核心修改】
        # 不使用 stdout=subprocess.PIPE，直接让子进程输出到屏幕
        # 这样 tqdm 进度条就能正常刷新了，不会被缓冲卡住
        result = subprocess.run(
            command,
            check=False,  # 允许非0退出码，手动处理
            timeout=timeout,
            cwd=os.path.dirname(__file__)
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print("-" * 50)
            print(f"✅ {description} 完成，耗时: {elapsed:.2f} 秒")
            return True
        else:
            print("-" * 50)
            print(f"❌ {description} 失败，返回代码: {result.returncode}")
            return False
                
    except subprocess.TimeoutExpired:
        print(f"\n❌ {description} 超时（{timeout}秒）")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        return False
    except Exception as e:
        print(f"\n❌ 运行 {description} 时出现未知错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🏙️  经济学模拟 - 城市人口迁移数据生成工具")
    print("📊  大规模数据生成 + 多进程优化 + DuckDB存储")
    print("=" * 60)
    print("⚙️  配置信息:")
    print("   • 人口基数: 500万")
    print("   • 多进程优化: 启用")
    print("   • 超时设置: 数据生成30分钟，数据库20分钟")
    print("   • 输出格式: CSV + Excel + DuckDB")
    print("")
    
    start_time = time.time()
    
    # 步骤1：使用优化版本生成合成数据
    print("📊 开始数据生成流程...")
    success1 = run_command(
        [sys.executable, "synthesis/main_optimized.py"],
        "步骤1: 生成优化版合成数据（500万人口基数）",
        timeout=1800  # 30分钟超时
    )
    
    if not success1:
        print("\n⚠️ 数据生成失败，尝试使用原始版本...")
        success1 = run_command(
            [sys.executable, "synthesis/main.py"],
            "备用方案: 使用原始版本生成数据",
            timeout=2400  # 40分钟超时
        )
    
    if not success1:
        print("\n❌ 所有数据生成方案都失败了")
        return False
    
    # 步骤2：构建数据库和采样
    print("\n🗄️ 开始数据库构建流程...")
    success2 = run_command(
        [sys.executable, "local_db/optimized_data_generator.py"],
        "步骤2: 构建优化版DuckDB数据库",
        timeout=1200  # 20分钟超时
    )
    
    if not success2:
        print("\n⚠️ 数据库构建失败，尝试原始版本...")
        success2 = run_command(
            [sys.executable, "local_db/local_data_generator.py"],
            "备用方案: 使用原始版本构建数据库",
            timeout=1800  # 30分钟超时
        )
    
    total_time = time.time() - start_time
    
    if success1 and success2:
        print("\n" + "=" * 60)
        print("🎉 全部任务完成！")
        print(f"⏱️  总耗时: {total_time:.2f} 秒 ({total_time/60:.1f} 分钟)")
        print("\n📁 生成的文件:")
        
        # 检查生成的文件
        output_dir = "output"
        files_to_check = [
            "migration_data.csv",
            "migration_data.xlsx", 
            "local_migration_data.db",
            "migration_sample_optimized.csv"
        ]
        
        for filename in files_to_check:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"  ✅ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"  ❌ {filename} (未找到)")
        
        print("\n📊 使用说明:")
        print("  1. 查看 migration_data.csv 获取完整数据")
        print("  2. 查看 migration_sample_optimized.csv 获取数据样本")
        print("  3. 使用 DuckDB 查询 local_migration_data.db 进行分析")
        
        return True
    else:
        print("\n❌ 部分任务失败")
        print(f"⏱️  总耗时: {total_time:.2f} 秒")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

