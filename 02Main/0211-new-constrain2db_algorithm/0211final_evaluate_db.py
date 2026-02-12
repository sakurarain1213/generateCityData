import pandas as pd
import numpy as np
import duckdb
import os
import sys
import re
import warnings

# 忽略 pandas 的一些警告
warnings.filterwarnings('ignore')

class DBValidator:
    def __init__(self, db_path, excel_dir):
        self.db_path = db_path
        self.excel_dir = excel_dir
        # 目标表名设定为 migration_data
        self.table_name = "migration_data"
        self.con = None

    def connect(self):
        if not os.path.exists(self.db_path):
            print(f"错误: 数据库文件不存在 {self.db_path}")
            sys.exit(1)
        
        # [关键修改 1] 必须设置为 read_only=False，否则无法修改表名
        self.con = duckdb.connect(self.db_path, read_only=False)
        
        # [关键修改 2] 连接后立即检查并修正表名
        self._fix_table_name_if_needed()

    def _fix_table_name_if_needed(self):
        """检查数据库中的表名，如果发现是旧名称 migration_records，则修正为 migration_data"""
        try:
            # 获取当前数据库中所有的表名
            tables = [t[0] for t in self.con.execute("SHOW TABLES").fetchall()]
            
            target_name = "migration_data"
            old_name = "migration_records"

            # 情况A: 目标表名已经存在 -> 无需操作
            if target_name in tables:
                # print(f"表名检查: 已存在标准表名 {target_name}")
                return

            # 情况B: 目标不存在，但旧表名存在 -> 执行重命名
            if old_name in tables:
                print(f"警告: 检测到旧表名 '{old_name}'。")
                print(f"正在自动修正为 '{target_name}'...")
                self.con.execute(f"ALTER TABLE {old_name} RENAME TO {target_name}")
                print(">>> 表名修正成功！")
            
            # 情况C: 都不存在 -> 留给后续 check_meta_info 报错
            
        except Exception as e:
            print(f"自动修正表名时发生错误: {e}")

    def close(self):
        if self.con:
            self.con.close()

    def check_meta_info(self):
        """1. 检查基础元数据：表名、行数、大小、Schema"""
        print("=" * 80)
        print(" [STEP 1] 数据库元数据检查")
        print("=" * 80)

        # 1. 文件大小
        size_bytes = os.path.getsize(self.db_path)
        size_mb = size_bytes / (1024 * 1024)
        print(f"数据库路径: {self.db_path}")
        print(f"占用空间  : {size_mb:.2f} MB")

        # 2. 表结构
        tables = self.con.execute("SHOW TABLES").fetchall()
        table_list = [t[0] for t in tables]
        print(f"包含表    : {table_list}")
        
        if self.table_name not in table_list:
            print(f"错误: 未找到表 {self.table_name} (且未能自动修正)")
            # 尝试提示用户
            if len(table_list) > 0:
                print(f"提示: 数据库中存在的表为 {table_list}，请检查生成脚本是否使用了其他名称。")
            return False

        # 3. Schema
        print(f"\n表结构 (Schema) - 当前表名: {self.table_name}")
        schema_info = self.con.execute(f"DESCRIBE {self.table_name}").fetchall()
        # 简单打印列名和类型
        headers = ["Column Name", "Type", "Null?"]
        print(f"{headers[0]:<20} | {headers[1]:<15} | {headers[2]:<10}")
        print("-" * 50)
        for col in schema_info:
            print(f"{col[0]:<20} | {col[1]:<15} | {col[2]:<10}")

        # 4. 总行数
        count = self.con.execute(f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
        print(f"\n总行数    : {count:,}")
        return True

    def check_actual_nulls(self):
        """1.5 检查实际数据中是否存在 NULL 值"""
        print("\n" + "=" * 80)
        print(" [STEP 1.5] 数据完整性检查 (Null Value Check)")
        print("=" * 80)

        # 获取所有列名
        columns = [r[0] for r in self.con.execute(f"DESCRIBE {self.table_name}").fetchall()]

        has_null = False
        print(f"{'Column':<30} | {'Null Count':<15} | {'Status'}")
        print("-" * 60)

        for col in columns:
            # 查询每一列的空值数量
            count = self.con.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE {col} IS NULL").fetchone()[0]
            status = "✅ OK" if count == 0 else "❌ WARNING"
            if count > 0:
                has_null = True
            print(f"{col:<30} | {count:<15} | {status}")

        if not has_null:
            print("\n结论: 数据完美！虽然Schema允许为空，但实际数据中没有任何空值。")
        else:
            print("\n结论: 警告！发现数据中存在空值，请检查生成逻辑。")

    def check_macro_stats(self):
        """2. 宏观统计：年份 | 总人口 | 迁徙人口 | 迁徙率"""
        print("\n" + "=" * 80)
        print(" [STEP 2] 宏观数据统计 (Macro Statistics)")
        print("=" * 80)

        # 聚合查询
        query = f"""
            SELECT 
                Year, 
                SUM(Total_Count) as Pop, 
                SUM(Outflow_Count) as Outflow 
            FROM {self.table_name} 
            GROUP BY Year 
            ORDER BY Year ASC
        """
        try:
            df_stats = self.con.execute(query).fetchdf()

            if df_stats.empty:
                print("数据库中无数据。")
                return

            print(f"{'Year':<6} | {'全国总人口':<15} | {'总迁徙人口':<12} | {'迁徙比率(%)':<12}")
            print("-" * 60)

            for _, row in df_stats.iterrows():
                year = int(row['Year'])
                pop = int(row['Pop'])
                mig = int(row['Outflow'])
                rate = (mig / pop * 100) if pop > 0 else 0.0
                
                print(f"{year:<6} | {pop:15,.0f} | {mig:12,.0f} | {rate:.4f}%")
        except Exception as e:
            print(f"宏观统计查询失败: {e}")

    def _get_excel_row(self, filename, city_code, year):
        """读取Excel原始行（复用生成逻辑）"""
        path = os.path.join(self.excel_dir, filename)
        if not os.path.exists(path):
            return None
        
        try:
            df = pd.read_excel(path, engine='openpyxl').fillna(0)
            df.columns = [c.strip() for c in df.columns]
            
            # 尝试匹配
            pattern = f'\({city_code}\)' # 匹配 (5000)
            mask = df['县市名'].astype(str).str.contains(pattern, regex=True) & (df['年份'] == year)
            rows = df[mask]
            
            if rows.empty:
                # 尝试纯数字匹配
                mask = df['县市名'].astype(str).str.contains(str(city_code)) & (df['年份'] == year)
                rows = df[mask]
            
            if not rows.empty:
                return rows.iloc[0]
            return None
        except Exception as e:
            print(f"读取Excel错误 {filename}: {e}")
            return None

    def validate_distribution(self, city_code, year):
        """3. 微观验证：对比 DB 分布与 Excel 约束"""
        print("\n" + "=" * 80)
        print(f" [STEP 3] 微观分布验证: 城市={city_code}, 年份={year}")
        print("=" * 80)

        # --- A. 获取 DB 数据 ---
        query = f"""
            SELECT Type_ID, Total_Count 
            FROM {self.table_name} 
            WHERE Birth_Region = '{city_code}' AND Year = {year}
        """
        try:
            df_db = self.con.execute(query).fetchdf()
        except Exception as e:
            print(f"查询失败: {e}")
            return

        if df_db.empty:
            print(f"警告：数据库中未找到 城市={city_code}, 年份={year} 的记录。")
            return

        # 解析 Type_ID
        try:
            # Type_ID 格式: M_20_EduLo_Agri_IncL_Split_5000
            split_df = df_db['Type_ID'].str.split('_', expand=True)
            df_db['Sex'] = split_df[0]
            df_db['Age'] = split_df[1]
            df_db['Edu'] = split_df[2]
            df_db['Ind'] = split_df[3]
            df_db['Fam'] = split_df[5] # 注意 Inc 是 [4]
        except Exception as e:
            print(f"解析 Type_ID 失败，可能是格式不匹配: {e}")
            return

        total_db_pop = df_db['Total_Count'].sum()
        print(f"数据库记录总人口: {total_db_pop:,}")

        # --- B. 对比验证各个维度 ---

        # 1. 验证 年龄(Age) x 性别(Sex)
        row_age = self._get_excel_row('表2_年龄与性别.xlsx', city_code, year)
        if row_age is not None:
            print("\n>>> [1] 年龄 x 性别 分布对比")
            age_mapping = {
                '20': ['15-19岁', '20-24岁'],
                '30': ['25-29岁', '30-34岁'],
                '40': ['35-39岁', '40-44岁', '45-49岁'],
                '55': ['50-54岁', '55-59岁', '60-64岁'],
                '65': ['65-69岁', '70-74岁', '75-79岁', '80-84岁', '85岁及以上']
            }
            sex_map_db = {'M': '男', 'F': '女'}
            
            print(f"{'Group':<15} | {'DB Ratio':<12} | {'Excel Ratio':<12} | {'Diff':<10}")
            print("-" * 55)
            
            total_excel_pop = 0
            for sex_chn in ['男', '女']:
                for cols in age_mapping.values():
                    for c in cols:
                        total_excel_pop += row_age.get(f"{c}_{sex_chn}", 0)
            
            for age_key, age_cols in age_mapping.items():
                for sex_db, sex_chn in sex_map_db.items():
                    # DB 计算
                    db_mask = (df_db['Age'] == age_key) & (df_db['Sex'] == sex_db)
                    db_count = df_db[db_mask]['Total_Count'].sum()
                    db_ratio = db_count / total_db_pop if total_db_pop > 0 else 0
                    
                    # Excel 计算
                    excel_count = sum(row_age.get(f"{c}_{sex_chn}", 0) for c in age_cols)
                    excel_ratio = excel_count / total_excel_pop if total_excel_pop > 0 else 0
                    
                    diff = db_ratio - excel_ratio
                    print(f"{sex_chn}_{age_key:<5} | {db_ratio:.4%}      | {excel_ratio:.4%}      | {diff:+.4%}")
        else:
            print("\n>>> [1] 年龄数据缺失 (Excel未找到)")

        # 2. 验证 学历 (Edu)
        row_edu = self._get_excel_row('表4_教育.xlsx', city_code, year)
        if row_edu is not None:
            print("\n>>> [2] 学历分布对比")
            edu_mapping = {
                'EduLo': ['未上过学', '小学', '初中'],
                'EduMid': ['高中'],
                'EduHi': ['大学专科', '大学本科及以上']
            }
            
            print(f"{'Edu Level':<15} | {'DB Ratio':<12} | {'Excel Ratio':<12} | {'Diff':<10}")
            print("-" * 55)

            total_excel_edu = 0
            for cols in edu_mapping.values():
                for c in cols:
                    total_excel_edu += row_edu.get(f"6岁及以上各种受教育程度人口_{c}_男", 0)
                    total_excel_edu += row_edu.get(f"6岁及以上各种受教育程度人口_{c}_女", 0)

            for edu_key, cols in edu_mapping.items():
                db_count = df_db[df_db['Edu'] == edu_key]['Total_Count'].sum()
                db_ratio = db_count / total_db_pop
                
                excel_count = 0
                for c in cols:
                    excel_count += row_edu.get(f"6岁及以上各种受教育程度人口_{c}_男", 0)
                    excel_count += row_edu.get(f"6岁及以上各种受教育程度人口_{c}_女", 0)
                
                excel_ratio = excel_count / total_excel_edu if total_excel_edu > 0 else 0
                
                diff = db_ratio - excel_ratio
                print(f"{edu_key:<15} | {db_ratio:.4%}      | {excel_ratio:.4%}      | {diff:+.4%}")
        else:
            print("\n>>> [2] 学历数据缺失 (Excel未找到)")

        # 3. 验证 行业 (Ind)
        row_ind = self._get_excel_row('表6_就业行业.xlsx', city_code, year)
        if row_ind is not None:
            print("\n>>> [3] 行业分布对比")
            ind_mapping = {
                'Agri': ['农林牧渔业'],
                'Mfg': ['制造业', '采矿业', '建筑业'],
                'Service': ['批发、零售、住宿、餐饮业', '仓储和邮政业', '房地产业'],
                'Wht': ['金融业', '科学研究、技术服务和地质勘察业', '教育、文化、体育和娱乐业', '公共管理和社会组织']
            }
            
            print(f"{'Industry':<15} | {'DB Ratio':<12} | {'Excel Ratio':<12} | {'Diff':<10}")
            print("-" * 55)
            
            total_excel_ind = 0
            for cols in ind_mapping.values():
                for c in cols:
                    total_excel_ind += row_ind.get(f"各种行业人口总计_{c}", 0)
            
            for ind_key, cols in ind_mapping.items():
                db_count = df_db[df_db['Ind'] == ind_key]['Total_Count'].sum()
                db_ratio = db_count / total_db_pop
                
                excel_count = sum(row_ind.get(f"各种行业人口总计_{c}", 0) for c in cols)
                excel_ratio = excel_count / total_excel_ind if total_excel_ind > 0 else 0
                
                diff = db_ratio - excel_ratio
                print(f"{ind_key:<15} | {db_ratio:.4%}      | {excel_ratio:.4%}      | {diff:+.4%}")
        else:
            print("\n>>> [3] 行业数据缺失 (Excel未找到)")

        # 4. 验证 家庭 (Fam)
        row_pop = self._get_excel_row('表1_户籍、民族、家户结构.xlsx', city_code, year)
        if row_pop is not None:
            print("\n>>> [4] 家庭状态对比")
            print(f"{'Family Status':<15} | {'DB Ratio':<12} | {'Excel Ratio':<12} | {'Diff':<10}")
            print("-" * 55)
            
            one_person = row_pop.get('家庭户_一人户', 0)
            total_fam = row_pop.get('家庭户_人口数', 0)
            
            excel_split = one_person
            excel_unit = max(0, total_fam - one_person)
            excel_total = excel_split + excel_unit
            
            fam_map = {
                'Split': excel_split,
                'Unit': excel_unit
            }
            
            for fam_key, excel_val in fam_map.items():
                db_count = df_db[df_db['Fam'] == fam_key]['Total_Count'].sum()
                db_ratio = db_count / total_db_pop
                
                excel_ratio = excel_val / excel_total if excel_total > 0 else 0
                
                diff = db_ratio - excel_ratio
                print(f"{fam_key:<15} | {db_ratio:.4%}      | {excel_ratio:.4%}      | {diff:+.4%}")
        else:
            print("\n>>> [4] 家庭数据缺失 (Excel未找到)")

    def export_sample_to_csv(self, sample_size=100):
        """导出随机采样数据到CSV文件"""
        print("\n" + "=" * 80)
        print(f" [STEP 4] 导出随机采样数据 (Sample Size: {sample_size})")
        print("=" * 80)

        try:
            # 使用 ORDER BY RANDOM() 进行真正的随机采样
            # 这样每行都是独立随机选取的，不会出现连续行聚集的问题
            query = f"""
                SELECT *
                FROM {self.table_name}
                ORDER BY RANDOM()
                LIMIT {sample_size}
            """
            df_sample = self.con.execute(query).fetchdf()

            # 生成输出文件路径（与脚本同目录）
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, "sampleDB.csv")

            # 保存到CSV
            df_sample.to_csv(output_path, index=False, encoding='utf-8-sig')

            print(f"✅ 采样成功！")
            print(f"   采样行数: {len(df_sample)}")
            print(f"   保存路径: {output_path}")
            print(f"   数据列数: {len(df_sample.columns)}")
            print(f"   列名: {', '.join(df_sample.columns[:5])}...")

        except Exception as e:
            print(f"❌ 导出采样数据失败: {e}")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 配置路径
    # 获取脚本所在目录的父目录 (02Main)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MAIN_DIR = os.path.dirname(SCRIPT_DIR)

    DB_PATH = os.path.join(MAIN_DIR, "local_migration_data.db")
    EXCEL_DIR = os.path.join(MAIN_DIR, r"0211处理后城市数据")
    
    # 验证对象初始化
    validator = DBValidator(DB_PATH, EXCEL_DIR)
    validator.connect()
    
    try:
        # 1. 基础检查
        is_valid = validator.check_meta_info()
        
        if is_valid:
            # 1.5 检查实际空值 (新增)
            validator.check_actual_nulls()

            # 2. 宏观检查
            validator.check_macro_stats()
            
            # 3. 微观分布验证 (选取几个典型样本)
            test_cases = [
                ('5000', 2000),
                ('5000', 2010),
                ('5000', 2020),
                ('3100', 2000),
                ('3100', 2010),
                ('3100', 2020),
            ]
            
            for city, year in test_cases:
                validator.validate_distribution(city, year)

            # 4. 导出随机采样数据到CSV
            validator.export_sample_to_csv(sample_size=100)

    except Exception as e:
        print(f"验证过程中发生未捕获异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        validator.close()
        print("\n验证脚本运行结束。")