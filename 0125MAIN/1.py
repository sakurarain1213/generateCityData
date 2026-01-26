import pandas as pd
import json

def load_cities_from_jsonl(jsonl_path):
    """
    从 jsonl 文件加载城市映射关系
    返回格式: [('1301', '石家庄'), ('1302', '唐山'), ...]
    """
    city_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # 将 json 中的 key 对应到元组中
                city_list.append((data['city_id'], data['name']))
    return city_list

def process_city_data(input_excel, city_jsonl, output_csv):
    # 1. 加载城市定义
    cities_mapping = load_cities_from_jsonl(city_jsonl)
    
    # 2. 读取原始数据
    df = pd.read_excel(input_excel)
    col_city = df.columns[0]
    col_year = df.columns[1]

    # 3. 定义转换函数
    def map_to_code(cell_value):
        cell_str = str(cell_value)
        for code, name in cities_mapping:
            if name in cell_str:
                return code
        return cell_str

    # 4. 执行替换与聚合
    df[col_city] = df[col_city].apply(map_to_code)
    
    # 聚合：按编码和年份分组，对所有数值列求和
    df_result = df.groupby([col_city, col_year]).sum(numeric_only=True).reset_index()

    # 5. 导出
    df_result.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"处理完成！结果已存至 {output_csv}")

# --- 配置路径并运行 ---
if __name__ == "__main__":
    # 假设你的 jsonl 文件名为 cities.jsonl
    config_jsonl = r'C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN\city_nodes.jsonl' 
    input_xlsx = r'C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN\1.xlsx'
    output_csv = r'C:\Users\w1625\Desktop\经济学模拟\cityDataGenerate\0125MAIN\2.csv'

    process_city_data(input_xlsx, config_jsonl, output_csv)