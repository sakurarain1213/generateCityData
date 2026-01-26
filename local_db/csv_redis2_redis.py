# -*- coding: utf-8 -*-
"""
修复版：将 ARDM 导出的十六进制 CSV 导入 Redis，并自动将 MessagePack 二进制转为 JSON
"""

import csv
import redis
import binascii
import json
import sys

# 尝试导入 msgpack，如果没有安装则报错提示
try:
    import msgpack
except ImportError:
    print("错误：请先安装 msgpack 库以解析二进制数据。")
    print("运行命令: pip install msgpack")
    sys.exit(1)

# ================= 配置区域 =================
REDIS_HOST = '127.0.0.1'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None

# 你的 CSV 文件路径
CSV_FILE_PATH = r'C:\Users\w1625\Desktop\Dump_20260125.csv'
# ===========================================

def decode_and_convert(raw_bytes):
    """
    核心逻辑：尝试将 MessagePack 二进制数据转换为 JSON 对象
    """
    if not raw_bytes:
        return None

    # 1. 尝试 MessagePack 解码
    try:
        # raw=False 让 msgpack 尝试将 bytes 解码为字符串
        # strict_map_key=False 允许非字符串作为字典的 key
        decoded_data = msgpack.unpackb(raw_bytes, raw=False, strict_map_key=False)
        
        # 如果解码成功，说明它是序列化数据
        # 我们把它转换成标准的 JSON 字符串返回
        return json.dumps(decoded_data, ensure_ascii=False)
    except Exception:
        # 如果不是 MessagePack，可能就是普通的二进制或已经是文本
        pass

    # 2. 尝试直接 UTF-8 解码（如果是普通文本）
    try:
        return raw_bytes.decode('utf-8')
    except:
        pass

    # 3. 实在解不开，返回原始二进制（但这通常不会发生，除非数据损坏）
    return raw_bytes

def import_csv_to_json():
    print(f"开始导入: {CSV_FILE_PATH}")
    
    # 连接 Redis
    try:
        r = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, 
            password=REDIS_PASSWORD, decode_responses=False # 写入时保持二进制灵活性
        )
        r.ping()
    except Exception as e:
        print(f"Redis 连接失败: {e}")
        return

    success = 0
    errors = 0

    with open(CSV_FILE_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        
        for i, row in enumerate(reader, 1):
            try:
                if len(row) < 2: continue
                
                # 1. 获取 CSV 中的十六进制字符串
                hex_key = row[0].strip()
                hex_val = row[1].strip()
                ttl = int(row[2]) if len(row) > 2 else -1

                if not hex_key: continue

                # 2. 十六进制 -> 原始字节 (Binary)
                key_bytes = binascii.unhexlify(hex_key)
                val_bytes = binascii.unhexlify(hex_val)

                # 3. 【关键】检测并转换数据格式
                # 你的日志显示原数据是 Hash，但在 CSV 导出的这个 value 里，
                # 看起来像是一个包含 'by_type', 'total', 'year_month' 的字典结构。
                # ARDM 导出 Hash 时，有时会把整个 Hash 打包成一个 MessagePack。
                
                final_value = decode_and_convert(val_bytes)

                # 4. 判断如何写入 Redis
                # 如果转换出来是 JSON 字符串 (str)，我们用 set 存（看起来像 String）
                # 但你之前的日志显示原 Key 类型是 Hash。
                # 这里有一个分歧点：
                # 情况 A: 你想恢复成 Redis 的 Hash 结构 (HSET)
                # 情况 B: 你只想要能在软件里看懂 JSON (SET)
                
                # 我们先解析一下转换后的数据结构
                try:
                    data_obj = json.loads(final_value)
                    
                    # 如果转换后的数据是一个字典，且看起来像你要的 Hash 结构
                    if isinstance(data_obj, dict):
                        # === 方案 A: 尝试恢复为 Redis Hash (推荐) ===
                        # 如果是 Hash，我们遍历字典，逐个字段写入
                        # 先删除旧的（防止字段残留）
                        r.delete(key_bytes) 
                        
                        # 转换字典里的所有值为字符串，以防 Redis 报错
                        safe_dict = {}
                        for k, v in data_obj.items():
                            if isinstance(v, (dict, list)):
                                safe_dict[k] = json.dumps(v, ensure_ascii=False)
                            else:
                                safe_dict[k] = str(v)
                        
                        if safe_dict:
                            r.hset(key_bytes, mapping=safe_dict)
                        # ==========================================
                    else:
                        # 不是字典，就直接存 String
                        r.set(key_bytes, final_value)
                except:
                    # 如果不是 JSON 格式，直接存 String
                    r.set(key_bytes, final_value)

                # 恢复过期时间
                if ttl != -1:
                    r.expire(key_bytes, ttl)

                success += 1
                if success % 100 == 0:
                    print(f"已处理: {success}")

            except Exception as e:
                errors += 1
                print(f"行 {i} 错误: {e}")

    print(f"\n完成！成功: {success}, 失败: {errors}")

if __name__ == "__main__":
    import_csv_to_json()