import sqlite3
import json
import os

def extract_schema(db_path, global_node_mapping):
    """从数据库中提取模式信息，并使用全局 node_mapping 确保索引唯一"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    db_name = os.path.basename(db_path)  # 例如 school.db、company.db
    schema_info = {
        "L0": [],
        "L1": [],
        "L2": [db_name],  # 数据库层
        "edges": [],
        "node_mapping": global_node_mapping  # 使用全局映射
    }

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema_info["L1"].extend([f"{db_name}.{table}" for table in tables])

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [f"{db_name}.{table}.{row[1]}" for row in cursor.fetchall()]
        schema_info["L0"].extend(columns)

        # 记录列 -> 表的边
        for col in columns:
            if col not in schema_info["node_mapping"]:
                schema_info["node_mapping"][col] = len(schema_info["node_mapping"])
            schema_info["edges"].append([col, f"{db_name}.{table}"])

        # 记录表 -> 数据库的边
        if f"{db_name}.{table}" not in schema_info["node_mapping"]:
            schema_info["node_mapping"][f"{db_name}.{table}"] = len(schema_info["node_mapping"])
        schema_info["edges"].append([f"{db_name}.{table}", db_name])

    # 记录数据库层
    if db_name not in schema_info["node_mapping"]:
        schema_info["node_mapping"][db_name] = len(schema_info["node_mapping"])

    conn.close()
    return schema_info

def build_graph():
    school_db_path = "./data/school.db"
    company_db_path = "./data/company.db"
    output_path = "./data/schema_graph.json"

    # 初始化全局唯一 node_mapping
    global_node_mapping = {}

    # 初始化合并后的图结构
    merged_schema = {
        "L0": [],
        "L1": [],
        "L2": [],
        "edges": [],
        "node_mapping": global_node_mapping  # 统一索引管理
    }

    # 处理 school.db
    school_schema = extract_schema(school_db_path, global_node_mapping)
    merged_schema["L0"].extend(school_schema["L0"])
    merged_schema["L1"].extend(school_schema["L1"])
    merged_schema["L2"].extend(school_schema["L2"])
    merged_schema["edges"].extend(school_schema["edges"])

    # 处理 company.db
    company_schema = extract_schema(company_db_path, global_node_mapping)
    merged_schema["L0"].extend(company_schema["L0"])
    merged_schema["L1"].extend(company_schema["L1"])
    merged_schema["L2"].extend(company_schema["L2"])
    merged_schema["edges"].extend(company_schema["edges"])

    # 保存结果
    with open(output_path, "w") as f:
        json.dump(merged_schema, f, indent=4, ensure_ascii=False)
    print(f"✅ 修正后的层次图已保存到 {output_path}")

if __name__ == "__main__":
    build_graph()
