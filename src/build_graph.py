import sqlite3
import json

def extract_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 获取所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema_info = {"L0": [], "L1": tables, "L2": [db_path], "edges": []}

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [f"{table}.{row[1]}" for row in cursor.fetchall()]
        schema_info["L0"].extend(columns)

        # 建立列到表的连接
        for col in columns:
            schema_info["edges"].append([col, table])

        # 建立表到数据库的连接
        schema_info["edges"].append([table, db_path])

    conn.close()
    return schema_info

def build_graph():
    school_db_path = "./data/school.db"
    company_db_path = "./data/company.db"
    output_path = "./data/schema_graph.json"

    schema_info = {"L0": [], "L1": [], "L2": ["merged_db"], "edges": []}

    # 提取 school.db 的模式信息
    school_schema = extract_schema(school_db_path)
    schema_info["L0"].extend(school_schema["L0"])
    schema_info["L1"].extend(school_schema["L1"])
    schema_info["edges"].extend(school_schema["edges"])

    # 提取 company.db 的模式信息
    company_schema = extract_schema(company_db_path)
    schema_info["L0"].extend(company_schema["L0"])
    schema_info["L1"].extend(company_schema["L1"])
    schema_info["edges"].extend(company_schema["edges"])

    # 合并两个数据库的表到 L1 层，连接到一个新的 L2 层
    schema_info["L1"].extend(["school_db", "company_db"])
    schema_info["edges"].append(["school_db", "merged_db"])
    schema_info["edges"].append(["company_db", "merged_db"])

    # 保存合并后的图
    with open(output_path, "w") as f:
        json.dump(schema_info, f, indent=4)

    print(f"✅ 合并的层次图已保存到 {output_path}")

if __name__ == "__main__":
    build_graph()
