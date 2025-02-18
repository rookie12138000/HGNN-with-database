import sqlite3
import os

def create_school_db():
    db_path = "./data/school.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建学生表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        class_id INTEGER
    );
    ''')

    # 创建班级表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS classes (
        id INTEGER PRIMARY KEY,
        class_name TEXT
    );
    ''')

    # 创建教师表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS teachers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        subject TEXT
    );
    ''')

    conn.commit()
    conn.close()
    print("✅ SQLite 数据库 'school.db' 创建成功！")

def create_company_db():
    db_path = "./data/company.db"
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # 连接到数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建员工表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY,
        name TEXT,
        position TEXT,
        department_id INTEGER
    );
    ''')

    # 创建部门表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS departments (
        id INTEGER PRIMARY KEY,
        department_name TEXT
    );
    ''')

    # 创建项目表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY,
        project_name TEXT,
        employee_id INTEGER
    );
    ''')

    conn.commit()
    conn.close()
    print("✅ SQLite 数据库 'company.db' 创建成功！")

if __name__ == "__main__":
    create_school_db()
    create_company_db()
