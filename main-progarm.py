# main-project.py

# 在主程序中使用
from utils.Projectreader import ProjectReader


def main():
    # 读取所有项目
    reader = ProjectReader()
    program = reader.read_projects_from_dir("data/projects/")

    # 验证项目结构
    print(f"Loaded {len(program.projects)} projects")
    print(f"Global resources: {program.global_resources}")
    for project in program.projects.values():
        print(f"Project {project.project_id} has {len(project.activities)} activities")


if __name__ == "__main__":
    main()
