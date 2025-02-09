# utils/config_loader.py

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_file: str = "config/settings.yaml") -> Dict[str, Any]:
    """
    加载YAML配置文件
    :param config_file: 配置文件路径（相对于项目根目录）
    :return: 配置字典
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件 {config_path} 不存在")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# 全局配置对象（可在其他模块直接导入）
config = load_config()