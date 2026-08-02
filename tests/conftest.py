"""pytest 共享配置：保证 `import src` 从仓库根解析。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
