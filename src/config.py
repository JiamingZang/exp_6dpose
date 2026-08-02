"""YAML 配置加载与消融覆盖工具。

configs/default.yaml 是唯一的超参来源；消融 yaml 只声明 sweep 字段与取值，
由 apply_override 按点号路径写回配置副本。
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any],
                _path: str = "") -> Dict[str, Any]:
    """递归合并：overlay 的叶子覆盖 base；两侧都是 dict 时逐键深合。

    列表按整体替换（不做元素级合并）——配置里的列表（objects/topk_best 等）
    语义上是原子值。

    base 侧是 dict 而 overlay 侧是 None 时直接报错：YAML 里一个写秃了的空
    冒号（`matching:` 后面什么都没写）会解析成 None，静默替换掉整段之后，
    错误要等到远端 `cfg["matching"].get(...)` 才以 AttributeError 现形，
    完全看不出是配置写错。要清空子段必须显式写 `{}`。
    """
    out = dict(base)
    for k, v in overlay.items():
        key_path = f"{_path}.{k}" if _path else str(k)
        if v is None and isinstance(out.get(k), dict):
            raise ValueError(
                f"配置项 `{key_path}` 在覆盖层里是空值（None），但 base 里是"
                f"配置段（dict）。这通常是 YAML 里写了 `{k}:` 却没写内容。"
                f"要真的清空这一段请显式写 `{k}: {{}}`；要沿用 base 请整行删掉。")
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v, key_path)
        else:
            out[k] = v
    return out


def load_config(path: str = "configs/default.yaml",
                _chain: tuple = ()) -> Dict[str, Any]:
    """加载配置。支持 `base: <相对路径>` 覆盖式继承：

    yaml 顶层含 base 时，先加载以本文件所在目录为基准解析的 base 配置，
    再把本文件其余字段深合并覆盖上去——legacy_mypose.yaml 只声明与
    default.yaml 的差异项，避免整份复制造成漂移。

    `_chain` 是内部参数（已展开的 base 链，绝对路径）：base 成环时原本是
    一句无信息量的 RecursionError，这里改成带完整链条的 ValueError。
    """
    p = Path(path)
    resolved = p.resolve()
    if resolved in _chain:
        loop = " → ".join(str(x) for x in (*_chain, resolved))
        raise ValueError(f"配置 base 链存在循环引用: {loop}")
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_rel = cfg.pop("base", None)
    if base_rel:
        base_cfg = load_config(str((p.parent / base_rel).resolve()),
                               _chain=(*_chain, resolved))
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


def apply_override(cfg: Dict[str, Any], dotted_field: str,
                   value: Any) -> Dict[str, Any]:
    """按点号路径覆盖配置（返回深拷贝，不改原 cfg）。

    特殊路径 `templates.__preset__` 用于模板数消融：value 是预设 dict，
    整段合入 templates（一个档位同时改视角模式/视角数/旋转数）。
    """
    out = copy.deepcopy(cfg)
    keys = dotted_field.split(".")
    if keys[-1] == "__preset__":
        node = out
        for k in keys[:-1]:
            node = node[k]
        assert isinstance(value, dict), "__preset__ 的取值必须是 dict"
        node.update(value)
        return out
    node = out
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return out


def load_ablation(cfg: Dict[str, Any], ablation_path: str):
    """读取消融 yaml，生成 (标签, 覆盖后配置, 是否需重新 onboard) 序列。"""
    with open(ablation_path, "r", encoding="utf-8") as f:
        ab = yaml.safe_load(f)["ablation"]
    field = ab["sweep_field"]
    presets = ab.get("presets", {})
    reonboard = bool(ab.get("requires_reonboard", False))
    runs = []
    for v in ab["sweep_values"]:
        value = presets[v] if presets else v
        label = f"{ab['name']}={v}"
        runs.append((label, apply_override(cfg, field, value), reonboard))
    return ab["name"], runs
