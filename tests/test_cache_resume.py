"""缓存重定向续跑回归测试（08-10 lamp/phone 重复处理事故）。"""
import json
from pathlib import Path

from src.pipeline import _load_cache_records


def _meta(matches_dir="m", cfg_hash="h" * 40):
    return {"matches_dir": matches_dir, "cfg_hash": cfg_hash}


def _rec(fid, cfg_hash="h" * 40):
    return {"frame_id": fid, "cfg_hash": cfg_hash,
            "success": True, "m": {"add_01d": 0.5}, "timings": {"pnp": 0.1}}


def test_load_cache_records_matching_meta(tmp_path):
    cp = tmp_path / "phone.jsonl"
    cp.write_text("\n".join([
        json.dumps({"__meta__": _meta()}), json.dumps(_rec(1)), json.dumps(_rec(2)),
    ]) + "\n")
    out = _load_cache_records(cp, _meta())
    assert set(out) == {1, 2}
    assert out[1]["m"]["add_01d"] == 0.5


def test_load_cache_records_mismatched_hash_returns_empty(tmp_path):
    cp = tmp_path / "phone.jsonl"
    cp.write_text(json.dumps(
        {"__meta__": _meta(matches_dir="m", cfg_hash="y" * 40)}) + "\n")
    assert _load_cache_records(cp, _meta()) == {}


def test_load_cache_records_ignores_matches_dir_provenance(tmp_path):
    """08-13 修复：matches_dir 只是产物来源（live 提取 vs 预提取），同一
    cfg_hash 下逐帧等价——只比对 cfg_hash，避免主表 hash 的消融档白重匹配。"""
    cp = tmp_path / "phone.jsonl"
    cp.write_text("\n".join([
        json.dumps({"__meta__": _meta(matches_dir="outputs/matches13_dc2/phone")}),
        json.dumps(_rec(1)), json.dumps(_rec(2)),
    ]) + "\n")
    out = _load_cache_records(cp, _meta(matches_dir=None))
    assert set(out) == {1, 2}


def test_load_cache_records_skips_wrong_fingerprint_lines(tmp_path):
    cp = tmp_path / "phone.jsonl"
    cp.write_text("\n".join([
        json.dumps({"__meta__": _meta()}),
        json.dumps(_rec(1)),
        json.dumps(_rec(2, cfg_hash="x" * 40)),  # 异指纹帧：忽略
        "not json",
    ]) + "\n")
    out = _load_cache_records(cp, _meta())
    assert set(out) == {1}


def test_load_cache_records_missing_file(tmp_path):
    assert _load_cache_records(tmp_path / "nope.jsonl", _meta()) == {}


def test_redirect_resume_chain_loads_redirected_content(tmp_path):
    """meta 不匹配 → 重定向到 <stem>_<hash8>.jsonl；该文件已有内容时
    重启必须加载它（旧实现漏加载导致重复处理全部帧）。"""
    meta = _meta()
    stale = tmp_path / "phone.jsonl"
    stale.write_text(json.dumps({"__meta__": _meta(matches_dir="old")}) + "\n")
    # 模拟第一轮重定向后的缓存：已写入 2 帧
    redirected = tmp_path / f"phone_{meta['cfg_hash'][:8]}.jsonl"
    redirected.write_text("\n".join([
        json.dumps({"__meta__": meta}), json.dumps(_rec(1)), json.dumps(_rec(2)),
    ]) + "\n")
    # 第二轮重启：stale 的 meta 仍不匹配 → 重定向；必须从重定向文件加载
    out = _load_cache_records(redirected, meta)
    assert set(out) == {1, 2}
