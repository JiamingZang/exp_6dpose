"""匹配器消融的替代匹配器：DINOv2 patch 特征 / LoFTR（TODO 接口）。

DINOv2PatchMatcher：用 DINOv2 ViT-L/14 的 patch token 当稠密描述子。
与 MASt3R 不同，DINOv2 是单图前向（无成对交叉注意力），因此模板侧
描述子可以完整预缓存，在线只前向查询一次——速度更快，但特征是
语义级的（14×14 patch 粒度 + 双线性上采样），几何定位精度预期低于
MASt3R 的像素级匹配头。

输出接口与 Mast3rMatcher.match 一致，pipeline 无感切换。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .correspondence import cycle_consistency_filter, sample_correspondences
from .mast3r_wrapper import (TemplateMatch, _resize_to_multiple16,
                             decode_template_indices)

_DINO_HINT = (
    "DINOv2 patch 匹配器需要 GPU 机器（torch.hub facebookresearch/dinov2）。"
)


class Dinov2PatchMatcher:
    """DINOv2 patch token 稠密匹配器（消融 baseline）。"""

    PATCH = 14

    def __init__(self, cfg_matching: Dict, device: str = "cuda",
                 model_name: str = "dinov2_vitl14", n_score_pixels: int = 2048):
        try:
            import torch
        except ImportError as e:
            raise ImportError(f"{_DINO_HINT}\n原始错误: {e}") from e
        self.torch = torch
        self.device = device
        self.cfg = cfg_matching
        self.long_side = int(cfg_matching.get("image_size", 512))
        self.n_score_pixels = n_score_pixels
        self.model = torch.hub.load("facebookresearch/dinov2:main", model_name
                                    ).to(device).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device
                                  ).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device
                                 ).view(3, 1, 1)
        self._tmpl_desc = None       # (M,) 每模板前景描述子
        self._tmpl_pix = None
        self._tmpl_fg = None

    # ------------------------------------------------------------------
    def _dense_desc(self, img_u8: np.ndarray):
        """单图 → 像素级描述子图 (H,W,C)（patch token 双线性上采样）。"""
        import torch.nn.functional as F
        torch = self.torch
        h, w = img_u8.shape[:2]
        # DINOv2 要求边长为 14 的倍数
        nh = max(self.PATCH, round(h / self.PATCH) * self.PATCH)
        nw = max(self.PATCH, round(w / self.PATCH) * self.PATCH)
        import cv2
        img = cv2.resize(img_u8, (nw, nh))
        x = torch.tensor(img, dtype=torch.float32, device=self.device
                         ).permute(2, 0, 1) / 255.0
        x = ((x - self._mean) / self._std)[None]
        with torch.no_grad():
            out = self.model.forward_features(x)
        tok = out["x_norm_patchtokens"][0]            # (nh/14*nw/14, C)
        gh, gw = nh // self.PATCH, nw // self.PATCH
        fmap = tok.reshape(gh, gw, -1).permute(2, 0, 1)[None]
        fmap = F.interpolate(fmap, size=(h, w), mode="bilinear",
                             align_corners=False)[0].permute(1, 2, 0)
        fmap = fmap / fmap.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return fmap                                    # (h,w,C) 单位范数

    def prepare_templates(self, images: np.ndarray, alphas: np.ndarray,
                          fg_thresh: float = 0.5):
        """模板描述子全量预缓存（单图前向，跨帧完全复用）。"""
        descs, pixs, fgs = [], [], []
        for im, a in zip(images, alphas):
            fmap = self._dense_desc(im)
            fg = np.asarray(a, dtype=np.float32) > fg_thresh
            tys, txs = np.nonzero(fg)
            flat = self.torch.tensor(tys * fg.shape[1] + txs,
                                     device=self.device)
            descs.append(fmap.reshape(-1, fmap.shape[-1])[flat].half())
            pixs.append(np.stack([txs, tys], axis=1))
            fgs.append(fg)
        self._tmpl_desc, self._tmpl_pix, self._tmpl_fg = descs, pixs, fgs

    # ------------------------------------------------------------------
    def match(self, query_crop_u8: np.ndarray, query_mask_crop: np.ndarray,
              top_k: int, sim_threshold: float, cycle_tau_px: float,
              n_sample: int, rng: Optional[np.random.Generator] = None,
              prefilter_order: Optional[np.ndarray] = None
              ) -> Tuple[List[TemplateMatch], Tuple[float, float], np.ndarray]:
        """接口与 Mast3rMatcher.match 一致（含 DINOv2 Top-K 预筛）。"""
        import cv2
        torch = self.torch
        if self._tmpl_desc is None:
            raise RuntimeError("先调用 prepare_templates()")
        if rng is None:
            rng = np.random.default_rng(0)

        q_img, (sxy) = _resize_to_multiple16(query_crop_u8, self.long_side)
        q_mask = cv2.resize(query_mask_crop.astype(np.uint8),
                            (q_img.shape[1], q_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        fmap = self._dense_desc(q_img)
        ys, xs = np.nonzero(q_mask)
        if len(ys) < 16:
            return [], sxy, np.full(len(self._tmpl_desc), -np.inf)
        pix_q_all = np.stack([xs, ys], axis=1)
        flat_q = torch.tensor(ys * q_img.shape[1] + xs, device=self.device)
        dq = fmap.reshape(-1, fmap.shape[-1])[flat_q].float()   # (Nq,C)

        sub = (rng.choice(len(ys), self.n_score_pixels, replace=False)
               if len(ys) > self.n_score_pixels else np.arange(len(ys)))
        sub_t = torch.tensor(sub, device=self.device)

        n_tmpl = len(self._tmpl_desc)
        scores = np.full(n_tmpl, -np.inf)
        # DINOv2 预筛时只对 Top-K 个模板打分/建对应
        decode_idxs = decode_template_indices(n_tmpl, top_k, prefilter_order)
        for i in decode_idxs:
            dt = self._tmpl_desc[i]
            sim_sub = dq[sub_t] @ dt.float().T
            scores[i] = float(sim_sub.max(dim=1).values.mean())

        if prefilter_order is not None:
            order = np.array(decode_idxs, dtype=int)
        else:
            order = np.argsort(-scores)[:min(top_k, n_tmpl)]
        matches = []
        for i in order:
            i = int(i)
            dt = self._tmpl_desc[i].float()
            sim = dq @ dt.T
            nn_q2t = sim.argmax(dim=1).cpu().numpy()
            nn_t2q = sim.argmax(dim=0).cpu().numpy()
            sims_fwd = sim.max(dim=1).values.cpu().numpy()
            del sim
            idx_q = np.arange(len(pix_q_all))
            keep = cycle_consistency_filter(
                pix_q_all.astype(np.float64), idx_q, nn_q2t, nn_t2q,
                tau_px=cycle_tau_px)
            ok = keep & (sims_fwd > sim_threshold)
            p2, p3, ss = sample_correspondences(
                pix_q_all[ok].astype(np.float64),
                self._tmpl_pix[i][nn_q2t[ok]].astype(np.float64),
                sims_fwd[ok], n_sample=n_sample, rng=rng)
            matches.append(TemplateMatch(int(i), float(scores[i]), p2, p3, ss))
        return matches, sxy, scores


class LoFTRMatcher:
    """LoFTR 匹配器接口（匹配器消融预留）。

    TODO: 接入 kornia.feature.LoFTR（indoor_ds 权重）。LoFTR 输出稀疏半稠密
    匹配对 (mkpts0, mkpts1, mconf)，接入方式：
      1. prepare_templates 缓存模板灰度图；
      2. match 里对 Top-K 模板逐对跑 LoFTR，把 (mkpts0, mkpts1) 直接作为
         (pix_q, pix_t)，mconf 作为 sims；模板级分数 sim(m) 用匹配数/均值
         置信度替代；
      3. 其余（坐标图映射、PnP）与主管线一致。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "LoFTR 匹配器尚未接入（匹配器消融预留接口）。"
            "实现说明见 LoFTRMatcher docstring；主实验请用 matcher=mast3r。")



class LightGlueMatcher:
    """SuperPoint + LightGlue 稀疏匹配器（对比基线，攻对应质量瓶颈）。

    与 Mast3rMatcher.match 同接口，pipeline 无感切换（matching.matcher:
    lightglue）。差异：稀疏关键点匹配（无成对 3D 输出 → pts3d_q=None，
    深度一致性自动关闭）；模板关键点+desc 预缓存。

    权重：weights/superpoint_v1.pth + weights/superpoint_lightglue.pth。
    """

    def __init__(self, cfg_matching: Dict, device: str = "cuda",
                 n_score_pixels: int = 2048):
        import sys as _sys
        from pathlib import Path
        _lg = Path(__file__).resolve().parents[2] / "third_party" / "lightglue"
        if str(_lg) not in _sys.path:
            _sys.path.insert(0, str(_lg))
        import torch
        from lightglue import LightGlue, SuperPoint
        self.torch = torch
        self.device = device
        self.cfg = cfg_matching
        self.long_side = int(cfg_matching.get("image_size", 512))
        self.n_score_pixels = n_score_pixels
        wdir = Path(__file__).resolve().parents[2] / "weights"
        self.extractor = SuperPoint(max_num_keypoints=int(
            cfg_matching.get("lg_max_kps", 2048))).eval().to(device)
        self.extractor.load_state_dict(torch.load(
            wdir / "superpoint_v1.pth", map_location="cpu",
            weights_only=False))
        self.lg = LightGlue(features="superpoint").eval().to(device)
        _sd = torch.load(wdir / "superpoint_lightglue.pth",
                         map_location="cpu", weights_only=False)
        # v0.1_arxiv 旧键名 → 新版 transformers.N 结构
        # （self_attn.N.* / cross_attn.N.* / log_assignment.N.r）
        _sd2 = {}
        for _k, _v in _sd.items():
            if _k.startswith("self_attn.") or _k.startswith("cross_attn."):
                _i, _rest = _k.split(".", 2)[1], _k.split(".", 2)[2]
                _sd2[f"transformers.{_i}.{_k.split('.')[0]}.{_rest}"] = _v
            elif _k.startswith("log_assignment."):
                _sd2[f"transformers.{_k.split('.')[1]}.log_assignment.r"] = _v
            else:
                _sd2[_k] = _v
        self.lg.load_state_dict(_sd2, strict=False)
        self.conf_tau = float(cfg_matching.get("lg_conf_tau", 0.5))
        self._tmpl_kps = None      # list[(N,2) 像素]
        self._tmpl_desc = None     # list[(N,256)]
        self._tmpl_shape = None    # list[(H,W)]

    # ------------------------------------------------------------------
    def _prep(self, img_u8: np.ndarray):
        import cv2
        g = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return self.torch.from_numpy(g)[None, None].to(self.device)

    def _extract(self, img_u8: np.ndarray):
        with self.torch.no_grad():
            feats = self.extractor.extract(self._prep(img_u8))
        return (feats["keypoints"][0].cpu().numpy(),
                feats["descriptors"][0].cpu().numpy())

    def _lg_input(self, kps: np.ndarray, desc: np.ndarray, hw):
        torch = self.torch
        return {
            "keypoints": torch.from_numpy(kps)[None].to(self.device),
            "descriptors": torch.from_numpy(desc)[None].to(self.device),
            "image_size": torch.tensor([[hw[0], hw[1]]],
                                       device=self.device),
        }

    def prepare_templates(self, images: np.ndarray, alphas: np.ndarray,
                          fg_thresh: float = 0.5):
        kps, descs, shapes = [], [], []
        for im in images:
            k, d = self._extract(im)
            kps.append(k)
            descs.append(d)
            shapes.append((im.shape[0], im.shape[1]))
        self._tmpl_kps, self._tmpl_desc = kps, descs
        self._tmpl_shape = shapes

    # ------------------------------------------------------------------
    def match(self, query_crop_u8: np.ndarray, query_mask_crop: np.ndarray,
              top_k: int, sim_threshold: float, cycle_tau_px: float,
              n_sample: int, rng: Optional[np.random.Generator] = None,
              prefilter_order: Optional[np.ndarray] = None
              ) -> Tuple[List[TemplateMatch], Tuple[float, float],
                         np.ndarray, None]:
        if rng is None:
            rng = np.random.default_rng(0)
        q_img, (sx, sy) = _resize_to_multiple16(query_crop_u8, self.long_side)
        kq, dq = self._extract(q_img)
        n_tmpl = len(self._tmpl_kps)
        scores = np.full(n_tmpl, -np.inf)
        decode_idxs = decode_template_indices(
            n_tmpl, top_k, prefilter_order,
            decode_k=int(self.cfg.get("top_k_prescreen", top_k)))
        matches = []
        for i in decode_idxs:
            i = int(i)
            kt, dt = self._tmpl_kps[i], self._tmpl_desc[i]
            if len(kt) == 0 or len(kq) == 0:
                continue
            with self.torch.no_grad():
                out = self.lg({
                    "image0": self._lg_input(kq, dq,
                                             (q_img.shape[0], q_img.shape[1])),
                    "image1": self._lg_input(kt, dt, self._tmpl_shape[i]),
                })
            m = out["matches0"].cpu().numpy()      # (Mq,) 每查询点的模板索引，-1=无匹配
            conf = out["matching_scores0"].cpu().numpy()
            ok = (m >= 0) & (conf > self.conf_tau)
            if ok.sum() == 0:
                continue
            iq_lg = np.nonzero(ok)[0]
            it_lg = m[ok]
            conf = conf[ok]
            px_q = kq[iq_lg]
            px_t = kt[it_lg]
            scores[i] = float(conf.mean())
            p2, p3, ss = sample_correspondences(
                px_q.astype(np.float64), px_t.astype(np.float64), conf,
                n_sample=n_sample, rng=rng)
            matches.append(TemplateMatch(
                template_idx=i, score=float(scores[i]),
                pix_q=p2, pix_t=p3, sims=ss, pts3d_q=None))
        matches.sort(key=lambda x: x.score, reverse=True)
        return matches, (sx, sy), scores, None

    # ------------------------------------------------------------------
    # pipeline 兼容桩：LightGlue 无稠密解码（_decode_top_desc 走不到）
    def _encode(self, img_u8: np.ndarray):
        raise AttributeError("LightGlueMatcher 无 _encode（无稠密解码）")
