"""MASt3R 稠密局部特征封装。GPU-only。

要点：
- 官方权重 MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric（CatMLP+DPT
  匹配头，局部特征维度 d=24，见 mast3r/README.md:134 与
  mast3r/catmlp_dpt_head.py:152 mlp_odim=24），经
  AsymmetricMASt3R.from_pretrained 加载（本地 .pth 路径走 load_model，
  强制 landscape_only=False，任意宽高比可推理，见 mast3r/model.py:20-36,46-51）；
- MASt3R 解码器对图像对做交叉注意力，desc 是成对输出——真正可以跨帧
  复用的只有 ViT 编码器 token。本封装在 prepare_templates() 里
  **预提取并缓存全部模板的编码器特征**（40 模板只编码一次），在线阶段
  每帧只需编码查询一次 + 跑 40 次解码（按 batch_size 分批并行）；
- 模板级分数 sim(m) = 查询前景像素对模板所有像素最大相似度的均值
  （下采样 n_score 个前景像素估计，控制打分开销）；
- Top-K 模板的稠密互最近邻在 GPU 上算 argmax（朴素 numpy 在 10^5×10^5
  相似度矩阵上不可行），随后把小规模的下标/相似度数组交回
  src.matching.correspondence 里经过单测的纯逻辑做 cycle 过滤与阈值筛选，
  保证 GPU 路径与本地测试的逻辑一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .correspondence import (cycle_consistency_filter, sample_correspondences)

_MAST3R_HINT = (
    "MASt3R 推理需要 GPU 机器，且需先克隆官方仓库并加入 PYTHONPATH（见 setup_gpu.sh）：\n"
    "  git clone --recursive https://github.com/naver/mast3r\n"
    "  export PYTHONPATH=$PWD/mast3r:$PWD/mast3r/dust3r:$PYTHONPATH\n"
    "权重: MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    "（setup_gpu.sh 自动下载）"
)


@dataclass
class TemplateMatch:
    """单个 Top-K 模板的匹配输出（坐标均在 MASt3R 输入分辨率下）。"""
    template_idx: int
    score: float                    # sim(m)
    pix_q: np.ndarray               # (M,2) 查询侧像素（resize 后坐标）
    pix_t: np.ndarray               # (M,2) 模板侧像素（模板原生 256 坐标）
    sims: np.ndarray                # (M,)
    pts3d_q: np.ndarray | None = None  # (M,3) 查询侧相机系 3D（成对重建，
                                       # 度量尺度）；PnP 深度一致性用


def _resize_to_multiple16(img: np.ndarray, long_side: int):
    """长边缩放到 long_side 并取 16 的倍数（ViT patch 约束），返回图与缩放比。"""
    import cv2
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    nh = max(16, int(round(h * scale / 16)) * 16)
    nw = max(16, int(round(w * scale / 16)) * 16)
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (nw / w, nh / h)


def decode_template_indices(n_tmpl: int, top_k: int,
                            prefilter_order: Optional[np.ndarray] = None
                            ) -> List[int]:
    """决定本帧需要过 MASt3R 解码的模板下标（Top-K 预筛语义）。

    - prefilter_order 为 None（template_ranking=mast3r，历史行为）：解码全部
      模板，再按 sim(m) 选 Top-K——省不了算力；
    - prefilter_order 给定（template_ranking=dinov2，默认）：定位阶段的
      DINOv2 相似度排序已选出候选，只解码前 min(top_k, M) 个，K<M 时真正
      省下 (M-K) 次成对解码与 PnP。
    """
    if prefilter_order is None:
        return list(range(n_tmpl))
    order = [int(i) for i in np.asarray(prefilter_order).ravel()
             if 0 <= int(i) < n_tmpl]
    return order[:min(top_k, n_tmpl)]


def resolve_prefilter_order(prescreen: str, template_ranking: str,
                            template_order: Optional[np.ndarray]
                            ) -> Optional[np.ndarray]:
    """决定传给 match() 的 prefilter_order（历史对照开关，见 VERIFICATION.md §8.3）。

    config: matching.template_prescreen: dinov2 | none
    - none：强制全模板逐一 MASt3R 匹配（历史对照口径：全部模板成对推理
      再打分，无 DINOv2 预筛）；
    - dinov2（默认）：template_ranking=dinov2 且定位阶段产出了排序时预筛
      只解码 Top-K，否则回退全解码（gt_mask/gt_bbox 等无 DINOv2 检索的
      定位路线 template_order 为 None）。

    未知取值显式 raise，绝不静默回退（本库一贯纪律）。
    `prescreen=none` + `ranking=dinov2` 这个组合同样 raise：预筛被跳过后
    Top-K 只能由 MASt3R 的 sim(m) 决定，`template_ranking: dinov2` 完全
    失效——静默生效的假象比报错危险得多（会让消融结论归错因）。
    """
    if prescreen not in ("dinov2", "none"):
        raise ValueError(
            f"未知 matching.template_prescreen: {prescreen!r}"
            f"（可选 dinov2 / none）")
    if prescreen == "none":
        if template_ranking == "dinov2":
            raise ValueError(
                "配置组合无效：matching.template_prescreen=none 会跳过 DINOv2 "
                "预筛，此时 matching.template_ranking=dinov2 无从生效（Top-K "
                "只能按 MASt3R sim(m) 选）。请改成 template_ranking: mast3r"
                "（历史对照口径），或把 template_prescreen 改回 dinov2。")
        return None
    if template_ranking == "dinov2" and template_order is not None:
        return template_order
    return None


class Mast3rMatcher:
    """MASt3R 匹配器：模板编码缓存 + 批量成对解码 + GPU 互最近邻。"""

    def __init__(self, cfg_matching: Dict, device: str = "cuda",
                 n_score_pixels: int = 2048):
        try:
            import torch
            from mast3r.model import AsymmetricMASt3R
        except ImportError as e:
            raise ImportError(f"{_MAST3R_HINT}\n原始错误: {e}") from e
        self.torch = torch
        self.device = device
        self.cfg = cfg_matching
        self.long_side = int(cfg_matching.get("image_size", 512))
        self.batch_size = int(cfg_matching.get("batch_size", 8))
        self.n_score_pixels = n_score_pixels

        ckpt = cfg_matching.get("mast3r_local_ckpt") or \
            cfg_matching.get("mast3r_checkpoint")
        self.model = AsymmetricMASt3R.from_pretrained(ckpt).to(device).eval()

        # 模板缓存（prepare_templates 填充）
        self._tmpl_feats = None      # 编码器 token 列表
        self._tmpl_pos = None
        self._tmpl_shapes = None
        self._tmpl_fg = None         # 每模板前景像素布尔图（模板原生分辨率）

    # ------------------------------------------------------------------
    def _to_tensor(self, img_u8: np.ndarray):
        """RGB uint8 → dust3r 归一化张量 (1,3,H,W)：(x/255 - 0.5)/0.5。

        与官方 ImgNorm 一致（dust3r/utils/image.py:23：
        Normalize(mean=0.5, std=0.5)）。
        """
        torch = self.torch
        x = torch.tensor(img_u8, dtype=torch.float32, device=self.device)
        x = (x / 255.0 - 0.5) / 0.5
        return x.permute(2, 0, 1)[None]

    def _encode(self, img_u8: np.ndarray):
        """单图过 ViT 编码器，返回 (feat, pos, true_shape)。

        _encode_image 签名与返回 (x, pos, None) 见 dust3r/model.py:128-140；
        true_shape 为 (B,2) 的 (H,W) 整型张量（dust3r/utils/image.py:122）。
        """
        torch = self.torch
        img = self._to_tensor(img_u8)
        true_shape = torch.tensor([img.shape[-2:]], dtype=torch.int32,
                                  device=self.device)
        with torch.no_grad():
            feat, pos, _ = self.model._encode_image(img, true_shape)
        return feat, pos, true_shape

    # ------------------------------------------------------------------
    def prepare_templates(self, images: np.ndarray, alphas: np.ndarray,
                          fg_thresh: float = 0.5):
        """预提取全部模板的编码器特征缓存（离线一次，供全部查询帧复用）。

        Args:
            images: (M,S,S,3) uint8 模板图（S=256，天然 16 的倍数）
            alphas: (M,S,S) 模板 alpha（前景掩码来源）
        """
        feats, poss, shapes, fgs = [], [], [], []
        for im, a in zip(images, alphas):
            f, p, s = self._encode(im)
            feats.append(f)
            poss.append(p)
            shapes.append(s)
            fgs.append(np.asarray(a, dtype=np.float32) > fg_thresh)
        self._tmpl_feats, self._tmpl_pos = feats, poss
        self._tmpl_shapes, self._tmpl_fg = shapes, fgs

    # ------------------------------------------------------------------
    def _decode_batch(self, fq, pq, sq, idxs):
        """一批 (query, template_i) 对过解码器+匹配头，i ∈ idxs。

        复刻官方 forward 的解码段（dust3r/model.py:199-210 →
        _decoder(f1,pos1,f2,pos2) dust3r/model.py:172-192；
        _downstream_head(head_num, [tok.float()...], true_shape)
        dust3r/model.py:193-197，fp32 autocast-off 同 dust3r/model.py:206、
        mast3r/model.py:208）。
        编码器输出已缓存（模板 40 次编码只做一次、查询每帧一次），解码器的
        成对交叉注意力必须逐对执行——但模板同为 256×256，token 形状一致，
        可按 batch_size 堆叠并行，查询侧 token 重复展开即可。

        Yields:
            (template_idx, desc_q (H,W,24), desc_t (S,S,24),
             pts3d_q (H,W,3), pts3d_t (S,S,3))；desc 由 reg_desc 归一到
            单位范数（mast3r/catmlp_dpt_head.py:19-24,36），点积即余弦
            相似度；pts3d 为各自相机系（几何一致性过滤用）
        """
        torch = self.torch
        b = len(idxs)
        ft = torch.cat([self._tmpl_feats[i] for i in idxs], dim=0)
        pt = torch.cat([self._tmpl_pos[i] for i in idxs], dim=0)
        st = torch.cat([self._tmpl_shapes[i] for i in idxs], dim=0)
        fq_b = fq.repeat(b, 1, 1)
        pq_b = pq.repeat(b, 1, 1)
        sq_b = sq.repeat(b, 1)
        with torch.no_grad():
            dec1, dec2 = self.model._decoder(fq_b, pq_b, ft, pt)
            with torch.autocast("cuda", enabled=False):
                res1 = self.model._downstream_head(
                    1, [tok.float() for tok in dec1], sq_b)
                res2 = self.model._downstream_head(
                    2, [tok.float() for tok in dec2], st)
        for bi, i in enumerate(idxs):
            yield (i, res1["desc"][bi], res2["desc"][bi],
                   res1["pts3d"][bi], res2["pts3d"][bi])

    # ------------------------------------------------------------------
    def match(self, query_crop_u8: np.ndarray, query_mask_crop: np.ndarray,
              top_k: int, sim_threshold: float, cycle_tau_px: float,
              n_sample: int, rng: Optional[np.random.Generator] = None,
              prefilter_order: Optional[np.ndarray] = None
              ) -> Tuple[List[TemplateMatch], Tuple[float, float], np.ndarray]:
        """查询裁剪区 vs 模板：打分 → Top-K → 稠密对应。

        Args:
            prefilter_order: DINOv2 相似度降序模板下标。给定时
                只解码前 top_k 个模板（template_ranking=dinov2）；为 None 时
                解码全部模板再按 sim(m) 选 Top-K（template_ranking=mast3r）。

        Returns:
            matches: Top-K 模板的 TemplateMatch 列表（降序）
            query_scale: (sx, sy)，resize 坐标 → 裁剪区坐标需除以该比例
            scores: (M,) 模板的 sim(m) 分数（未解码模板为 -inf）
        """
        import cv2
        torch = self.torch
        if self._tmpl_feats is None:
            raise RuntimeError("先调用 prepare_templates() 缓存模板特征")
        if rng is None:
            rng = np.random.default_rng(0)

        q_img, (sx, sy) = _resize_to_multiple16(query_crop_u8, self.long_side)
        q_mask = cv2.resize(query_mask_crop.astype(np.uint8),
                            (q_img.shape[1], q_img.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        fq, pq, sq = self._encode(q_img)

        ys, xs = np.nonzero(q_mask)
        if len(ys) < 16:
            return [], (sx, sy), np.full(len(self._tmpl_feats), -np.inf)
        pix_q_all = np.stack([xs, ys], axis=1)            # (Nq,2) x,y
        flat_q_all = ys * q_img.shape[1] + xs
        # 打分用下采样像素（估计 sim(m) 均值足够，控制解码显存）
        if len(flat_q_all) > self.n_score_pixels:
            sub = rng.choice(len(flat_q_all), self.n_score_pixels,
                             replace=False)
        else:
            sub = np.arange(len(flat_q_all))
        flat_q_score = torch.tensor(flat_q_all[sub], device=self.device)
        flat_q_full = torch.tensor(flat_q_all, device=self.device)

        n_tmpl = len(self._tmpl_feats)
        scores = np.full(n_tmpl, -np.inf)
        desc_cache = {}      # template_idx -> (desc_q_fg, desc_t_fg, pix_t)

        # DINOv2 预筛时只解码 Top-K 个模板（K<M 真正省算）
        decode_idxs = decode_template_indices(n_tmpl, top_k, prefilter_order)

        # ---- 第一遍：待解码模板批量解码 + 打分，desc 暂存 CPU fp16 ----
        # 记录 top1 模板的稠密 desc（查询全图 + 模板前景），供 solve 阶段
        # 引导式对应精化（粗位姿投影 → 局部窗口重匹配）复用，避免重解码
        top_full = None      # (template_idx, desc_q_full, desc_t_fg, pix_t)
        for start in range(0, len(decode_idxs), self.batch_size):
            idxs = decode_idxs[start:start + self.batch_size]
            for i, desc_q, desc_t, pts3d_q, pts3d_t in self._decode_batch(
                    fq, pq, sq, idxs):
                dq = desc_q.reshape(-1, desc_q.shape[-1])
                fg_t = self._tmpl_fg[i]
                tys, txs = np.nonzero(fg_t)
                if len(tys) == 0:
                    continue
                flat_t = torch.tensor(tys * fg_t.shape[1] + txs,
                                      device=self.device)
                dt = desc_t.reshape(-1, desc_t.shape[-1])[flat_t]   # (Nt,24)
                # sim(m) = mean_y max_{y'} S(y,y')
                sim_sub = dq[flat_q_score] @ dt.T
                scores[i] = float(sim_sub.max(dim=1).values.mean())
                # 前景像素的相机系 3D（几何一致性过滤用；只取 fg，省内存）
                p3_q = pts3d_q.reshape(-1, 3)[flat_q_full].float().cpu().numpy()
                p3_t = pts3d_t.reshape(-1, 3)[flat_t].float().cpu().numpy()
                desc_cache[i] = (
                    dq[flat_q_full].half().cpu(), dt.half().cpu(),
                    np.stack([txs, tys], axis=1), p3_q, p3_t)
                if prefilter_order is not None:
                    is_top = (i == int(decode_idxs[0]))
                else:
                    is_top = (top_full is None
                              or scores[i] > scores[top_full[0]])
                if is_top:
                    top_full = (int(i),
                                desc_q.float().cpu().numpy().astype(np.float16),
                                dt.half().cpu().numpy(),
                                np.stack([txs, tys], axis=1))
                del desc_q, desc_t, dq, dt, sim_sub, p3_q, p3_t

        # ---- 第二遍：Top-K 稠密互最近邻 + cycle 过滤 + 阈值 + 采样 ----
        if prefilter_order is not None:
            # DINOv2 排序即 Top-K，保持其顺序（已只解码这 K 个）
            order = np.array(decode_idxs, dtype=int)
        else:
            # MASt3R 排序：全解码后按 sim(m) 取 Top-K
            order = np.argsort(-scores)[:min(top_k, n_tmpl)]
        geom_on = bool(self.cfg.get("geom_filter", True))
        geom_tau = float(self.cfg.get("geom_tau_frac", 0.08))
        geom_iters = int(self.cfg.get("geom_iters", 120))
        sel = [int(i) for i in order
               if np.isfinite(scores[i]) and i in desc_cache]
        matches = []
        if not sel:
            return matches, (sx, sy), scores, top_full

        if bool(self.cfg.get("fusion", True)):
            # ---- 融合版（OnePose++/MixRI 零样本版）：模板侧 desc 拼接，
            # 一次全局互最近邻。desc_q 由查询图解码、对所有模板相同，只需
            # 一份；每个查询像素与"全部 Top-K 模板的前景 desc"竞争最近邻，
            # 跨模板取全局最优（原版逐模板独立 NN，同一查询像素可重复匹配
            # 多个模板且各自只看局部最优）。对应按模板归属拆回落盘，下游
            # PnP 接口不变。融合池限 fusion_topk（默认 12，与联合 PnP 的
            # 模板数一致）：全 80 模板拼接的 desc 矩阵 ~40 万列，matmul
            # 单块即超 6GB，易 OOM。
            sel = sel[:int(self.cfg.get("fusion_topk", 12))]
            dq_fg = desc_cache[sel[0]][0].to(self.device).float()
            lens = [len(desc_cache[i][2]) for i in sel]
            dt_fg = torch.cat(
                [desc_cache[i][1].to(self.device).float() for i in sel],
                dim=0)
            pix_t_all = np.concatenate(
                [desc_cache[i][2] for i in sel], axis=0)
            tmpl_of = np.repeat(np.array(sel, dtype=np.int64), lens)
            chunk = 4096
            nn_q2t = torch.empty(len(dq_fg), dtype=torch.long,
                                 device=self.device)
            sims_fwd = torch.empty(len(dq_fg), dtype=torch.float32,
                                   device=self.device)
            best_col = torch.full((len(dt_fg),), float("-inf"),
                                  device=self.device)
            nn_t2q = torch.zeros(len(dt_fg), dtype=torch.long,
                                 device=self.device)
            for c0 in range(0, len(dq_fg), chunk):
                sim_c = dq_fg[c0:c0 + chunk] @ dt_fg.T   # (chunk, Nt_sum)
                nn_c = sim_c.argmax(dim=1)
                nn_q2t[c0:c0 + chunk] = nn_c
                sims_fwd[c0:c0 + chunk] = sim_c.gather(
                    1, nn_c[:, None])[:, 0]
                col_max, col_idx = sim_c.max(dim=0)
                upd = col_max > best_col
                best_col[upd] = col_max[upd]
                nn_t2q[upd] = col_idx[upd] + c0
                del sim_c
            nn_q2t = nn_q2t.cpu().numpy()
            nn_t2q = nn_t2q.cpu().numpy()
            sims_fwd = sims_fwd.cpu().numpy()
            idx_q = np.arange(len(pix_q_all))
            keep = cycle_consistency_filter(
                pix_q_all.astype(np.float64), idx_q, nn_q2t, nn_t2q,
                tau_px=cycle_tau_px)
            ok = keep & (sims_fwd > sim_threshold)
            iq, it = idx_q[ok], nn_q2t[ok]
            ss = sims_fwd[ok]
            for j, i in enumerate(sel):
                m_ok = tmpl_of[it] == i
                if m_ok.sum() == 0:
                    continue
                p3q_all = desc_cache[sel[0]][3]          # 查询侧 3D 与模板无关
                if p3q_all is not None:
                    p2, p3, ss_, p3q = sample_correspondences(
                        pix_q_all[iq[m_ok]].astype(np.float64),
                        pix_t_all[it[m_ok]].astype(np.float64), ss[m_ok],
                        n_sample=n_sample, rng=rng,
                        extras=[p3q_all[iq[m_ok]]])
                else:
                    p2, p3, ss_ = sample_correspondences(
                        pix_q_all[iq[m_ok]].astype(np.float64),
                        pix_t_all[it[m_ok]].astype(np.float64), ss[m_ok],
                        n_sample=n_sample, rng=rng)
                    p3q = None
                matches.append(TemplateMatch(
                    template_idx=i, score=float(scores[i]),
                    pix_q=p2, pix_t=p3, sims=ss_, pts3d_q=p3q))
            return matches, (sx, sy), scores, top_full

        for i in order:
            i = int(i)
            if not np.isfinite(scores[i]) or i not in desc_cache:
                continue
            dq_fg, dt_fg, pix_t, p3_q, p3_t = desc_cache[i]
            dq_fg = dq_fg.to(self.device).float()
            dt_fg = dt_fg.to(self.device).float()
            # 稠密互最近邻：相似度矩阵按查询侧分块算，避免一次性建
            # (Nq, Nt) 全矩阵（ape 大特写帧可达 ~70 亿元素 ≈ 29GB，OOM）。
            # 列 argmax 用严格大于才更新，保留最先出现的行（等价全量
            # argmax 的最小索引语义）。
            chunk = 4096
            nn_q2t = torch.empty(len(dq_fg), dtype=torch.long,
                                 device=self.device)
            sims_fwd = torch.empty(len(dq_fg), dtype=torch.float32,
                                   device=self.device)
            best_col = torch.full((len(dt_fg),), float("-inf"),
                                  device=self.device)
            nn_t2q = torch.zeros(len(dt_fg), dtype=torch.long,
                                 device=self.device)
            for c0 in range(0, len(dq_fg), chunk):
                sim_c = dq_fg[c0:c0 + chunk] @ dt_fg.T   # (chunk, Nt)
                nn_c = sim_c.argmax(dim=1)
                nn_q2t[c0:c0 + chunk] = nn_c
                sims_fwd[c0:c0 + chunk] = sim_c.gather(1, nn_c[:, None])[:, 0]
                col_max, col_idx = sim_c.max(dim=0)
                upd = col_max > best_col
                best_col[upd] = col_max[upd]
                nn_t2q[upd] = col_idx[upd] + c0
                del sim_c
            nn_q2t = nn_q2t.cpu().numpy()
            nn_t2q = nn_t2q.cpu().numpy()
            sims_fwd = sims_fwd.cpu().numpy()

            idx_q = np.arange(len(pix_q_all))
            # 共享纯逻辑：cycle consistency（τ px，互最近邻为其 τ=0 特例）
            keep = cycle_consistency_filter(
                pix_q_all.astype(np.float64), idx_q, nn_q2t, nn_t2q,
                tau_px=cycle_tau_px)
            if geom_on and len(nn_q2t) > 0:
                # MASt3R 两视图 3D 一致性过滤（官方 fast_nn 几何验证步）：
                # desc 最近邻在重复纹理下会找错同名点，cycle 同样失效；
                # 用 RANSAC-相似变换按 3D 残差滤掉几何不一致的对应
                from .correspondence import geometric_consistency_filter
                keep = keep & geometric_consistency_filter(
                    p3_q, p3_t, nn_q2t, sims_fwd, rng,
                    tau_obj_frac=geom_tau, ransac_iters=geom_iters)
            ok = keep & (sims_fwd > sim_threshold)     # 相似度阈值过滤
            iq = idx_q[ok]
            it = nn_q2t[ok]
            if p3_q is not None:
                p2, p3, ss, p3q = sample_correspondences(
                    pix_q_all[iq].astype(np.float64),
                    pix_t[it].astype(np.float64), sims_fwd[ok],
                    n_sample=n_sample, rng=rng,
                    extras=[p3_q[iq]])
            else:
                p2, p3, ss = sample_correspondences(
                    pix_q_all[iq].astype(np.float64),
                    pix_t[it].astype(np.float64), sims_fwd[ok],
                    n_sample=n_sample, rng=rng)
                p3q = None
            matches.append(TemplateMatch(
                template_idx=int(i), score=float(scores[i]),
                pix_q=p2, pix_t=p3, sims=ss, pts3d_q=p3q))
        return matches, (sx, sy), scores, top_full
