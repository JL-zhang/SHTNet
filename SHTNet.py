"""
SHTNet: SAM3-based Hypothesis Testing Network for Salient and Camouflaged Object Detection
==========================================================================================
论文实现 (sam_cod_2.pdf, Sec. III Method)。

三级结构:
  Stage 1  Structural Evidence Extraction
           · PVTv2-B3 主干 -> {X1,X2,X3,X4}                        (Sec. III-B-1)
           · SAM3 先验  PS -> 多尺度 {P1,P2,P3,P4} = F_map(PS)
           · BEM (Boundary Evidence Mining): 跨层差异挖掘边界证据 EI (Sec. III-B-2, Eq.1-7)
  Stage 2  Structural Hypothesis Testing
           · HRH (Hypothesis Rejection Head): 形态学梯度 + 局部 patch 余弦相似
             -> 连续 Rejection Field R = 1 - sigmoid(<eS_hat, eI_hat> / tau) (Sec. III-C, Eq.8-11)
  Stage 3  Rejection-Guided Structure Adaptation
           · HGM (Hypothesis-guided Modulation): Eq.12-16
           · MSM (Multi-scale Structure-aware Interaction): Eq.17-21
  Loss     L_total = L_str + lambda_e * L_dice                       (Sec. III-E, Eq.22-24)

接口与 prior/priornet_v2.py 保持一致, 可直接被 etrainV2 风格的脚本调用:
    forward(x, depth, P_sam) -> o3, o2, o1, edge_map
其中 o3/o2/o1 对应论文的 O1/O2/O3 (1/16, 1/8, 1/4 分辨率), edge_map 是 BEM 的边界证据。

依赖: 仅 torch + timm + prior/pvt_v2_b3.py (不依赖 mmcv/mmseg)

自检:
    python prior/SHTNet.py
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接 `python prior/SHTNet.py` 运行时, 仓库根目录不在 sys.path 里
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prior.pvt_v2_b3 import build_pvt_v2_b3, PVT_V2_B3_CONFIG
except ImportError:
    from pvt_v2_b3 import build_pvt_v2_b3, PVT_V2_B3_CONFIG

try:
    from prior.module import AdaptiveCombiner
except ImportError:
    from module import AdaptiveCombiner


__all__ = ['SHTNet', 'BEM', 'HRH', 'HGM', 'MSM', 'SimAM', 'CBR', 'MorphologicalEdge',
           'Net']


# ============================================================
# 基础算子
# ============================================================

class CBR(nn.Module):
    """论文 Eq.(1): ReLU(BN(Conv(nxn)(·)))  —— convolution + BN + ReLU"""

    def __init__(self, in_ch, out_ch, k=1, stride=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=stride,
                      padding=dilation * (k // 2), dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SimAM(nn.Module):
    """参数无关 3-D 注意力 (论文 Fig.3 中的 SimAM, 对应 Yang et al. 2021)"""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)


class MorphologicalEdge(nn.Module):
    """论文 Eq.(9): ES = delta_k(PS) - eps_k(PS), 用 max_pool 实现膨胀/腐蚀, 无学习参数"""

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.pad = k // 2

    def forward(self, mask):
        dilated = F.max_pool2d(mask, kernel_size=self.k, stride=1, padding=self.pad)
        eroded = -F.max_pool2d(-mask, kernel_size=self.k, stride=1, padding=self.pad)
        return (dilated - eroded).clamp(0, 1)


def patchify(x, p):
    """
    Eq.(10) 用的 p×p 邻域提取: [B,C,H,W] -> [B, C*p*p, H, W]
    用 unfold 实现 (无参数), 每个 (x,y) 位置取 p×p 邻域展平成向量。
    """
    b, c, h, w = x.shape
    pad = p // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode='replicate')
    # [B, C, H, W, p*p]
    xp = F.unfold(xp, kernel_size=p, stride=1).view(b, c, p * p, h, w)
    return xp.reshape(b, c * p * p, h, w)


# ============================================================
# Stage 1: BEM —— Boundary Evidence Mining Module  (Eq.2-7)
# ============================================================

class DCBlock(nn.Module):
    """
    Difference Computation block (Eq.2-4):
        X~    = SimAM(X)
        D     = X~ - AvgPool(X~)              (逐元素相减)
        A     = Sigmoid(Conv(D))
        DC(X) = SimAM( M * X~ ),  M = A * X~

    Eq.(5)(6) 里 DC(X_i) 要与 phi(X_j) 逐元素相乘, 因此这里先算完差异,
    再用 CBR(1x1) 把通道数投影到 out_ch, 使两路对齐。
    """

    def __init__(self, ch, out_ch, k=3):
        super().__init__()
        self.simam1 = SimAM()
        self.avg = nn.AvgPool2d(3, stride=1, padding=1)
        self.conv = nn.Conv2d(ch, ch, k, padding=k // 2, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.simam2 = SimAM()
        self.proj = CBR(ch, out_ch, 1)

    def forward(self, x):
        xs = self.simam1(x)
        d = xs - self.avg(xs)
        a = self.sigmoid(self.conv(d))
        m = a * xs
        return self.proj(self.simam2(m * xs))


class BEM(nn.Module):
    """
    Boundary Evidence Mining Module (论文 Sec. III-B-2, Eq.5-7)
        X12 = phi(X1) * DC(X2)                       Eq.(5)
        X34 = phi(X3) * DC(X4)                       Eq.(6)
        EI  = psi( phi(X12) - X34 )                  Eq.(7)
    其中 phi(·) 是 CBR(1x1) + 上采样; psi(·) 是 CBR(1x1) 后接 1x1 卷积 -> 单通道边界证据。
    """

    def __init__(self, c1, c2, c3, c4, mid=128):
        super().__init__()
        # DC 分支 (对 X2 / X4 做差异计算, 通道数对齐到 mid)
        self.dc2 = DCBlock(c2, mid)
        self.dc4 = DCBlock(c4, mid)

        # phi(·): 1x1 投影, 统一到 mid 通道
        self.phi1 = CBR(c1, mid, 1)
        self.phi2 = CBR(c2, mid, 1)
        self.phi3 = CBR(c3, mid, 1)
        self.phi4 = CBR(c4, mid, 1)

        # psi(·): CBR(1x1) + Conv(1x1) -> 1 通道
        self.psi = CBR(mid, mid, 1)
        self.out = nn.Conv2d(mid, 1, 1)

    def forward(self, x1, x2, x3, x4):
        # Eq.(5): X12 = phi(X1) * DC(X2)
        x12 = self.up_to(self.phi1(x1), x2) * self.dc2(x2)
        # Eq.(6): X34 = phi(X3) * DC(X4)
        x34 = self.up_to(self.phi3(x3), x4) * self.dc4(x4)
        # Eq.(7): EI = psi( phi(X12) - X34 )
        x12p = self.up_to(self.phi2(x12), x34)
        ei = self.out(self.psi(x12p - x34))
        return ei, x12, x34

    @staticmethod
    def up_to(src, ref):
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return src


# ============================================================
# Stage 2: HRH —— Hypothesis Rejection Head  (Eq.8-11)
# ============================================================

class HRH(nn.Module):
    """
    比较 SAM3 预测的边界 ES 与 BEM 证据 EI 的局部结构一致性, 输出连续 Rejection Field R。

        ES      = delta_k(PS) - eps_k(PS)                        Eq.(9)
        eq      = eq - mean(eq)                                  Eq.(10) 去均值
        eq_hat  = eq / (||eq||_2 + eps)                           Eq.(10) L2 归一化
        R       = 1 - sigmoid( <eS_hat, eI_hat> / tau )           Eq.(11)

    patch 均值/范数用 grouped conv 等价实现 (p*p 组, 每组 1 个通道), 无额外参数。
    """

    def __init__(self, k=3, p=3, tau=0.05, eps=1e-6, dilation=True,
                 flat_thr=1e-6, ratio_flat=0.25, compare_size=168):
        """
        compare_size: 假设检验的执行尺度 (ES 与 EI 都在该尺度上比 p×p patch)。
                      None 表示跟随 P_sam 的尺度 (即 672, 与原行为一致)。

        为什么要单独设对比尺度:
            EI (BEM/EAM 的边界证据) 若在 168 上产生再上采样到 672, 它是**模糊**的;
            而 ES = morph(P_sam) 是在 672 上从清晰的 SAM 概率图算出来的, 是**锐利**的。
            拿模糊图与锐利图比 3×3 patch, 尺度上不对等 —— 会系统性地抬高 R。
            把两者都放到边界证据的原生尺度 (168) 再比, 比较才是同尺度的。
        """
        super().__init__()
        self.morph = MorphologicalEdge(k)
        self.p = p
        self.tau = tau
        self.eps = eps
        # 见 forward 里的说明: 对 1 像素宽的边界做 p×p patch 比较时,
        # 先在 patch 范围内取 max 才能让"局部边界是否存在"可比, 而不是要求像素级对齐。
        self.dilation = dilation
        # 两边都接近零 -> 视为都平坦 (绝对下限, 只用来兜底)
        self.flat_thr = flat_thr
        # 单边平坦的相对判据: min(范数)/max(范数) 小于该比例即认为一边无结构
        self.ratio_flat = ratio_flat
        # 假设检验的对比尺度; None = 跟随 P_sam
        self.compare_size = compare_size

    def _mean_kernel(self, device, dtype):
        """
        在 unfold 出来的 [B, p*p, H, W] 上做「每个 p×p patch 内部求均值」。
        patchify 用 unfold + reshape 把 (kh,kw) 展平成 p*p 通道, 所以这里用
        全 1/p^2 的 p×p 核做 conv, 得到的就是每个 patch 的均值。
        (注意: 核必须归一化, 否则算出来是"和"而不是"均值"; 也不能只点中心。)
        """
        pp = self.p * self.p
        return torch.full((pp, 1, self.p, self.p), 1.0 / pp,
                          device=device, dtype=dtype)

    def forward(self, P_sam, E_I):
        """
        P_sam : [B,1,H,W] SAM3 预测 (概率图)
        E_I   : [B,1,H,W] BEM 边界证据 (已 sigmoid)
        返回 R: [B,1,H,W], 越大表示局部结构越不一致

        比较在 self.compare_size 上执行 (默认 168); 若为 None 则跟随 P_sam 的尺度。
        返回的 R 会被双线性拉回 P_sam 的尺度, 保证调用方拿到的 R 与 P_sam/特征对齐。
        """
        out_size = P_sam.shape[2:]                       # R 最终要回到这个尺度

        # 统一空间尺度后再算(见 __init__ 里 compare_size 的说明)
        if self.compare_size is not None:
            s = self.compare_size
            if P_sam.shape[2] != s:
                P_sam = F.interpolate(P_sam, size=(s, s), mode='bilinear', align_corners=False)
            if E_I.shape[2] != s:
                E_I = F.interpolate(E_I, size=(s, s), mode='bilinear', align_corners=False)
        elif P_sam.shape[2:] != E_I.shape[2:]:
            E_I = F.interpolate(E_I, size=P_sam.shape[2:], mode='bilinear', align_corners=False)

        E_S = self.morph(P_sam)                      # Eq.(9)

        p = self.p
        pp = p * p

        # 边界图只有 ~1 像素宽, 一个 p×p patch 里通常只有 1 个非零像素。
        # 去均值后这种 patch 会变成 one-hot 向量, 两条边界只有在"落在 patch 内
        # 完全相同的那一格"时内积才非零, 否则恒为 0 -> R 恒等于 0.5, 拒绝场失效。
        # 因此先在 patch 范围内取 max (对边界做轻微膨胀), 让比较的是
        # "该邻域内是否存在边界"这种局部结构模式, 与论文"比较局部边界模式"的意图一致。
        #
        # 注意: 必须对 ES 和 EI **对称**膨胀。只膨胀一边会造成系统性错配 ——
        # 即使 ES == EI, 比较的也是"膨胀后的 ES"和"未膨胀的 EI", R 被抬高。
        if self.dilation:
            E_S = F.max_pool2d(E_S, p, stride=1, padding=p // 2)
            E_I = F.max_pool2d(E_I, p, stride=1, padding=p // 2)

        eS = patchify(E_S, p)                        # [B, pp, H, W]
        eI = patchify(E_I, p)

        # --- Eq.(10): 去均值 + L2 归一化 ---
        # 注意: 这里的"patch 均值"必须与 patchify 取到的 patch 内容严格对应。
        # patchify 内部用的是 replicate padding, 而 F.avg_pool2d 默认是 zero padding,
        # 两者在图像边界处不一致, 会让均值与 patch 内容对不上, 所以这里显式用
        # replicate padding 的卷积求均值 (核为全 1/p^2, 分 p*p 组)。
        pad = p // 2
        pad4 = (pad,) * 4
        # 单通道图的局部均值: 核为全 1/p^2 的 p×p, 一次普通卷积即可 (1 组)
        mean_k = torch.full((1, 1, p, p), 1.0 / pp, device=E_S.device, dtype=E_S.dtype)
        mS = F.conv2d(F.pad(E_S, pad4, mode='replicate'), mean_k)
        mI = F.conv2d(F.pad(E_I, pad4, mode='replicate'), mean_k)

        eS = eS - mS
        eI = eI - mI

        nS = eS.norm(dim=1, keepdim=True)
        nI = eI.norm(dim=1, keepdim=True)
        uS = eS / (nS + self.eps)
        uI = eI / (nI + self.eps)

        # --- Eq.(11): 局部模式余弦相似 ---
        cos = (uS * uI).sum(dim=1, keepdim=True)      # [B,1,H,W]

        # --- 退化 patch 的处理 ---
        # 去均值后, "平坦" patch (局部全 0 或全 1) 会变成全零向量, 此时余弦无定义
        # (恒为 0 -> R 恒为 0.5), 拒绝场会大面积失效。而平坦 patch 恰恰最有信息量:
        #   两边都平坦 且响应接近  -> 一致, 应被接受 (R 低)
        #   只有一边平坦           -> 一边有边界一边没有, 结构冲突 (R 高)
        #   两边都平坦 但响应差得远 -> 例如"先验有目标、证据没有", 也算分歧
        #
        # 【判据必须尺度无关】patch 范数的量级随输入对比度变化很大
        # (实测中位数可低至 2e-2), 用固定绝对阈值会让大量 patch 恰好落在阈值两侧,
        # 于是"只有一边平坦"被误触发 —— 即使两路输入完全相同, R 也会被抬到 1。
        # 因此用相对判据: 小范数 / 大范数 < ratio_flat 才算单边平坦。
        n_max = torch.maximum(nS, nI)
        n_min = torch.minimum(nS, nI)
        both_flat = (n_max <= self.flat_thr).float()             # 两边都接近零
        one_flat = (n_min <= self.ratio_flat * n_max).float()    # 一边有结构一边没有
        one_flat = one_flat * (1 - both_flat)                    # 排除"都平坦"

        # 两边都平坦: 用 patch 自身的均值(即局部响应值)比较
        agree_flat = (1.0 - 2.0 * (mS - mI).abs()).clamp(-1, 1)

        cos = (1 - both_flat - one_flat) * cos + both_flat * agree_flat

        R = 1.0 - torch.sigmoid(cos / self.tau)
        # 一边有边界、一边没有 -> 局部结构冲突, 直接给强拒绝
        R = (1 - one_flat) * R + one_flat * 1.0

        # 比较是在 compare_size 上做的, 这里把 R 拉回 P_sam 原本的尺度,
        # 使 HGM 里的 Rn 与特征 Xn 空间对齐 (HGM 自己也会对齐, 这里先回到一致的尺度)
        if R.shape[2:] != out_size:
            R = F.interpolate(R, size=out_size, mode='bilinear', align_corners=False)
        return R


# ============================================================
# Stage 3a: HGM —— Hypothesis-guided Modulation  (Eq.12-16)
# ============================================================

class HGM(nn.Module):
    """
        P~n = U(Pn) * Xn + Xn
        E~n = U(EI) * Xn + Xn                       Eq.(12)   <-- 论文此处按等号原样实现
        Zg  = AC(P~g, E~g)                          Eq.(13)(14)
        Fc  = CBR1x1(Cat[Z1..ZG])                   Eq.(15)
        Fo  = (1-R)*Fc + R*CBR1x1(Xn)               Eq.(16)
    """

    def __init__(self, ch, G=4):
        super().__init__()
        assert ch % G == 0, 'HGM: 通道数 {} 需能被组数 {} 整除'.format(ch, G)
        self.G = G
        self.ac = nn.ModuleList([AdaptiveCombiner() for _ in range(G)])
        self.proj = CBR(ch, ch, 1)
        self.vis = CBR(ch, ch, 1)

    def forward(self, X_n, P_n, E_I, R_n):
        # 空间对齐
        if P_n.shape[2:] != X_n.shape[2:]:
            P_n = F.interpolate(P_n, size=X_n.shape[2:], mode='bilinear', align_corners=False)
        if E_I.shape[2:] != X_n.shape[2:]:
            E_I = F.interpolate(E_I, size=X_n.shape[2:], mode='bilinear', align_corners=False)
        if R_n.shape[2:] != X_n.shape[2:]:
            R_n = F.interpolate(R_n, size=X_n.shape[2:], mode='bilinear', align_corners=False)

        # Eq.(12)
        P_t = P_n * X_n + X_n                       # 1 通道先验, 广播到 C
        E_t = E_I * X_n + X_n                       # 同上, 按论文等号原样

        # Eq.(13)(14)
        P_g = torch.chunk(P_t, self.G, dim=1)
        E_g = torch.chunk(E_t, self.G, dim=1)
        Z = [self.ac[g](P_g[g], E_g[g]) for g in range(self.G)]

        # Eq.(15)
        Fc = self.proj(torch.cat(Z, dim=1))

        # Eq.(16)
        return (1.0 - R_n) * Fc + R_n * self.vis(X_n)


# ============================================================
# Stage 3b: MSM —— Multi-scale Structure-aware Interaction  (Eq.17-21)
# ============================================================

class MSM(nn.Module):
    """
        Fn   = Fn_o + U(Fn+1_o)                     Eq.(17)
        {Fi} = Split(Fn, 4)                          Eq.(18)
        F1=F1, Fi = Fi + F(i-1)                      Eq.(19)
        Fa   = Conv1x1(Cat[{CBR3x3(Fi)}])            Eq.(20)
        On   = CBR3x3(Fa + Fn)                       Eq.(21)
    """

    def __init__(self, ch):
        super().__init__()
        assert ch % 4 == 0, 'MSM: 通道数 {} 需能被 4 整除'.format(ch)
        self.ch = ch
        self.cbr = nn.ModuleList([CBR(ch // 4, ch // 4, 3) for _ in range(4)])
        self.proj = nn.Conv2d(ch, ch, 1)
        self.refine = CBR(ch, ch, 3)
        # 相邻尺度通道数不同 (深层更宽), 首次前向时按需建一个 1x1 对齐卷积
        self._align = None

    def _get_align(self, in_ch, ref):
        if self._align is None or self._align.in_channels != in_ch:
            self._align = nn.Conv2d(in_ch, self.ch, 1, bias=False).to(ref.device, ref.dtype)
        return self._align

    def forward(self, F_n, F_next=None):
        # Eq.(17)
        F_agg = F_n
        if F_next is not None:
            if F_next.shape[2:] != F_n.shape[2:]:
                F_next = F.interpolate(F_next, size=F_n.shape[2:],
                                       mode='bilinear', align_corners=False)
            if F_next.shape[1] != F_n.shape[1]:
                F_next = self._get_align(F_next.shape[1], F_n)(F_next)
            F_agg = F_n + F_next

        # Eq.(18)
        parts = list(torch.chunk(F_agg, 4, dim=1))

        # Eq.(19): 逐组累积上下文
        ctx = []
        acc = None
        for i in range(4):
            acc = parts[i] if acc is None else acc + parts[i]
            ctx.append(acc)

        # Eq.(20)
        Fa = self.proj(torch.cat([self.cbr[i](ctx[i]) for i in range(4)], dim=1))

        # Eq.(21)
        return self.refine(Fa + F_agg)


# ============================================================
# SHTNet 主体
# ============================================================

class SHTNet(nn.Module):
    """
    Args:
        fun_str          : 主干名, 目前支持 'pvt_v2_b3'
        pvt_checkpoint   : PVTv2-B3 预训练权重路径, None 时按 net/config.py 查找
        pretrained       : 是否加载主干预训练权重
        hrh_k / hrh_p    : 形态学结构元尺寸 k 与 patch 尺寸 p (论文: 奇数)
        hrh_tau          : Eq.(11) 的锐度参数 tau
        hrh_compare_size : 假设检验 (HRH) 的执行尺度, 默认 168。ES 与 EI 都拉到该尺度
                           再比 patch, 避免"锐利的 672 SAM 边界"对上"上采样后模糊的 EI"。
                           None = 跟随 P_sam 的尺度 (旧行为, 672)。
        groups           : HGM 的通道分组数 G
        edge_supervision : 'gt'  = Edge/ 目录的 GT 边界 (BEM 的监督目标)
                           'sam' = SAM3 先验的形态学边界 (无额外标注时可用)
    """

    def __init__(self,
                 fun_str='pvt_v2_b3',
                 pvt_checkpoint=None,
                 pretrained=True,
                 hrh_k=3,
                 hrh_p=3,
                 hrh_tau=0.05,
                 hrh_compare_size=168,
                 groups=4,
                 edge_supervision='gt',
                 native_scale=True):
        """
        关于 hrh_tau (论文 Eq.11 的 tau):
            实测在边界图上 cos 通常落在 [0.6, 1.0], 即 (1-cos) 的动态范围约 [0, 0.4]。
            tau 直接决定 R 的动态范围: tau 太大会让 R 恒接近 0 (拒绝场失效),
            tau 太小会让 R 饱和成 0/1 (退化成硬判决)。
            论文写明 k、p、tau 由验证集选定, 所以 tau 属于需要按数据集调的超参;
            这里给的经验默认值是 0.05。
        """
        super().__init__()

        assert edge_supervision in ('gt', 'sam'), \
            "edge_supervision 只支持 'gt' 或 'sam'"
        self.edge_supervision = edge_supervision
        self.native_scale = native_scale

        # ---------- Stage 1: 特征提取 ----------
        self.backbone_name = fun_str
        self.pvt, self.embed_dims = self._build_backbone(fun_str, pvt_checkpoint, pretrained)
        c1, c2, c3, c4 = self.embed_dims          # PVTv2-B3: 64/128/320/512
        self.embed_dims = list(self.embed_dims)

        # ---------- 各尺度通道数: 64 / 128 / 256 / 256 ----------
        self.dims = [64, 128, 256, 256]
        self.reduce1 = CBR(c1, self.dims[0], 1)
        self.reduce2 = CBR(c2, self.dims[1], 1)
        self.reduce3 = CBR(c3, self.dims[2], 1)
        self.reduce4 = CBR(c4, self.dims[3], 1)

        # 先验映射 F_map(PS) -> {P1..P4} 由 forward 中按尺度插值实现 (无参数)

        # ---------- Stage 1: BEM (在原始主干特征上做跨层差异) ----------
        self.bem = BEM(c1, c2, c3, c4, mid=128)

        # ---------- Stage 2: HRH ----------
        self.hrh = HRH(k=hrh_k, p=hrh_p, tau=hrh_tau, compare_size=hrh_compare_size)

        # ---------- Stage 3: HGM + MSM ----------
        self.hgm1 = HGM(self.dims[0], G=groups)
        self.hgm2 = HGM(self.dims[1], G=groups)
        self.hgm3 = HGM(self.dims[2], G=groups)
        self.hgm4 = HGM(self.dims[3], G=groups)

        self.msm1 = MSM(self.dims[0])
        self.msm2 = MSM(self.dims[1])
        self.msm3 = MSM(self.dims[2])
        self.msm4 = MSM(self.dims[3])

        # 像素级跨尺度组合 (把细粒度 MSM 输出融合进深层的预测)
        # 结构对应 Eq.(17), 权重取自 AC 的参数, 这里用普通 1x1 卷积保持简洁
        self.comb4 = nn.Conv2d(self.dims[3], self.dims[3], 1)
        self.comb3 = nn.Conv2d(self.dims[3] + self.dims[2], self.dims[2], 1)
        self.comb2 = nn.Conv2d(self.dims[2] + self.dims[1], self.dims[1], 1)
        self.comb1 = nn.Conv2d(self.dims[1] + self.dims[0], self.dims[0], 1)

        # ---------- 三个预测头 (论文 Eq.22 的 O1,O2,O3) ----------
        self.head3 = nn.Sequential(CBR(self.dims[2], 64, 3), nn.Conv2d(64, 1, 1))
        self.head2 = nn.Sequential(CBR(self.dims[1], 64, 3), nn.Conv2d(64, 1, 1))
        self.head1 = nn.Sequential(CBR(self.dims[0], 64, 3), nn.Conv2d(64, 1, 1))

    # --------------------------------------------------------
    def _build_backbone(self, fun_str, pvt_checkpoint=None, pretrained=True):
        if fun_str != 'pvt_v2_b3':
            raise NotImplementedError(
                'SHTNet 目前只内置 pvt_v2_b3, 收到 {!r}'.format(fun_str))

        cfg = PVT_V2_B3_CONFIG
        model, embed_dims = build_pvt_v2_b3(
            num_classes=0,
            pretrained=None,
            drop_path_rate=cfg['drop_path_rate'] if pretrained else 0.0,
        )

        if pretrained:
            ckpt = pvt_checkpoint
            if ckpt is None:
                ckpt = os.path.join('net', 'pvt_v2_b3.pth')
                try:
                    import net.config as _config
                    ckpt = os.path.join(_config.pvtv2_checkpoint_dir, 'pvt_v2_b3.pth')
                except Exception:
                    pass
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(
                    'PVTv2-B3 预训练权重不存在: {}\n'
                    '请放到 net/pvt_v2_b3.pth, 或用 pvt_checkpoint= 指定路径。'.format(ckpt))
            state = torch.load(ckpt, map_location='cpu')
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            missing, unexpected = model.load_state_dict(state, strict=False)
            print('Load {} succeed! (missing={}, unexpected={})'.format(
                ckpt, len(missing), len(unexpected)))
        return model, embed_dims

    @staticmethod
    def _align(src, ref):
        if src.shape[2:] != ref.shape[2:]:
            src = F.interpolate(src, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return src

    # --------------------------------------------------------
    def forward(self, x, depth, P_sam):  # depth 保留兼容旧接口, SHTNet 不使用
        if P_sam.dim() == 3:
            P_sam = P_sam.unsqueeze(1)
        out_size = x.shape[2:]

        # ================= Stage 1: Structural Evidence Extraction =================
        # 主干输出的原始多尺度特征 (BEM 的差异计算在原始特征上进行, 见 Eq.5-7)
        X1, X2, X3, X4 = self.pvt(x)                       # 64/168, 128/84, 320/42, 512/21 @672

        # BEM 边界证据 EI (Eq.5-7)
        E_I_logit, _, _ = self.bem(X1, X2, X3, X4)
        E_I = torch.sigmoid(E_I_logit)

        # 统一到 64/128/256/256 供 HGM 使用
        F1_in, F2_in = self.reduce1(X1), self.reduce2(X2)
        F3_in, F4_in = self.reduce3(X3), self.reduce4(X4)

        # 边界监督图: GT 边界 (Edge/) 或 SAM3 先验的形态学边界
        #
        # 【重要】返回的 edge_map 已经是 sigmoid 后的**概率** (下面两路都是 [0,1]):
        #           edge_src = E_I        -> torch.sigmoid(E_I_logit), 已激活
        #           edge_src = morph(...) -> 形态学梯度, 天然落在 [0,1]
        #         所以调用方算 dice 时**不要再套一次 sigmoid**, 否则会把输出压进
        #         [0.5, 0.731] 的窄区间, 使损失无法下降。与 priornet_v2 的约定一致:
        #         那边也是 "E = sigmoid(eam(x4, x1))" 之后直接喂给 dice_loss。
        if self.edge_supervision == 'sam':
            edge_src = self.hrh.morph(P_sam)
        else:
            edge_src = E_I
        edge_map = F.interpolate(edge_src, size=out_size, mode='bilinear', align_corners=False)

        # ================= Stage 2: Structural Hypothesis Testing =================
        R = self.hrh(P_sam, E_I)                            # Rejection Field (Eq.8-11)

        # ================= Stage 3: Rejection-Guided Structure Adaptation ==========
        # HGM: 每个尺度注入 Pn / EI, 并由 Rn 调制 (Eq.12-16)
        F1o = self.hgm1(F1_in, P_sam, E_I, R)
        F2o = self.hgm2(F2_in, P_sam, E_I, R)
        F3o = self.hgm3(F3_in, P_sam, E_I, R)
        F4o = self.hgm4(F4_in, P_sam, E_I, R)

        # MSM: 自深向浅逐级交互 (Eq.17-21), 深层的 F(n+1)o 上采样后与 Fno 聚合
        F4 = self.msm4(F4o, None)
        F3 = self.msm3(F3o, F4)
        F2 = self.msm2(F2o, F3)
        F1 = self.msm1(F1o, F2)

        # 像素级跨尺度组合
        F34 = self.comb3(torch.cat([self._align(F4, F3), F3], dim=1))
        F234 = self.comb2(torch.cat([self._align(F34, F2), F2], dim=1))
        F1234 = self.comb1(torch.cat([self._align(F234, F1), F1], dim=1))

        # ================= 三个预测头 =================
        # native_scale=True: 直接返回各头的原生预测尺度, 不做上采样。
        #   预测原生尺度 (672 输入): o3 -> 42², o2 -> 84², o1 -> 168², edge -> edge_size
        #   这样训练脚本可以把 GT 下采样到同一尺度再算损失, 做到 "pred 与 GT 同尺度"。
        #   否则(默认)图省事统一上采样到 out_size, 但那是"低分辨率预测 vs 高分辨率 GT",
        #   尺度并不对齐。
        if self.native_scale:
            o3 = self.head3(F34)                                 # 42²
            o2 = self.head2(F234)                                # 84²
            o1 = self.head1(F1234)                               # 168²
            return o3, o2, o1, edge_map

        o3 = self.head3(F34)
        o3 = F.interpolate(o3, size=out_size, mode='bilinear', align_corners=False)

        o2 = self.head2(F234)
        o2 = F.interpolate(o2, size=out_size, mode='bilinear', align_corners=False)

        o1 = self.head1(F1234)                               # 168²
        o1 = F.interpolate(o1, size=out_size, mode='bilinear', align_corners=False)

        # 返回值顺序与 priornet_v2 / priornet_v3_pvt 保持一致:
        # (o3, o2, o1, boundary)
        return o3, o2, o1, edge_map


# 兼容别名 (与其它 prior 文件一致的类名习惯)
Net = SHTNet


if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('=' * 74)
    print('SHTNet 自检 (随机初始化, 不加载预训练)')
    print('=' * 74)

    net = SHTNet(pretrained=False).to(dev).eval()
    n_all = sum(p.numel() for p in net.parameters())
    n_bb = sum(p.numel() for p in net.pvt.parameters())
    print('参数量: {:.2f} M (主干 {:.2f} M / 其余 {:.2f} M)'.format(
        n_all / 1e6, n_bb / 1e6, (n_all - n_bb) / 1e6))

    img = torch.randn(1, 3, 672, 672).to(dev)
    ps = (torch.rand(1, 1, 672, 672) > 0.7).float().to(dev)

    with torch.no_grad():
        outs = net(img, img, ps)

    names = ['o3 (1/16)', 'o2 (1/8)', 'o1 (1/4)', 'edge_map']
    # native_scale=True: 各预测头保持原生尺度 (672 输入 -> 42/84/168), edge_map 恒为输入尺度
    expect = [(1, 1, 42, 42), (1, 1, 84, 84), (1, 1, 168, 168), (1, 1, 672, 672)]
    ok = True
    for nm, o, ex in zip(names, outs, expect):
        good = tuple(o.shape) == ex
        ok &= good
        print('  {:<12} {:<20} 期望 {:<18} {}'.format(
            nm, str(tuple(o.shape)), str(ex), 'OK' if good else 'MISMATCH'))
    print('  (native_scale=True: 各预测头返回原生尺度, 训练脚本把 GT 下采样到同尺度)')
    # 顺手验证 Rejection Field 的数值范围
    with torch.no_grad():
        x1, x2, x3, x4 = net.pvt(img)
        ei_logit, _, _ = net.bem(x1, x2, x3, x4)
        r = net.hrh(ps, torch.sigmoid(ei_logit))
    print()
    print('  Rejection Field R: shape={} min={:.4f} max={:.4f} mean={:.4f}'.format(
        tuple(r.shape), r.min().item(), r.max().item(), r.mean().item()))
    print()
    print('自检通过' if ok else '自检失败')
