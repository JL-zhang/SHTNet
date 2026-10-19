"""
SHTNet: 训练 + 测试 + 评估 一键脚本
================================================================================
结构对齐 etrainV2.py (损失、LR 调度、保存节奏、auto_test_eval 全流程),
网络换成 prior/SHTNet.py。

论文 (sam_cod_2.pdf) 的实验设置:
    · 主干 PVTv2-B3, AdamW, 50 epochs, batch 8
    · 初始学习率 1e-4, weight decay 1e-4, 梯度裁剪 1.0, 随机种子 2021
    · L_total = L_str + lambda_e * L_dice        (Eq.24)
      其中 L_str = sum_{i=1..3} [L_wbce(Oi,Gt) + L_wiou(Oi,Gt)]   (Eq.22)
           L_dice 作用在 BEM 预测的边界证据 EI 与 GT 边界上          (Eq.23)

路径 (默认值取自 etrainV2.py, 并加本机自动探测):
    train : <train_path>/{Imgs,GT,Edge,Depth,sam3}/
    test  : <test_root>/<DATASET>/{Imgs,GT,Depth,sam3}/

用法
----
所有配置都写在下面的"配置区"里, 不需要任何命令行参数:

    nohup python train_eval_SHTNet.py > log/shtnet_bem_1.log 2>&1 &

只保留两个可选的命令行开关 (不传就用配置区的值):

    python train_eval_SHTNet.py --epoch 50 --batchsize 8

换实验/换数据/调超参 = 直接改配置区里的常量。
"""

import os
import sys
import csv
import shutil
import argparse
from datetime import datetime

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prior.SHTNet import SHTNet
from utils.tdataloader import get_loader, test_dataset
from utils.utils import clip_gradient, AvgMeter

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
torch.backends.cudnn.benchmark = True

# ============================================================================
# 配置区  ← 只改这里
#
#   除 EPOCH / BATCHSIZE 外, 本文件不再接受任何命令行参数, 所有变量都是模块级常量。
#   EPOCH / BATCHSIZE 也在这里给默认值, 想临时覆盖可以用 --epoch / --batchsize。
#
#   运行:  python train_eval_SHTNet.py
#          python train_eval_SHTNet.py --epoch 50 --batchsize 8
# ============================================================================

# ---- 数据路径 ----
TRAIN_ROOT = '/home/zhangjl/COD/train_moca/TrainDataset'
TEST_ROOT_CFG = '/home/zhangjl/COD/test'

# ---- SAM 先验来源 (两个阶段不同, 见 etrainV2.py 的说明) ----
#   ⚠ 训练先验**不是** <TRAIN_ROOT>/sam3 —— 那份只有约 1% 前景、且含大量全黑图,
#     所以 etrainV2.py 已弃用它, 改用别的模型在训练集上跑出的预测 (BGNet45)。
#     测试先验同理, 用预测图目录而非 <test_root>/<数据集>/sam3。
#   若你确实想用 sam3/, 把下面两个常量改成 os.path.join(TRAIN_ROOT, 'sam3') 一类即可。
TRAIN_SAM_ROOT = '/home/zhangjl/COD/TrainDataset/BGNet45'
PRED_ROOT = '/home/zhangjl/COD/spatialnet-v3/pred/sam_v2_1/BGNet-45'

# ---- 权重 / 输出 ----
CKPT_ROOT = '/home/zhangjl/COD/spatialnet-v3/checkpoints'
TRAIN_SAVE = 'shtnet_bem_1'          # 权重存到 <CKPT_ROOT>/<TRAIN_SAVE>/SHTNet-*.pth
# 预测图输出根目录; 刻意不跟 TRAIN_SAVE 联动, 避免覆盖正在读的 PRED_ROOT
SAVE_ROOT_CFG = '/home/zhangjl/COD/spatialnet-v3/pred/shtnet_bem_1'

# ---- 训练超参 (对齐 etrainV2.py / priornet_v2) ----
EPOCH = 60                           # 可用 --epoch 覆盖
BATCHSIZE = 6                        # 可用 --batchsize 覆盖
TRAINSIZE = 672
BACKBONE_LR = 1e-5
HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
CLIP = 1.0
W_O1 = 3.0                           # 结构损失里 o1 的权重 (etrainV2 的 loss1*3)
W_DICE = 5.0                         # 边界损失权重 (etrainV2 的 edge*5)
NUM_WORKERS = 4

# ---- 网络超参 (论文 sam_cod_2.pdf) ----
HRH_TAU = 0.05                       # Eq.11 的 tau; 论文说由验证集选定, 对 R 的动态范围影响很大
HRH_K = 3                            # 形态学结构元尺寸 k
HRH_P = 3                            # 局部 patch 尺寸 p
HRH_COMPARE_SIZE = 168               # 假设检验(HRH)的对比尺度; 168 让 ES/EI 同尺度
                                     # (EI 原生就在 168 上算)。0 或负数 = 跟随 P_sam 的尺度 (672)
GROUPS = 4                           # HGM 的通道分组数 G
EDGE_SUPERVISION = 'gt'              # 边界损失目标: 'gt'=Edge/ 目录; 'sam'=SAM 先验的形态学边界

# ---- 主干预训练权重 ----
# None = 走 net/config.py 的 pvtv2_checkpoint_dir 自动查找
PVT_CHECKPOINT = None

# ---- 验证集 ----
VAL_DATASETS = ['COD10K', 'CAMO']
VAL_EVERY = 1                        # 每多少个 epoch 验证一次; 0=关闭验证
VAL_START = 1                        # 从第几个 epoch 开始验证
VAL_METRIC = 'Smeasure'              # 'Smeasure' 越大越好 / 'MAE' 越小越好
VAL_KEEP_PRED = False                # False = 验证完立即删除预测图, 只留 eval 数值

# ---- 权重保存 / 最终测试 ----
SAVE_EVERY = 0                       # 0 = 只保留 best/last, 不存周期性快照
EVAL_START = 45                      # 配合 SAVE_EVERY: 评估 range(EVAL_START, EPOCH+1, SAVE_EVERY)
                                     # SAVE_EVERY=0 时改为评估 best/last
DATASETS = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']

# ---- 流程开关 ----
NO_TRAIN = False                     # True = 跳过训练, 直接拿已有权重 test+eval
NO_AUTO_EVAL = False                 # True = 只训练, 不自动 test+eval
# 要 test+eval 的权重; None = 自动推导 (见上面的 SAVE_EVERY / EVAL_START 说明)
#   可给数字 epoch, 也可给字符串 'best' / 'last'
WEIGHTS = None

LOG_FILE = None


def log(msg):
    print(msg, flush=True)
    if LOG_FILE is not None:
        LOG_FILE.write(msg + '\n')
        LOG_FILE.flush()


def sub_dir(root, name):
    """tdataloader 内部用字符串拼接读路径, 所以要补结尾分隔符"""
    return os.path.join(root, name) + os.sep


def downsample_to(gt, pred):
    """
    把 GT 下采样到预测 pred 的空间尺度 (nearest)。
    用于"预测与 GT 同尺度"的监督: 预测在低分辨率头上, 就把 GT 降下来, 而不是把预测升上去。
    用 nearest 是为了保持掩码的二值语义 —— 双线性会产生介于前景/背景之间的值。
    """
    if gt.shape[2:] == pred.shape[2:]:
        return gt
    return F.interpolate(gt, size=pred.shape[2:], mode='nearest')


def pick_existing(cands, label):
    """按顺序挑第一个真实存在的路径 (相对路径优先), 保证同一份代码跨机器可跑"""
    uniq = []
    for c in cands:
        c = (c or '').strip()
        if c and c not in uniq:
            uniq.append(c)
    for c in uniq:
        if os.path.isdir(c):
            log('  [路径] {} -> {}'.format(label, os.path.abspath(c)))
            return c
    log('  [警告] {} 候选路径都不存在: {}'.format(label, uniq))
    return uniq[0] if uniq else ''


def checkpoint_prefix(model):
    """按网络类型决定权重文件名前缀, 避免和 BGNet / BGNetPVT 的旧结果混淆"""
    return 'SHTNet'


# ============================================================
# 损失函数 (论文 Eq.22-24)
# ============================================================

def structure_loss(pred, mask):
    """L_wbce + L_wiou (Eq.22)"""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def dice_loss(predict, target, eps=1e-6):
    """L_dice (Eq.23): 1 - 2|P∩G| / (|P|+|G|)"""
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    num = torch.sum(predict * target, dim=1) * 2 + eps
    den = predict.pow(2).sum(dim=1) + target.pow(2).sum(dim=1) + eps
    return (1 - num / den).mean()


# ============================================================
# 差异化 Warmup + Poly LR (与 etrainV2.py 一致)
# ============================================================

def warmup_poly_differential(optimizer, init_lrs, curr_iter, max_iter):
    warmup_start = 1e-7
    warmup_steps = 1000
    if curr_iter <= warmup_steps:
        ratio = curr_iter / warmup_steps
        for i, g in enumerate(optimizer.param_groups):
            g['lr'] = warmup_start + (init_lrs[i] - warmup_start) * ratio
    else:
        ratio = (1 - (curr_iter - warmup_steps) / (max_iter - warmup_steps)) ** 0.9
        for i, g in enumerate(optimizer.param_groups):
            g['lr'] = init_lrs[i] * ratio


# ============================================================
# 训练
# ============================================================

def train(train_loader, model, optimizer, epoch, opt, init_lrs, total_step, prefix):
    from tqdm import tqdm

    model.train()

    r3, r2, r1, re = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

    # 每个 epoch 一个进度条, 实时显示 4 个损失和 lr
    pbar = tqdm(train_loader, total=total_step, ncols=110,
                desc='Epoch [{:03d}/{:03d}]'.format(epoch, opt.epoch))

    for i, pack in enumerate(pbar, start=1):
        curr_iter = (epoch - 1) * total_step + i
        max_iter = opt.epoch * total_step
        warmup_poly_differential(optimizer, init_lrs, curr_iter, max_iter)

        optimizer.zero_grad()

        images, gts, edges, depths, gt_sam = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        edges = Variable(edges).cuda()
        depths = Variable(depths).cuda()
        gt_sam = Variable(gt_sam).cuda()

        # SHTNet.forward(x, depth, P_sam) -> o3, o2, o1, edge_map
        # native_scale=True 时各预测保持原生尺度 (672 输入下: 42/84/168), 不做上采样
        o3, o2, o1, edge_map = model(images, depths, gt_sam)

        # Eq.22: 三个预测图的结构损失
        #
        # 【同尺度监督】预测各自在自己的原生尺度上 (42²/84²/168²), 所以把 GT 下采样到
        # 对应尺度再算, 而不是把预测上采样到 672²。
        #   下采样用 nearest: 分割掩码是 0/1, 用双线性会产生中间值(既非前景也非背景),
        #   对 wBCE 是噪声。nearest 保持二值语义。
        g3 = downsample_to(gts, o3)      # -> 42²
        g2 = downsample_to(gts, o2)      # -> 84²
        g1 = downsample_to(gts, o1)      # -> 168²
        loss3 = structure_loss(o3, g3)
        loss2 = structure_loss(o2, g2)
        loss1 = structure_loss(o1, g1)

        # Eq.23: 边界证据 vs GT 边界 —— 同样对齐到预测的尺度
        if opt.edge_supervision == 'sam':
            edge_gt = model.hrh.morph(gt_sam)
        else:
            edge_gt = edges
        if edge_gt.shape[2:] != edge_map.shape[2:]:
            # 边界标签前景极稀疏 (约 0.4%), 用 bilinear 下采样会产生亚像素值;
            # 这里同样用 nearest 保持二值, 避免"一半是前景"的假标签
            edge_gt = F.interpolate(edge_gt, size=edge_map.shape[2:], mode='nearest')
        # 【不要再套 sigmoid】SHTNet.forward 返回的 edge_map 已经是 sigmoid 后的概率
        # (E_I = sigmoid(BEM 输出))。再套一层会变成 sigmoid(sigmoid(x)), 把值压进
        # [0.5, 0.731] 的窄区间 —— 即使预测形状完美, dice 也下不去 0.965, 且梯度被压小。
        loss_e = dice_loss(edge_map, edge_gt)

        # Eq.24
        loss = loss3 + loss2 + loss1 * opt.w_o1 + loss_e * opt.w_dice

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        r3.update(loss3.data, opt.batchsize)
        r2.update(loss2.data, opt.batchsize)
        r1.update(loss1.data, opt.batchsize)
        re.update(loss_e.data, opt.batchsize)

        # 实时进度: 进度条 + 当前 lr
        cur_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix_str(
            'o3={:.4f} o2={:.4f} o1={:.4f} edge={:.4f} lr={:.2e}'.format(
                r3.avg, r2.avg, r1.avg, re.avg, cur_lr))

        # 按间隔写日志; 并保证小数据集 (total_step < 60) 每 epoch 至少留一行
        if i % 60 == 0 or i == total_step or (total_step < 60 and i == 1):
            log('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                '[o3: {:.4f}], [o2: {:.4f}], [o1: {:.4f}], [edge: {:.4f}]'.format(
                    datetime.now(), epoch, opt.epoch, i, total_step,
                    r3.avg, r2.avg, r1.avg, re.avg))
    pbar.close()

    # 周期性快照 (可选)。save_every<=0 时跳过 —— 不能写 epoch % opt.save_every, 会除零。
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    if opt.save_every and opt.save_every > 0 and epoch % opt.save_every == 0:
        ckpt = save_path + '{}-{}.pth'.format(prefix, epoch)
        torch.save(model.state_dict(), ckpt)
        log('[Saving Snapshot:] {}'.format(ckpt))

    return {'o3': r3.avg, 'o2': r2.avg, 'o1': r1.avg, 'edge': re.avg}


def save_ckpt(model, ckpt_dir, prefix, name):
    """统一的权重保存入口 (用于 -last / -best)"""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, '{}-{}.pth'.format(prefix, name))
    torch.save(model.state_dict(), path)
    log('[Saving Snapshot:] {}'.format(path))
    return path


# ============================================================
# 验证集机制 (与 prior/prior_SHTNet/train_test_eval.py 同一套)
# ============================================================

def validate_eval(model, test_root, datasets, opt, tmp_root):
    """
    在验证集上做一次 预测 + 评估。返回 (per_dataset_dict, mean_smeasure)

      · 预测图写到 tmp_root, 由调用方用完删除
      · 每个数据集都新建一个 test_dataset —— 它内部是"按下标自增"的状态机,
        复用同一实例在切换数据集时会越界
      · SAM 先验来自 opt.test_sam_path (默认 PRED_ROOT), **不是** <test_root>/<d>/sam3
    """
    import imageio
    import cv2
    from tqdm import tqdm
    from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

    was_training = model.training
    model.eval()
    results = {}

    for d in datasets:
        gt_root = sub_dir(sub_dir(test_root, d), 'GT')
        img_root = sub_dir(sub_dir(test_root, d), 'Imgs')
        depth_root = sub_dir(sub_dir(test_root, d), 'Depth')
        # 先验来自独立目录: <test_sam_path>/<数据集>/*.png
        sam_root = sub_dir(opt.test_sam_path, d)

        if not os.path.isdir(gt_root) or not os.path.isdir(sam_root):
            log('    [Val][SKIP] {} 缺少 GT 或 先验目录: {}'.format(d, sam_root))
            continue

        pred_dir = os.path.join(tmp_root, d)
        os.makedirs(pred_dir, exist_ok=True)

        loader = test_dataset(img_root, gt_root, depth_root, sam_root, opt.trainsize)
        for _ in tqdm(range(loader.size), desc='    [Val] {} predict'.format(d),
                      leave=False, ncols=88):
            image, gt, depth, gt_sam, name = loader.load_data()
            name = os.path.basename(name)
            gt_arr = np.asarray(gt, np.float32)
            gt_arr /= (gt_arr.max() + 1e-8)

            with torch.no_grad():
                _, _, pred, _ = model(image.cuda(), depth.cuda(), gt_sam.cuda())

            res = F.interpolate(pred, size=gt_arr.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            imageio.imwrite(os.path.join(pred_dir, name), (res * 255).astype(np.uint8))

        # ---- 评估这一个数据集 ----
        mask_list = sorted([f for f in os.listdir(gt_root) if f.endswith('.png')])
        FM, WFM, SM, EM, M = Fmeasure(), WeightedFmeasure(), Smeasure(), Emeasure(), MAE()
        n_used = 0
        for mask_name in tqdm(mask_list, desc='    [Val] {} eval   '.format(d),
                              leave=False, ncols=88):
            pp = os.path.join(pred_dir, mask_name)
            if not os.path.exists(pp):
                continue
            mask = cv2.imread(os.path.join(gt_root, mask_name), cv2.IMREAD_GRAYSCALE)
            pr = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
            if mask is None or pr is None:
                continue
            FM.step(pred=pr, gt=mask)
            WFM.step(pred=pr, gt=mask)
            SM.step(pred=pr, gt=mask)
            EM.step(pred=pr, gt=mask)
            M.step(pred=pr, gt=mask)
            n_used += 1

        if n_used == 0:
            continue

        results[d] = {
            'n': n_used,
            'Smeasure': round(float(SM.get_results()['sm']), 4),
            'meanEm': round(float(EM.get_results()['em']['curve'].mean()), 4),
            'wFmeasure': round(float(WFM.get_results()['wfm']), 4),
            'meanFm': round(float(FM.get_results()['fm']['curve'].mean()), 4),
            'MAE': round(float(M.get_results()['mae']), 4),
        }

    if was_training:
        model.train()

    mean_sm = float(np.mean([r['Smeasure'] for r in results.values()])) if results else -1.0
    return results, mean_sm


def run_validation(model, opt, epoch, tmp_root, hist_path):
    """跑一次验证并把明细追加到历史 csv。返回 val_Smeasure (无结果时 -1)"""
    log('')
    log('-' * 70)
    log('[Val] Epoch {:03d} 在 {} 上验证'.format(epoch, list(opt.val_datasets)))
    log('-' * 70)

    res, mean_sm = validate_eval(model, opt.test_root_resolved, opt.val_datasets, opt, tmp_root)

    for d, r in res.items():
        log('    {}: Smeasure={:.4f} meanEm={:.4f} wFmeasure={:.4f} meanFm={:.4f} MAE={:.4f} (n={})'.format(
            d, r['Smeasure'], r['meanEm'], r['wFmeasure'], r['meanFm'], r['MAE'], r['n']))
    if res:
        log('    >> val_Smeasure = {:.4f}'.format(mean_sm))
    else:
        log('    >> 没有可用的验证结果 (检查验证数据集路径)')

    # 追加到历史 csv (不覆盖, 便于事后画曲线)
    write_header = not os.path.exists(hist_path)
    with open(hist_path, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['epoch', 'dataset', 'n', 'Smeasure', 'meanEm',
                        'wFmeasure', 'meanFm', 'MAE'])
        for d, r in res.items():
            w.writerow([epoch, d, r['n'], r['Smeasure'], r['meanEm'],
                        r['wFmeasure'], r['meanFm'], r['MAE']])
        if res:
            w.writerow([epoch, 'MEAN', '', round(mean_sm, 4), '', '', '', ''])

    # 用完即删: 只保留 eval 数值, 不留预测图
    if opt.val_keep_pred:
        log('    [Val] 预测图保留在 {}'.format(tmp_root))
    else:
        shutil.rmtree(tmp_root, ignore_errors=True)
        log('    [Val] 预测图已删除; 结果已记录到 {}'.format(hist_path))

    return mean_sm


# ============================================================
# 自动测试 + 评估
# ============================================================

def auto_test_eval(ckpt_dir, test_root, datasets, weights, save_root, opt, prefix):
    log('')
    log('=' * 70)
    log('Starting Auto Test & Evaluation...')
    log('=' * 70)

    from tqdm import tqdm
    import imageio
    import cv2
    try:
        from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
    except ImportError as e:
        log('ERROR: 无法导入 py_sod_metrics ({}), 请 pip install pysodmetrics'.format(e))
        return {}

    os.makedirs(save_root, exist_ok=True)
    log_file = os.path.join(save_root, 'eval_results.txt')
    per_weight = {}

    with open(log_file, 'w', encoding='utf-8') as f_log:
        f_log.write('SHTNet evaluation\n')
        f_log.write('generated: {}\n'.format(datetime.now()))
        f_log.write('ckpt_dir: {}\n'.format(os.path.abspath(ckpt_dir)))
        f_log.write('=' * 70 + '\n\n')

        for wid in weights:
            ckpt_path = os.path.join(ckpt_dir, '{}-{}.pth'.format(prefix, wid))
            if not os.path.exists(ckpt_path):
                log('[SKIP] 权重不存在: {}'.format(ckpt_path))
                continue

            log('')
            log('--- Testing: {}-{} ---'.format(prefix, wid))

            # pretrained=False: 马上被 ckpt 覆盖, 没必要再读一次 PVT 预训练文件
            model = SHTNet(pretrained=False,
                           edge_supervision=opt.edge_supervision,
                           hrh_tau=opt.hrh_tau,
                           hrh_k=opt.hrh_k,
                           hrh_p=opt.hrh_p,
                           hrh_compare_size=(opt.hrh_compare_size if opt.hrh_compare_size > 0 else None),
                           groups=opt.groups)
            state = torch.load(ckpt_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                log('  [warn] missing={} unexpected={}'.format(len(missing), len(unexpected)))
            model = model.cuda().eval()

            for d in datasets:
                gt_root = sub_dir(sub_dir(test_root, d), 'GT')
                img_root = sub_dir(sub_dir(test_root, d), 'Imgs')
                depth_root = sub_dir(sub_dir(test_root, d), 'Depth')
                # 先验来自独立目录: <test_sam_path>/<数据集>/*.png
                sam_root = sub_dir(opt.test_sam_path, d)
                pred_save = os.path.join(save_root, '{}-{}'.format(prefix, wid), d)

                if not os.path.isdir(gt_root):
                    log('  [SKIP] GT 目录不存在: {}'.format(gt_root))
                    continue
                if not os.path.isdir(sam_root):
                    log('  [SKIP] 先验目录不存在: {}'.format(sam_root))
                    continue

                os.makedirs(pred_save, exist_ok=True)
                loader = test_dataset(img_root, gt_root, depth_root, sam_root, opt.trainsize)
                log('  {}: {} images'.format(d, loader.size))

                for _ in range(loader.size):
                    image, gt, depth, gt_sam, name = loader.load_data()
                    name = os.path.basename(name)   # 不能用 name.split('/'), Windows 路径会出错
                    gt_arr = np.asarray(gt, np.float32)
                    gt_arr /= (gt_arr.max() + 1e-8)

                    with torch.no_grad():
                        _, _, pred, _ = model(image.cuda(), depth.cuda(), gt_sam.cuda())

                    res = F.interpolate(pred, size=gt_arr.shape, mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    imageio.imwrite(os.path.join(pred_save, name), (res * 255).astype(np.uint8))

            del model
            torch.cuda.empty_cache()

            # ---- 评估这个权重 ----
            log('--- Evaluating: {}-{} ---'.format(prefix, wid))
            sms, maes = [], []

            for d in datasets:
                pred_root = os.path.join(save_root, '{}-{}'.format(prefix, wid), d)
                mask_root = sub_dir(sub_dir(test_root, d), 'GT')
                if not os.path.isdir(pred_root) or not os.path.isdir(mask_root):
                    continue

                mask_list = sorted([f for f in os.listdir(mask_root) if f.endswith('.png')])
                FM, WFM, SM, EM, M = Fmeasure(), WeightedFmeasure(), Smeasure(), Emeasure(), MAE()
                n_used = 0

                for mask_name in tqdm(mask_list, total=len(mask_list),
                                      desc='  {}'.format(d), leave=False):
                    mp = os.path.join(mask_root, mask_name)
                    pp = os.path.join(pred_root, mask_name)
                    if not os.path.exists(pp):
                        continue
                    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    pred = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
                    if mask is None or pred is None:
                        continue
                    FM.step(pred=pred, gt=mask)
                    WFM.step(pred=pred, gt=mask)
                    SM.step(pred=pred, gt=mask)
                    EM.step(pred=pred, gt=mask)
                    M.step(pred=pred, gt=mask)
                    n_used += 1

                if n_used == 0:
                    continue

                row = {
                    'weight': '{}-{}'.format(prefix, wid),
                    'dataset': d,
                    'n': n_used,
                    'Smeasure': round(float(SM.get_results()['sm']), 4),
                    'meanEm': round(float(EM.get_results()['em']['curve'].mean()), 4),
                    'wFmeasure': round(float(WFM.get_results()['wfm']), 4),
                    'meanFm': round(float(FM.get_results()['fm']['curve'].mean()), 4),
                    'MAE': round(float(M.get_results()['mae']), 4),
                }
                sms.append(row['Smeasure'])
                maes.append(row['MAE'])
                log('    {}'.format(row))
                f_log.write('{}\n'.format(row))

            if sms:
                per_weight[wid] = (float(np.mean(sms)), float(np.mean(maes)))
                log('  >> {}-{} 平均: Smeasure={:.4f}  MAE={:.4f}'.format(
                    prefix, wid, per_weight[wid][0], per_weight[wid][1]))

        if per_weight:
            best = max(per_weight.items(), key=lambda kv: kv[1][0])[0]
            log('')
            log('=' * 70)
            log('SUMMARY (按平均 Smeasure 排序)')
            log('=' * 70)
            for wid, (s, m) in sorted(per_weight.items(), key=lambda kv: -kv[1][0]):
                # wid 可能是 int (epoch 号) 或 str ('best' / 'last'), 所以用 :<8 而不是 :<4d
                log('  {}-{:<8} mean_Smeasure={:.4f}  mean_MAE={:.4f}{}'.format(
                    prefix, str(wid), s, m, '   <== BEST' if wid == best else ''))
            f_log.write('\nSUMMARY\n')
            for wid, (s, m) in sorted(per_weight.items(), key=lambda kv: -kv[1][0]):
                f_log.write('{}-{}\tmean_Smeasure={:.4f}\tmean_MAE={:.4f}\n'.format(prefix, wid, s, m))
            f_log.write('\nBEST: {}-{}\n'.format(prefix, best))
            log('')
            log('最优权重: {}-{}'.format(prefix, best))

    log('结果日志: {}'.format(os.path.abspath(log_file)))
    return per_weight


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    # 只暴露 epoch / batchsize 两个开关; 其余全部来自文件顶部的配置区。
    parser = argparse.ArgumentParser(
        description='SHTNet: train -> test -> eval (配置见文件顶部配置区)')
    parser.add_argument('--epoch', type=int, default=EPOCH,
                        help='epoch number (默认 {} = 配置区的 EPOCH)'.format(EPOCH))
    parser.add_argument('--batchsize', type=int, default=BATCHSIZE,
                        help='batch size (默认 {} = 配置区的 BATCHSIZE)'.format(BATCHSIZE))
    opt = parser.parse_args()

    # 把配置区的常量挂到 opt 上, 这样函数签名 (train/validate_eval/auto_test_eval)
    # 不用改, 内部统一读 opt.xxx。
    opt.trainsize = TRAINSIZE
    opt.backbone_lr = BACKBONE_LR
    opt.head_lr = HEAD_LR
    opt.weight_decay = WEIGHT_DECAY
    opt.clip = CLIP
    opt.w_o1 = W_O1
    opt.w_dice = W_DICE
    opt.save_every = SAVE_EVERY
    opt.num_workers = NUM_WORKERS
    # 网络
    opt.hrh_tau = HRH_TAU
    opt.hrh_k = HRH_K
    opt.hrh_p = HRH_P
    opt.hrh_compare_size = HRH_COMPARE_SIZE
    opt.groups = GROUPS
    opt.edge_supervision = EDGE_SUPERVISION
    opt.pvt_checkpoint = PVT_CHECKPOINT
    # 路径
    opt.train_path = TRAIN_ROOT
    opt.train_sam_path = TRAIN_SAM_ROOT
    opt.test_root = TEST_ROOT_CFG
    opt.test_sam_path = PRED_ROOT
    opt.datasets = list(DATASETS)
    opt.train_save = TRAIN_SAVE
    opt.ckpt_root = CKPT_ROOT
    opt.pred_root = SAVE_ROOT_CFG
    # 验证集
    opt.val_datasets = list(VAL_DATASETS)
    opt.val_every = VAL_EVERY
    opt.val_start = VAL_START
    opt.val_keep_pred = VAL_KEEP_PRED
    opt.val_metric = VAL_METRIC
    # 测试/评估
    opt.eval_start = EVAL_START
    opt.weights = WEIGHTS
    opt.no_train = NO_TRAIN
    opt.no_auto_eval = NO_AUTO_EVAL

    # 合法性自检 (写错了立刻报错, 别等跑起来才发现)
    assert opt.edge_supervision in ('gt', 'sam'), \
        "EDGE_SUPERVISION 只支持 'gt' 或 'sam'"
    assert opt.val_metric in ('Smeasure', 'MAE'), \
        "VAL_METRIC 只支持 'Smeasure' 或 'MAE'"
    assert opt.epoch >= 1 and opt.batchsize >= 1, 'epoch/batchsize 必须 >= 1'

    # WEIGHTS 可以给数字 epoch, 也可以给 'best' / 'last' (验证机制保存的两个权重)
    if opt.weights:
        EVAL_WEIGHTS = [w if isinstance(w, str) else int(w) for w in opt.weights]
    elif opt.save_every and opt.save_every > 0:
        EVAL_WEIGHTS = list(range(opt.eval_start, opt.epoch + 1, opt.save_every)) or ['best', 'last']
    else:
        # 没有周期性快照时, 默认评估验证机制保存的 best 和 last
        EVAL_WEIGHTS = ['best', 'last']

    if not os.path.isdir('log'):
        os.makedirs('log')
    LOG_FILE = open('log/train_eval_shtnet.txt', 'a', encoding='utf-8')

    # 权重 / 输出目录 (来自配置区, 命令行可覆盖)
    CKPT_DIR = os.path.join(opt.ckpt_root, opt.train_save) + os.sep
    SAVE_ROOT = opt.pred_root if opt.pred_root else os.path.join('pred', opt.train_save)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # 验证集相关路径
    val_tmp_root = os.path.join(SAVE_ROOT, '_val_tmp')
    val_hist = os.path.join('log', 'val_history_{}.csv'.format(opt.train_save))

    # ---- 训练/测试根目录 ----
    TRAIN_ROOT = pick_existing([opt.train_path, './TrainDataset', '../TrainDataset'],
                               'train_path').rstrip('/\\')
    TEST_ROOT = pick_existing([opt.test_root, '../test', './test'],
                              'test_root').rstrip('/\\')
    # ---- SAM 先验目录 (两个阶段来源不同) ----
    TRAIN_SAM = pick_existing([opt.train_sam_path], 'train_sam_path').rstrip('/\\')
    TEST_SAM = pick_existing([opt.test_sam_path], 'test_sam_path').rstrip('/\\')

    log('')
    log('#' * 70)
    log('SHTNet (SAM3-based Hypothesis Testing Network) : train -> test -> eval')
    log('  start      : {}'.format(datetime.now()))
    log('  train_path : {}'.format(TRAIN_ROOT))
    log('  train_sam  : {}'.format(TRAIN_SAM))
    log('  test_root  : {}'.format(TEST_ROOT))
    log('  test_sam   : {}'.format(TEST_SAM))
    log('  train_save : {}'.format(opt.train_save))
    log('  torch {} | cuda {}'.format(torch.__version__, torch.cuda.is_available()))
    log('#' * 70)

    # ---- 路径检查 ----
    # 训练集: Imgs/GT/Edge/Depth 在 TRAIN_ROOT 下; 先验在 TRAIN_SAM (独立目录)
    train_ok = all(os.path.isdir(sub_dir(TRAIN_ROOT, s))
                   for s in ['Imgs', 'GT', 'Edge', 'Depth'])
    test_ok = any(os.path.isdir(sub_dir(sub_dir(TEST_ROOT, d), 'GT')) for d in opt.datasets)
    # run_validation 用的是这个字段
    opt.test_root_resolved = TEST_ROOT

    for s in ['Imgs', 'GT', 'Edge', 'Depth']:
        log('  [{}] {}'.format('OK' if os.path.isdir(sub_dir(TRAIN_ROOT, s)) else '--',
                               sub_dir(TRAIN_ROOT, s)))
    log('  [{}] {}   <- 训练 SAM 先验 (独立目录)'.format(
        'OK' if os.path.isdir(TRAIN_SAM) else '--', TRAIN_SAM))
    for d in opt.datasets:
        gt_ok = os.path.isdir(sub_dir(sub_dir(TEST_ROOT, d), 'GT'))
        sam_ok = os.path.isdir(sub_dir(TEST_SAM, d))
        log('  [{}{}] {} + {}'.format('OK' if gt_ok else '--', 'OK' if sam_ok else '--',
                                      sub_dir(sub_dir(TEST_ROOT, d), 'GT'),
                                      sub_dir(TEST_SAM, d)))

    if not opt.no_train and not train_ok:
        log('ERROR: 训练集目录不完整, 请用 --train_path 指定。')
        LOG_FILE.close()
        sys.exit(1)
    if not opt.no_train and not os.path.isdir(TRAIN_SAM):
        log('ERROR: 训练 SAM 先验目录不存在: {}'.format(TRAIN_SAM))
        log('       请用 --train_sam_path 指定 (默认值在文件顶部的配置区)。')
        LOG_FILE.close()
        sys.exit(1)
    if not torch.cuda.is_available():
        log('ERROR: 没有可用 GPU。')
        LOG_FILE.close()
        sys.exit(1)

    # ---- build model ----
    model = SHTNet(pretrained=True,
                   pvt_checkpoint=opt.pvt_checkpoint,
                   hrh_k=opt.hrh_k, hrh_p=opt.hrh_p, hrh_tau=opt.hrh_tau,
                   hrh_compare_size=(opt.hrh_compare_size if opt.hrh_compare_size > 0 else None),
                   groups=opt.groups,
                   edge_supervision=opt.edge_supervision).cuda()

    PREFIX = checkpoint_prefix(model)

    n_all = sum(p.numel() for p in model.parameters())
    n_bb = sum(p.numel() for p in model.pvt.parameters())

    # ---- 差异化参数分组 ----
    backbone_params = list(model.pvt.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith('pvt')]
    assert len(backbone_params) + len(head_params) == len(list(model.parameters())), \
        '参数分组不完整, 请检查主干属性名是否为 model.pvt'
    optimizer = torch.optim.AdamW(
        [{'params': backbone_params, 'lr': opt.backbone_lr},
         {'params': head_params, 'lr': opt.head_lr}],
        weight_decay=opt.weight_decay)
    init_lrs = [opt.backbone_lr, opt.head_lr]

    # ---- data loader ----
    image_root = sub_dir(TRAIN_ROOT, 'Imgs')
    gt_root = sub_dir(TRAIN_ROOT, 'GT')
    edge_root = sub_dir(TRAIN_ROOT, 'Edge')
    depth_root = sub_dir(TRAIN_ROOT, 'Depth')
    # 训练先验来自 TRAIN_SAM (独立目录), 不再是 <TRAIN_ROOT>/sam3
    sam_root = sub_dir(TRAIN_SAM, '')

    train_loader = get_loader(image_root, gt_root, edge_root, depth_root, sam_root,
                              batchsize=opt.batchsize, trainsize=opt.trainsize,
                              num_workers=opt.num_workers)
    total_step = len(train_loader)

    log('')
    log('Start Training (SHTNet - PVTv2-B3)')
    log('  Model: {:.2f} M params (backbone {:.2f} M / rest {:.2f} M)'.format(
        n_all / 1e6, n_bb / 1e6, (n_all - n_bb) / 1e6))
    log('  Epochs: {}, Batch: {}, Size: {}'.format(opt.epoch, opt.batchsize, opt.trainsize))
    log('  LR: backbone {:.1e} / head {:.1e} | AdamW wd {:.1e}'.format(
        opt.backbone_lr, opt.head_lr, opt.weight_decay))
    log('  LR Schedule: Warmup(1000) + Poly(0.9)')
    log('  Loss: L_str(o3+o2+o1*{}) + {:.1f}*L_dice   [Eq.22-24]'.format(opt.w_o1, opt.w_dice))
    log('  Clip: {} | HRH: k={} p={} tau={} | HGM groups={}'.format(
        opt.clip, opt.hrh_k, opt.hrh_p, opt.hrh_tau, opt.groups))
    log('  HRH: k={} p={} tau={} 对比尺度={}'.format(
        opt.hrh_k, opt.hrh_p, opt.hrh_tau,
        opt.hrh_compare_size if opt.hrh_compare_size > 0 else 'follow P_sam(672)'))
    log('  edge_supervision: {}'.format(opt.edge_supervision))
    log('  Steps/epoch: {}'.format(total_step))
    log('  EVAL_WEIGHTS: {}'.format(EVAL_WEIGHTS))
    if any(isinstance(w, str) for w in EVAL_WEIGHTS):
        log('    (best/last 由验证机制保存; 若某次训练没有产生验证结果, 就只会有 last)')
    if opt.val_every > 0:
        log('  Validation: 每 {} 个 epoch (从 epoch {} 起) 在 {} 上验证, 指标={}'.format(
            opt.val_every, opt.val_start, list(opt.val_datasets), opt.val_metric))
        log('              验证完自动删除预测图, 数值记入 {}'.format(val_hist))
    else:
        log('  Validation: 已关闭 (--val_every 0)')

    # ---- 训练 (含验证 / 最优权重跟踪) ----
    if not opt.no_train:
        if total_step == 0:
            log('ERROR: 训练集为空, 请检查 {} 下是否有 .jpg'.format(image_root))
            LOG_FILE.close()
            sys.exit(1)

        if opt.val_every > 0:
            # 训练阶段要验证, 但验证集缺失 -> 直接退出, 避免白跑几十小时
            missing = [d for d in opt.val_datasets
                       if not os.path.isdir(sub_dir(sub_dir(TEST_ROOT, d), 'GT'))]
            if missing:
                log('ERROR: --val_every={} 要求验证, 但 test_root 下缺少这些数据集: {}'.format(
                    opt.val_every, missing))
                log('       实际 test_root = {}'.format(TEST_ROOT))
                log('       请修正路径, 或用 --val_every 0 关闭验证。')
                LOG_FILE.close()
                sys.exit(1)

        best_val = None
        best_epoch = -1
        last_val = -1.0

        for epoch in range(1, opt.epoch + 1):
            stat = train(train_loader, model, optimizer, epoch, opt, init_lrs,
                         total_step, PREFIX)

            # ---- 每 val_every 个 epoch 验证一次 ----
            if opt.val_every > 0 and epoch >= opt.val_start and epoch % opt.val_every == 0:
                val_sm = run_validation(model, opt, epoch, val_tmp_root, val_hist)
                last_val = val_sm

                if val_sm >= 0:
                    # MAE 越小越好, 其余越大越好
                    if opt.val_metric == 'MAE':
                        better = (best_val is None) or (val_sm < best_val)
                    else:
                        better = (best_val is None) or (val_sm > best_val)

                    if better:
                        best_val = val_sm
                        best_epoch = epoch
                        save_ckpt(model, CKPT_DIR, PREFIX, 'best')
                        log('    [Val] 新的最优 ({}={:.4f}) -> 已保存 {}-best.pth'.format(
                            opt.val_metric, val_sm, PREFIX))
                    else:
                        log('    [Val] 未超过历史最优 {:.4f} (epoch {})'.format(
                            best_val, best_epoch))

        # ---- 训练结束: 保存最后一次 ----
        save_ckpt(model, CKPT_DIR, PREFIX, 'last')

        log('')
        log('Training done!')
        if opt.val_every > 0:
            if best_epoch > 0:
                log('  最优验证权重的 epoch: {} ({}={:.4f})'.format(
                    best_epoch, opt.val_metric, best_val))
                log('  权重: {}-best.pth  /  {}-last.pth'.format(PREFIX, PREFIX))
            else:
                log('  WARN: 整个训练过程没有产生有效验证结果, 只保存了 -last')
    else:
        log('---- 已跳过训练 (--no_train) ----')

    # ---- 测试 + 评估 ----
    if not opt.no_auto_eval:
        if not test_ok:
            log('WARN: 测试集目录不存在, 测试阶段会全部跳过。')
        auto_test_eval(CKPT_DIR, TEST_ROOT, opt.datasets, EVAL_WEIGHTS, SAVE_ROOT, opt, PREFIX)
    else:
        log('---- 已跳过自动 test+eval (--no_auto_eval) ----')

    log('')
    log('#' * 70)
    log('# ALL DONE : {}'.format(datetime.now()))
    log('# ckpt dir : {}'.format(os.path.abspath(CKPT_DIR)))
    log('# pred dir : {}'.format(os.path.abspath(SAVE_ROOT)))
    log('#' * 70)

    LOG_FILE.close()
