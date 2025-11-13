import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
from data import ImageDetectionsField, TextField, RawField
from data import COCO, DataLoader
import evaluation
from evaluation import PTBTokenizer, Cider

from models.fsgr import TransformerEncoder, TransformerDecoderLayer, ScaledDotProductAttention
from models.fsgr.transformer import Transformer
from models.fsgr.optim_entry import build_optimizer, SupConLoss

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import NLLLoss
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
import pickle
import numpy as np
import itertools
import multiprocessing
from shutil import copyfile
import torch.multiprocessing as mp
import torch.distributed as dist
import warnings

warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ====== 快速评估/限步控制（通过环境变量）======
FAST_EVAL   = os.getenv("FSGR_FAST_EVAL", "0") == "1"     # 评估阶段限步（默认关）
EVAL_STEPS  = int(os.getenv("FSGR_EVAL_STEPS", "0"))      # 评估最多步数，0 表示不限制
VAL_STEPS   = int(os.getenv("FSGR_VAL_STEPS", "0"))       # 验证损失最多步数
TRAIN_STEPS = int(os.getenv("FSGR_TRAIN_STEPS", "0"))     # 训练阶段最多步数
PRINT_EVERY = 50


def _parse_trainval_batch(batch, device):
    """
    统一不同数据管道的 batch 形态，返回 (detections/images, labels(or None), captions)

    兼容以下几种：
      1) (detections, labels, captions)                      # 原训练/验证三元组
      2) (image_ids, images, placeholder, captions)          # 4 元，直接读图像
      3) ((images, _), caps_gt)                              # 2 元，dict_dataloader 风格（无 labels）
      4) 其它二元组合 (images, captions)
    """
    import torch

    # 1) 三元组
    if isinstance(batch, (list, tuple)) and len(batch) == 3:
        detections, labels, captions = batch
        return detections.to(device), labels.to(device), captions.to(device)

    # 2) 四元组（无 labels）
    if isinstance(batch, (list, tuple)) and len(batch) == 4:
        _, images, _, captions = batch
        return images.to(device), None, captions.to(device)

    # 3) 二元组：((images, _), caps) 或 (images, caps)
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        img_pack, caps = batch
        if isinstance(img_pack, (list, tuple)) and len(img_pack) >= 1:
            images = img_pack[0]
        else:
            images = img_pack
        # caps 可能是 list[str]（评估）或 tensor（训练/验证）
        if torch.is_tensor(caps):
            caps_t = caps.to(device)
        else:
            caps_t = caps
        return images.to(device), None, caps_t

    raise ValueError(f"[parse] 不支持的batch结构: type={type(batch)}, len={len(batch) if hasattr(batch,'__len__') else 'NA'}")


def evaluate_loss(model, dataloader, loss_fn, text_field, beta=0.25):
    """验证集交叉熵损失（可限步）"""
    model.eval()
    running_loss = 0.0
    total = min(len(dataloader), VAL_STEPS) if VAL_STEPS > 0 else len(dataloader)

    with tqdm(desc='Epoch %d - validation' % e, unit='it', total=total) as pbar:
        with torch.no_grad():
            for it, batch in enumerate(dataloader):
                detections, labels, captions = _parse_trainval_batch(batch, device)

                out = model(detections, captions)
                ca_loss = 0.0
                if isinstance(out, tuple):
                    out, *supcon = out
                    if labels is not None:
                        # SupCon 分支在验证阶段通常没有 labels，这里仅在存在 labels 时计算
                        if hasattr(loss_contrast, 'forward_similarity'):
                            ca_loss = loss_contrast.forward_similarity(supcon[0], labels)
                        else:
                            ca_loss = loss_contrast(supcon[0], labels)

                captions_gt = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                ce_loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                loss = ce_loss + beta * ca_loss

                running_loss += float(loss.item())

                if (it + 1) % PRINT_EVERY == 0 or (it + 1) == total:
                    pbar.set_postfix(loss=round(running_loss / (it + 1), 4))
                pbar.update(1)

                if VAL_STEPS > 0 and (it + 1) >= VAL_STEPS:
                    break

    denom = max(1, (VAL_STEPS if VAL_STEPS > 0 else len(dataloader)))
    return running_loss / denom


def evaluate_metrics(model, dataloader, text_field):
    """CIDEr/ROUGE/BLEU 等指标（可限步；dict_dataloader 风格）"""
    model.eval()
    gen, gts = {}, {}

    use_cap = FAST_EVAL and EVAL_STEPS > 0
    total_steps = min(len(dataloader), EVAL_STEPS) if use_cap else len(dataloader)

    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=total_steps) as pbar:
        for it, ((images, _), caps_gt) in enumerate(iter(dataloader)):
            images = images.to(device)
            with torch.no_grad():
                out, _ = model.beam_search(
                    images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1
                )

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                # 去除重复空格
                gen_i = ' '.join([k for k, _ in itertools.groupby(gen_i)])
                gen[f'{it}_{i}'] = [gen_i]
                gts[f'{it}_{i}'] = gts_i

            pbar.update(1)
            if use_cap and (it + 1) >= EVAL_STEPS:
                break

    gts_tok = evaluation.PTBTokenizer.tokenize(gts)
    gen_tok = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts_tok, gen_tok)
    return scores


def train_xe(model, dataloader, optim, text_field, beta=0.25):
    """交叉熵训练（可限步）"""
    model.train()

    if len(optim.param_groups) > 1:
        print('Backbone lr = ', optim.param_groups[0]['lr'])
        print('Dec lr = ', optim.param_groups[1]['lr'])
    else:
        print('lr = ', optim.param_groups[0]['lr'])

    running_loss = 0.0
    total = min(len(dataloader), TRAIN_STEPS) if TRAIN_STEPS > 0 else len(dataloader)

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=total) as pbar:
        for it, batch in enumerate(dataloader):
            detections, labels, captions = _parse_trainval_batch(batch, device)

            with torch.cuda.amp.autocast():
                out = model(detections, captions)
                ca_loss = 0.0
                if isinstance(out, tuple):
                    out, *supcon = out
                    if labels is not None:
                        if hasattr(loss_contrast, 'forward_similarity'):
                            ca_loss = loss_contrast.forward_similarity(supcon[0], labels)
                        else:
                            ca_loss = loss_contrast(supcon[0], labels)

                captions_gt = captions[:, 1:].contiguous()
                out = out[:, :-1].contiguous()
                ce_loss = loss_fn(out.view(-1, len(text_field.vocab)), captions_gt.view(-1))
                loss = ce_loss + beta * ca_loss

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += float(loss.item())
            if (it + 1) % PRINT_EVERY == 0 or (it + 1) == total:
                pbar.set_postfix(loss=round(running_loss / (it + 1), 4))
            pbar.update(1)

            if TRAIN_STEPS > 0 and (it + 1) >= TRAIN_STEPS:
                break

    scheduler.step()
    return running_loss / max(1, total)


def train_scst(model, dataloader, optim, cider, text_field):
    """SCST 训练（需要 Java 环境用于 CIDEr，且 dict_dataloader 风格）"""
    tokenizer_pool = multiprocessing.Pool()
    running_reward = 0.0
    running_reward_baseline = 0.0

    model.train()
    print('RL lr = ', optim_rl.state_dict()['param_groups'][0]['lr'])
    running_loss = 0.0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((detections, _), caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            with torch.cuda.amp.autocast():
                outs, log_probs = model.beam_search(
                    detections, seq_len, text_field.vocab.stoi['<eos>'],
                    beam_size, out_size=beam_size
                )
                # Rewards
                caps_gen = text_field.decode(outs.view(-1, seq_len))
                caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
                caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
                reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
                reward_baseline = torch.mean(reward, -1, keepdim=True)
                loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)
                loss = loss.mean()

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += float(loss.item())
            running_reward += float(reward.mean().item())
            running_reward_baseline += float(reward_baseline.mean().item())
            pbar.set_postfix(
                loss=round(running_loss / (it + 1), 4),
                reward=round(running_reward / (it + 1), 4),
                reward_baseline=round(running_reward_baseline / (it + 1), 4)
            )
            pbar.update(1)

    scheduler_rl.step()
    tokenizer_pool.close()
    tokenizer_pool.join()

    loss = running_loss / len(dataloader)
    reward = running_reward / len(dataloader)
    reward_baseline = running_reward_baseline / len(dataloader)
    return loss, reward, reward_baseline


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--exp_name', type=str, default='fsgr')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--m', type=int, default=40)
    parser.add_argument('--head', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--resume_last', action='store_true')
    parser.add_argument('--resume_best', action='store_true')
    parser.add_argument('--text', action='store_true')
    parser.add_argument('--return_index', action='store_true')
    parser.add_argument('--adapter_b', type=int, default=6)
    parser.add_argument('--adapter_e', type=int, default=11)
    parser.add_argument('--beta', type=float, default=0.25)

    parser.add_argument('--features_path', type=str, default='../../datasets/coco/images/')
    parser.add_argument('--labels_path', type=str, default='../local_text_label_trainval.hdf5')
    parser.add_argument('--annotation_folder', type=str, default='./m2_annotations')
    parser.add_argument('--text_embed_path', type=str, default='../asanet_vitb_supcon/pretrain/ram_ViT16_clip_text.pth')
    parser.add_argument('--pre_vs_path', type=str, default='../asanet_vitb_supcon/pretrain/clip/ViT-B-16.pt')
    parser.add_argument("--pre_name", type=str, default='ViT-B/16')
    parser.add_argument('--logs_folder', type=str, default='./tensorboard_logs')
    parser.add_argument('--xe_least', type=int, default=15)
    parser.add_argument('--xe_most', type=int, default=100)
    parser.add_argument('--refine_epoch_rl', type=int, default=28)

    parser.add_argument('--xe_base_lr', type=float, default=2e-4)
    parser.add_argument('--rl_base_lr', type=float, default=1e-5)
    # 只跑若干个 epoch（从 start_epoch 开始再跑这么多）
    parser.add_argument('--epochs', type=int, default=100,
                        help='总训练 epoch 数（从 start_epoch 开始再跑这么多）')

    args = parser.parse_args()
    print(args)
    print('Transformer Training')
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # 数据管道
    labels_path = args.labels_path if args.return_index else None
    image_field = ImageDetectionsField(
        detections_path=args.features_path, labels_path=labels_path,
        max_detections=49, load_in_tmp=False
    )
    text_field = TextField(
        init_token='<bos>', eos_token='<eos>', lower=True,
        tokenize='spacy', remove_punctuation=True, nopoints=False
    )

    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    # 词表
    os.makedirs('./vocab_language', exist_ok=True)
    if not os.path.isfile('./vocab_language/vocab.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('./vocab_language/vocab.pkl', 'wb'))
    else:
        print('Loading from vocabulary')
        text_field.vocab = pickle.load(open('./vocab_language/vocab.pkl', 'rb'))
        print(len(text_field.vocab))

    # 模型
    adapter_layer_list = [args.adapter_b, args.adapter_e]
    encoder = TransformerEncoder(
        2, 0, text=args.text,
        attention_module=ScaledDotProductAttention,
        attention_module_kwargs={'m': args.m}
    )
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(
        text_field.vocab.stoi['<bos>'], encoder, decoder,
        adapter_layer_list, pre_vs_path=args.pre_vs_path,
        text_emb_path=args.text_embed_path, pre_name=args.pre_name,
        text=args.text, return_index=args.return_index
    ).to(device)

    # 字典数据集（评估用）
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    # === 学习率调度（返回“乘法因子”） ===
    def lr_lambda(epoch):
        t = epoch + 1
        if t <= 3:
            return t / 4.0          # 0.25, 0.5, 0.75
        elif t <= 6:
            return 1.0
        elif t <= 12:
            return 0.2
        else:
            return 0.04

    def lr_lambda_rl(epoch):
        t = epoch + 1
        refine_epoch = args.refine_epoch_rl
        if t <= refine_epoch:
            return 1.0
        elif t <= refine_epoch + 3:
            return 0.2
        elif t <= refine_epoch + 6:
            return 0.04
        else:
            return 0.008

    # 优化器
    optim = build_optimizer(model)  # 通常分 backbone/decoder 两组
    # === 设置初始 lr，避免出现 0.0 的情况 ===
    if len(optim.param_groups) >= 1:
        optim.param_groups[0]['lr'] = args.xe_base_lr * 0.1   # backbone
    if len(optim.param_groups) >= 2:
        optim.param_groups[1]['lr'] = args.xe_base_lr         # decoder

    scheduler = LambdaLR(optim, lr_lambda)

    optim_rl = Adam(model.parameters(), lr=args.rl_base_lr, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lr_lambda_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    loss_contrast = SupConLoss(temperature=0.07)
    scaler = torch.cuda.amp.GradScaler()
    use_rl = False
    best_cider = 0.0
    best_test_cider = 0.0
    patience = 0
    start_epoch = 0

    # 可选断点恢复
    if args.resume_last or args.resume_best:
        fname = f'./save_models/{args.exp_name}_last.pth' if args.resume_last else './save_models/batch100_25.pth'
        if os.path.exists(fname):
            data = torch.load(fname, map_location=device)
            torch.set_rng_state(data['torch_rng_state'])
            if torch.cuda.is_available() and data.get('cuda_rng_state') is not None:
                torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            start_epoch = data['epoch'] + 1
            best_cider = data.get('best_cider', 0.0)
            best_test_cider = data.get('best_test_cider', 0.0)
            patience = data.get('patience', 0)
            use_rl = data.get('use_rl', False)

            if use_rl:
                optim_rl.load_state_dict(data['optimizer'])
                scheduler_rl.load_state_dict(data['scheduler'])
            else:
                optim.load_state_dict(data['optimizer'])
                scheduler.load_state_dict(data['scheduler'])

            print('Resuming from epoch %d, validation loss %f, best cider %f, best_test_cider %f' % (
                data['epoch'], data['val_loss'], best_cider, best_test_cider))
            print('patience:', patience)

    print("Training starts")
    num_epochs = int(getattr(args, "epochs", 100))
    for e in range(start_epoch, start_epoch + num_epochs):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True, num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field, beta=args.beta)
            writer.add_scalar('data/train_loss', train_loss, e)
            print(f"训练损失: {train_loss:.4f}")
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)
            print(f"训练损失: {train_loss:.4f}, 平均奖励: {reward:.4f}")

        # 验证
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, beta=args.beta)
        writer.add_scalar('data/val_loss', val_loss, e)
        print(f"验证损失: {val_loss:.4f}")

        # 验证集指标
        scores_val = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("验证集指标:", scores_val)
        val_cider = scores_val.get('CIDEr', 0.0)
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores_val['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores_val['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores_val.get('METEOR', 0.0), e)
        writer.add_scalar('data/val_rouge', scores_val.get('ROUGE', 0.0), e)

        # 测试集指标
        scores_test = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("测试集指标:", scores_test)
        test_cider = scores_test.get('CIDEr', 0.0)
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores_test['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores_test['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores_test.get('METEOR', 0.0), e)
        writer.add_scalar('data/test_rouge', scores_test.get('ROUGE', 0.0), e)

        # 早停/切换RL逻辑（保持与原逻辑一致）
        best = False
        if val_cider >= best_cider:
            best_cider = val_cider
            patience = 0
            best = True
        else:
            patience += 1

        best_test = False
        if test_cider >= best_test_cider:
            best_test_cider = test_cider
            best_test = True

        switch_to_rl = False
        exit_train = False

        if patience == 5:
            if e < args.xe_least:
                print('special treatment, e = {}'.format(e))
                use_rl = False
                switch_to_rl = False
                patience = 0
            elif not use_rl:
                use_rl = True
                switch_to_rl = True
                patience = 0
                optim_rl = Adam(model.parameters(), lr=args.rl_base_lr, betas=(0.9, 0.98))
                scheduler_rl = LambdaLR(optim_rl, lr_lambda_rl)
                for _ in range(e - 1):
                    scheduler_rl.step()
                print("Switching to RL")
            else:
                print('patience reached.')
                exit_train = True

        if e == args.xe_most and not use_rl:
            use_rl = True
            switch_to_rl = True
            patience = 0
            optim_rl = Adam(model.parameters(), lr=args.rl_base_lr, betas=(0.9, 0.98))
            scheduler_rl = LambdaLR(optim_rl, lr_lambda_rl)
            for _ in range(e - 1):
                scheduler_rl.step()
            print("Switching to RL")

        if switch_to_rl and not best:
            best_path = f'./save_models/{args.exp_name}_best.pth'
            if os.path.exists(best_path):
                data = torch.load(best_path, map_location=device)
                model.load_state_dict(data['state_dict'])
                print('Resuming from epoch %d, val_loss %f, best_cider %f, best_test_cider %f' % (
                    data['epoch'], data['val_loss'], data.get('best_cider', 0.0), data.get('best_test_cider', 0.0)
                ))

        # 保存
        save_obj = {
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate(),
            'epoch': e,
            'val_loss': val_loss,
            'val_cider': val_cider,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict() if not use_rl else optim_rl.state_dict(),
            'scheduler': scheduler.state_dict() if not use_rl else scheduler_rl.state_dict(),
            'patience': patience,
            'best_cider': best_cider,
            'best_test_cider': best_test_cider,
            'use_rl': use_rl,
        }
        os.makedirs('./save_models', exist_ok=True)
        torch.save(save_obj, f'./save_models/{args.exp_name}_last.pth')

        if switch_to_rl:
            copyfile(f'./save_models/{args.exp_name}_best.pth', f'./save_models/{args.exp_name}_ce_stage1.pth')
        if best:
            copyfile(f'./save_models/{args.exp_name}_last.pth', f'./save_models/{args.exp_name}_best.pth')
        if best_test:
            copyfile(f'./save_models/{args.exp_name}_last.pth', f'./save_models/{args.exp_name}_best_test.pth')

        if e >= 55:
            copyfile(f'./save_models/{args.exp_name}_last.pth', f'./save_models/{args.exp_name}_{e}.pth')

        if exit_train:
            writer.close()
            break

    writer.close()
