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

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ====== 快速评估/限步控制（环境变量）======
FAST_EVAL   = os.getenv("FSGR_FAST_EVAL", "0") == "1"     # 评估阶段限步
EVAL_STEPS  = int(os.getenv("FSGR_EVAL_STEPS", "0"))      # 评估最多步数（仅 FAST_EVAL=1 生效）
VAL_STEPS   = int(os.getenv("FSGR_VAL_STEPS", "0"))       # 验证损失最多步数
TRAIN_STEPS = int(os.getenv("FSGR_TRAIN_STEPS", "0"))     # 训练阶段最多步数
PRINT_EVERY = 50


def _parse_trainval_batch(batch, device):
    """
    兼容以下三种 batch 形态，返回 (detections, labels_or_None, captions):
      1) (detections, labels, captions)                      # 原训练/验证三元组
      2) (image_ids, images, placeholder, captions)          # 4 元，直接读图像
      3) ((images, _), caps)                                 # 2 元（dict风格，无 labels）
    """
    import torch
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            detections, labels, captions = batch
            return detections.to(device), labels.to(device), captions.to(device)
        if len(batch) == 4:
            _, images, _, captions = batch
            return images.to(device), None, captions.to(device)
        if len(batch) == 2:
            img_pack, caps = batch
            if isinstance(img_pack, (list, tuple)) and len(img_pack) >= 1:
                images = img_pack[0]
            else:
                images = img_pack
            # caps 可能是 list（字符串）或 Tensor；Tensor 则留到调用处处理
            return images.to(device), None, caps
    raise ValueError(f"[parse] 不支持的batch结构: type={type(batch)}, len={len(batch) if hasattr(batch,'__len__') else 'NA'}")


def _extract_eval_batch(batch, device):
    """
    评估用：尽量从各种结构里提取 (images_tensor[B,3,H,W], caps_gt(list-of-str))
    兼容：
      a) ((images, _), caps_gt)
      b) (ids, images, placeholder, caps_gt)
      c) (images, caps_gt)
      d) (ids, (images,_), caps_gt)
    """
    import torch

    def _find_images(x):
        if torch.is_tensor(x) and x.ndim == 4 and x.shape[1] in (1, 3):
            return x
        if isinstance(x, (list, tuple)):
            for y in x:
                if torch.is_tensor(y) and y.ndim == 4 and y.shape[1] in (1, 3):
                    return y
        return None

    if not isinstance(batch, (list, tuple)):
        raise ValueError(f"[eval_parse] 非期望batch类型: {type(batch)}")

    # 先尝试常见的二元 ((images,_), caps_gt)
    if len(batch) == 2:
        img_pack, caps_gt = batch
        images = _find_images(img_pack)
        if images is None:
            images = _find_images(batch)
        if images is None:
            raise ValueError("[eval_parse] 找不到图像张量")
        return images.to(device), caps_gt

    # 四元组： (ids, images, placeholder, caps_gt)
    if len(batch) == 4:
        _, images, _, caps_gt = batch
        images = _find_images(images)
        if images is None:
            raise ValueError("[eval_parse] 四元组中找不到图像张量")
        return images.to(device), caps_gt

    # 三元组：尽力猜测 (ids/imgs/..., imgs/..., caps_gt)
    if len(batch) == 3:
        images = None
        caps_gt = None
        for x in batch:
            if images is None:
                images = _find_images(x)
        # caps_gt 优先选择 list/tuple[str]
        for x in reversed(batch):
            if x is not images and (isinstance(x, (list, tuple)) or isinstance(x, str)):
                caps_gt = x
                break
        if images is None or caps_gt is None:
            raise ValueError("[eval_parse] 三元组中无法解析出 (images, caps_gt)")
        return images.to(device), caps_gt

    raise ValueError(f"[eval_parse] 非期望长度: {len(batch)}")


def evaluate_loss(model, dataloader, loss_fn, text_field, beta=0.25):
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
                        # SupCon 只有有 labels 才计算
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

    return running_loss / max(1, (VAL_STEPS if VAL_STEPS > 0 else len(dataloader)))


def evaluate_metrics(model, dataloader, text_field):
    model.eval()
    gen, gts = {}, {}
    use_cap = FAST_EVAL and EVAL_STEPS > 0
    total_steps = min(len(dataloader), EVAL_STEPS) if use_cap else len(dataloader)

    with tqdm(desc='Epoch %d - evaluation' % e, unit='it', total=total_steps) as pbar:
        for it, batch in enumerate(dataloader):
            images, caps_gt = _extract_eval_batch(batch, device)
            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)

            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, _ in itertools.groupby(gen_i)])
                gen[f'{it}_{i}'] = [gen_i]
                gts[f'{it}_{i}'] = gts_i

            pbar.update(1)
            if use_cap and (it + 1) >= EVAL_STEPS:
                break

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)
    return scores


def train_xe(model, dataloader, optim, text_field, beta=0.25):
    model.train()
    print('Backbone lr = ', optim.param_groups[0]['lr'])
    if len(optim.param_groups) > 1:
        print('Dec lr = ', optim.param_groups[1]['lr'])

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

            optim.zero_grad(set_to_none=True)
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
    tokenizer_pool = multiprocessing.Pool()
    running_reward = 0.0
    running_reward_baseline = 0.0

    model.train()
    print('RL lr = ', optim.param_groups[0]['lr'])
    running_loss = 0.0
    seq_len = 20
    beam_size = 5

    with tqdm(desc='Epoch %d - train' % e, unit='it', total=len(dataloader)) as pbar:
        for it, ((detections, _), caps_gt) in enumerate(dataloader):
            detections = detections.to(device)
            with torch.cuda.amp.autocast():
                outs, log_probs = model.beam_search(detections, seq_len, text_field.vocab.stoi['<eos>'],
                                                    beam_size, out_size=beam_size)
                caps_gen = text_field.decode(outs.view(-1, seq_len))
                caps_gt = list(itertools.chain(*([c, ] * beam_size for c in caps_gt)))
                caps_gen, caps_gt = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [caps_gen, caps_gt])
                reward = cider.compute_score(caps_gt, caps_gen)[1].astype(np.float32)
                reward = torch.from_numpy(reward).to(device).view(detections.shape[0], beam_size)
                reward_baseline = torch.mean(reward, -1, keepdim=True)
                loss = -torch.mean(log_probs, -1) * (reward - reward_baseline)
                loss = loss.mean()

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            running_loss += float(loss.item())
            running_reward += float(reward.mean().item())
            running_reward_baseline += float(reward_baseline.mean().item())
            pbar.set_postfix(loss=round(running_loss / (it + 1), 4),
                             reward=round(running_reward / (it + 1), 4),
                             reward_baseline=round(running_reward_baseline / (it + 1), 4))
            pbar.update(1)
    scheduler_rl.step()
    tokenizer_pool.close()
    tokenizer_pool.join()
    return (running_loss / len(dataloader),
            running_reward / len(dataloader),
            running_reward_baseline / len(dataloader))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    parser.add_argument('--epochs', type=int, default=100,
                        help='总训练 epoch 数（从 start_epoch 开始再跑这么多）')

    args = parser.parse_args()
    print(args)
    print('Transformer Training')
    writer = SummaryWriter(log_dir=os.path.join(args.logs_folder, args.exp_name))

    # 数据
    labels_path = args.labels_path if args.return_index else None
    image_field = ImageDetectionsField(detections_path=args.features_path, labels_path=labels_path, max_detections=49, load_in_tmp=False)
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    dataset = COCO(image_field, text_field, args.features_path, args.annotation_folder, args.annotation_folder)
    train_dataset, val_dataset, test_dataset = dataset.splits

    if not os.path.isfile('./vocab_language/vocab.pkl'):
        print("Building vocabulary")
        text_field.build_vocab(train_dataset, val_dataset, min_freq=5)
        pickle.dump(text_field.vocab, open('vocab.pkl', 'wb'))
    else:
        print('Loading from vocabulary')
        text_field.vocab = pickle.load(open('./vocab_language/vocab.pkl', 'rb'))
        print(len(text_field.vocab))

    # 模型
    adapter_layer_list = [args.adapter_b, args.adapter_e]
    encoder = TransformerEncoder(2, 0, text=args.text,
                                 attention_module=ScaledDotProductAttention,
                                 attention_module_kwargs={'m': args.m})
    decoder = TransformerDecoderLayer(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder,
                        adapter_layer_list, pre_vs_path=args.pre_vs_path,
                        text_emb_path=args.text_embed_path, pre_name=args.pre_name,
                        text=args.text, return_index=args.return_index).to(device)

    # dict 数据集（评估/rl）
    dict_dataset_train = train_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    ref_caps_train = list(train_dataset.text)
    cider_train = Cider(PTBTokenizer.tokenize(ref_caps_train))
    dict_dataset_val = val_dataset.image_dictionary({'image': image_field, 'text': RawField()})
    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField()})

    # ===== 学习率调度（注意：LambdaLR 返回的是乘法因子）=====
    def lambda_lr(epoch):
        # 用 epoch+1 保证第1个 epoch 就有 0.25 倍率
        t = epoch + 1
        if t <= 3:
            return t / 4.0       # 0.25, 0.5, 0.75
        elif t <= 6:
            return 1.0
        elif t <= 12:
            return 0.2
        else:
            return 0.04

    def lambda_lr_rl(epoch):
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

    # ===== 优化器与初始 lr（关键：先设 non-zero base lr 再用 LambdaLR 乘法因子）=====
    optim = build_optimizer(model)              # 通常分两组：backbone / decoder
    if len(optim.param_groups) >= 1:
        optim.param_groups[0]['lr'] = args.xe_base_lr * 0.1   # backbone 小 10 倍
    if len(optim.param_groups) >= 2:
        optim.param_groups[1]['lr'] = args.xe_base_lr         # decoder 基础 lr

    scheduler = LambdaLR(optim, lambda_lr)

    optim_rl = Adam(model.parameters(), lr=args.rl_base_lr, betas=(0.9, 0.98))
    scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)

    loss_fn = NLLLoss(ignore_index=text_field.vocab.stoi['<pad>'])
    loss_contrast = SupConLoss(temperature=0.07)
    scaler = torch.cuda.amp.GradScaler()
    use_rl = False
    best_cider = 0.0
    best_test_cider = 0.0
    patience = 0
    start_epoch = 0

    # 断点恢复
    if args.resume_last or args.resume_best:
        fname = './save_models/%s_last.pth' % args.exp_name if args.resume_last else './save_models/batch100_25.pth'
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
            print('Resuming from epoch %d, best_cider %f, best_test_cider %f' %
                  (data['epoch'], best_cider, best_test_cider))

    print("Training starts")
    num_epochs = int(getattr(args, "epochs", 100))
    for e in range(start_epoch, start_epoch + num_epochs):
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                      drop_last=True)
        dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        dict_dataloader_train = DataLoader(dict_dataset_train, batch_size=args.batch_size // 5, shuffle=True,
                                           num_workers=args.workers)
        dict_dataloader_val = DataLoader(dict_dataset_val, batch_size=args.batch_size // 5)
        dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size // 5)

        if not use_rl:
            train_loss = train_xe(model, dataloader_train, optim, text_field, beta=args.beta)
            writer.add_scalar('data/train_loss', train_loss, e)
        else:
            train_loss, reward, reward_baseline = train_scst(model, dict_dataloader_train, optim_rl, cider_train, text_field)
            writer.add_scalar('data/train_loss', train_loss, e)
            writer.add_scalar('data/reward', reward, e)
            writer.add_scalar('data/reward_baseline', reward_baseline, e)

        # 验证
        val_loss = evaluate_loss(model, dataloader_val, loss_fn, text_field, beta=args.beta)
        writer.add_scalar('data/val_loss', val_loss, e)

        # 验证集指标
        scores = evaluate_metrics(model, dict_dataloader_val, text_field)
        print("Validation scores", scores)
        val_cider = scores['CIDEr']
        writer.add_scalar('data/val_cider', val_cider, e)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/val_meteor', scores.get('METEOR', 0.0), e)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], e)

        # 测试集指标
        scores = evaluate_metrics(model, dict_dataloader_test, text_field)
        print("Test scores", scores)
        test_cider = scores['CIDEr']
        writer.add_scalar('data/test_cider', test_cider, e)
        writer.add_scalar('data/test_bleu1', scores['BLEU'][0], e)
        writer.add_scalar('data/test_bleu4', scores['BLEU'][3], e)
        writer.add_scalar('data/test_meteor', scores.get('METEOR', 0.0), e)
        writer.add_scalar('data/test_rouge', scores['ROUGE'], e)

        # Early stopping & RL切换（维持你原逻辑）
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
                scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
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
            scheduler_rl = LambdaLR(optim_rl, lambda_lr_rl)
            for _ in range(e - 1):
                scheduler_rl.step()
            print("Switching to RL")

        if switch_to_rl and not best:
            data = torch.load('./save_models/%s_best.pth' % args.exp_name, map_location=device)
            model.load_state_dict(data['state_dict'])
            print('Resumed best CE model to start RL')

        # 保存
        save_data = {
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
        torch.save(save_data, './save_models/%s_last.pth' % args.exp_name)

        if switch_to_rl:
            copyfile('./save_models/%s_best.pth' % args.exp_name, './save_models/%s_ce_stage1.pth' % args.exp_name)
        if best:
            copyfile('./save_models/%s_last.pth' % args.exp_name, './save_models/%s_best.pth' % args.exp_name)
        if best_test:
            copyfile('./save_models/%s_last.pth' % args.exp_name, './save_models/%s_best_test.pth' % args.exp_name)
        if e >= 55:
            copyfile('./save_models/%s_last.pth' % args.exp_name, './save_models/{}_{}.pth'.format(args.exp_name, e))
        if exit_train:
            writer.close()
            break

    writer.close()
