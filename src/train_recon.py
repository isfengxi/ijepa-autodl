# train_recon.py
import os, copy, sys, yaml, logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.tensors import repeat_interleave_batch
from src.datasets.csi_pt import make_csi_pt
from src.transforms import make_csi_transforms
from src.helper import load_checkpoint, init_model

from src.utils.patchify import patchify_2d
from src.models.pixel_decoder import PixelDecoderMLP

logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def freeze_(m):
    for p in m.parameters():
        p.requires_grad = False

def init_decoder_opt(decoder, lr, wd):
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=wd)
    return opt

def main(args):
    # ---- META
    use_bfloat16 = args['meta'].get('use_bfloat16', False)
    model_name = args['meta']['model_name']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    ckpt_path = args['meta']['read_checkpoint']  # 直接给具体ckpt路径
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # ---- DATA
    root_path = args['data']['root_path']
    crop_size = args['data']['crop_size']      # 128
    in_chans  = args['data'].get('in_chans', 16)
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    noise_sigma = args['data'].get('noise_sigma', 0.0)
    normalization = args['data'].get('normalization', None)

    # ---- MASK（保持与你 train.py 一致）
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # ---- RECON OPT
    recon_epochs = args['recon']['epochs']
    dec_lr = args['recon']['lr']
    dec_wd = args['recon']['weight_decay']
    hidden_mult = args['recon']['decoder'].get('hidden_mult', 4)
    dropout = args['recon']['decoder'].get('dropout', 0.0)

    # ---- LOG
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'params-recon.yaml'), 'w') as f:
        yaml.dump(args, f)

    # ---- distributed（沿用你的做法）
    # world_size, rank = init_distributed()
    world_size, rank = 1, 0
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.6f', 'recon_loss'))

    # ---- init encoder/predictor（与 train.py 同源）
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        in_chans=in_chans,
    )

    target_encoder = copy.deepcopy(encoder)  # 仅用于 load_checkpoint 对齐（不参与recon训练）

    # ---- masks + data（与你 train.py 同构）
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep
    )

    transform = make_csi_transforms(
        crop_size=crop_size,
        noise_sigma=noise_sigma,
        normalization=normalization,
    )

    _, loader, sampler = make_csi_pt(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        training=True,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        drop_last=True,
    )

    # ---- wrap DDP（保持与 load_checkpoint 存的 state_dict key 一致的可能性）
    # encoder = DistributedDataParallel(encoder, static_graph=True)
    # predictor = DistributedDataParallel(predictor, static_graph=True)
    # target_encoder = DistributedDataParallel(target_encoder)

    # ---- load pretrained checkpoint（复用你现有的 load_checkpoint）
    encoder, predictor, target_encoder, _, _, _ = load_checkpoint(
        device=device,
        r_path=ckpt_path,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        opt=None,
        scaler=None
    )

    # ---- freeze backbone (方案A：只训decoder)
    freeze_(encoder)
    freeze_(predictor)
    encoder.eval()
    predictor.eval()

    # ---- build pixel decoder
    # token_dim 用 pred_emb_dim（你config里 predictor embedding dim）
    patch_dim = in_chans * patch_size * patch_size
    token_dim = getattr(encoder, 'embed_dim', None) or getattr(encoder, 'num_features', None)
    if token_dim is None:
        token_dim = 768  # fallback for vit_base
    decoder = PixelDecoderMLP(token_dim=token_dim, patch_dim=patch_dim,
                          hidden_mult=hidden_mult, dropout=dropout).to(device)
    # decoder = DistributedDataParallel(decoder, static_graph=True)

    dec_opt = init_decoder_opt(decoder, lr=dec_lr, wd=dec_wd)

    # ---- train loop
    for epoch in range(recon_epochs):
        logger.info(f'Epoch {epoch+1}/{recon_epochs}')
        sampler.set_epoch(epoch)

        loss_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(loader):
            imgs = udata[0].to(device, non_blocking=True)
            masks_enc = [u.to(device, non_blocking=True) for u in masks_enc]
            masks_pred = [u.to(device, non_blocking=True) for u in masks_pred]

            def train_step():
                # 线1：拿预测token（完全对齐你的 forward_context）：
                with torch.no_grad():
                    z = encoder(imgs, masks_enc)          # [B', K_enc, 768]  (K_enc = visible tokens)

                patches = patchify_2d(imgs, patch_size)   # [B, N, patch_dim]
                gt = apply_masks(patches, masks_enc)      # 用 masks_enc 对齐 z 的 token集合
                B = patches.shape[0]
                gt = repeat_interleave_batch(gt, B, repeat=len(masks_enc))

                pred = decoder(z)                         # [B', K_enc, patch_dim]
                loss = F.mse_loss(pred, gt)
                # loss = AllReduce.apply(loss)  # 和你 train.py 一样做全卡reduce（单卡也没坏处）
                dec_opt.zero_grad(set_to_none=True)
                loss.backward()
                dec_opt.step()
                return float(loss)

            loss, _ = gpu_timer(train_step)
            loss_meter.update(loss)
            csv_logger.log(epoch + 1, itr, loss)

            if itr % 10 == 0 and rank == 0:
                logger.info(f'[{epoch+1},{itr}] recon_loss(avg): {loss_meter.avg:.6f}')

        if rank == 0:
            torch.save({
                'decoder': decoder.state_dict(),
                'meta': {
                    'patch_size': patch_size,
                    'in_chans': in_chans,
                    'patch_dim': patch_dim,
                    'pred_emb_dim': pred_emb_dim,
                },
                'epoch': epoch + 1,
                'recon_loss': loss_meter.avg,
                'backbone_ckpt': ckpt_path,
            }, latest_path)

if __name__ == "__main__":
    main(yaml.safe_load(open(sys.argv[1], 'r')))

