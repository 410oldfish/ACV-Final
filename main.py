"""
This script allows you to generate images using either the Stable Diffusion (SD) or EDM backend,
with a choice of scorer and sampling method

Usage examples:
    python main.py --backend sd --scorer brightness --method naive --prompt "A beautiful landscape"
    python main.py --backend edm --scorer imagenet --method zero_order

Arguments:
    --backend   : 'sd' or 'edm' (required)
    --scorer    : 'brightness', 'compressibility', 'clip', or 'imagenet' (required)
    --method    : Sampling method (available: 'naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy') (default: 'naive')
    --prompt    : Prompt for SD (default: 'A beautiful landscape')
    --output    : Output filename
    --N, --lambda_, --eps, --K, --B, --S : sampling parameters (see code for defaults)
    --seed      : Random seed (default: 0)
    --device    : Device (default: 'cuda')
"""
import os
import sys
import argparse
import importlib.util
import torch
from pathlib import Path
from PIL import Image, ImageOps
import importlib
import torch, torch.nn.functional as F
# 复用项目里的 CLIP scorer
from sd.scorers import CLIPScorer  # 注意：你SD分支的scorers在 sd/ 目录下
import numpy as np

#custom timesteps
from types import MethodType

def _set_timesteps_with_custom(self, num_inference_steps=None, device=None, timesteps=None, **kwargs):
    if timesteps is None:
        # 沿用原始行为
        return type(self).set_timesteps(self, num_inference_steps=num_inference_steps, device=device)
    import torch
    self.timesteps = torch.as_tensor(timesteps, dtype=torch.long, device=device)
    self.num_inference_steps = int(self.timesteps.numel())

def vae_decode_to_image(latents, pipe):
    # latents: (B,4,H,W) → VAE decode → (B,3,H*8,W*8) in [0,1]
    latents = latents / pipe.vae.config.scaling_factor
    with torch.no_grad():
        imgs = pipe.vae.decode(latents).sample  # [-1,1]
    imgs = (imgs.clamp(-1, 1) + 1) / 2.0
    return imgs

def score_image_lowres(img, scorer_fn, downscale_factor=4):
    # img: (1,3,H,W) in [0,1]
    if downscale_factor and downscale_factor > 1:
        h, w = img.shape[-2:]
        img = F.interpolate(img, size=(max(1,h//downscale_factor), max(1,w//downscale_factor)),
                            mode='area', align_corners=None)
    s = scorer_fn(img)  # 直接用你现有的 scorer
    if isinstance(s, torch.Tensor):
        s = float(s.detach().cpu().item())
    elif hasattr(s, 'item'):
        s = float(s.item())
    else:
        s = float(s)
    return s


# =========================
# EDM Import Helper
# =========================
def import_edm():
    """Dynamically import EDM modules and scorers."""
    edm_dir = Path(__file__).parent / 'edm'
    sys.path.insert(0, str(edm_dir))
    dnnlib = importlib.import_module('dnnlib')
    dnnlib_util = importlib.import_module('dnnlib.util')
    from scorers import BrightnessScorer, CompressibilityScorer, ImageNetScorer, OneStepGenerationScorer
    return dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer

# =========================
# SD Import Helper
# =========================
def import_sd():
    """Dynamically import SD pipeline and scorers."""
    sd_dir = Path(__file__).parent / 'sd'
    diffusers_path = sd_dir / 'diffusers' / 'src' / 'diffusers' / '__init__.py'
    spec = importlib.util.spec_from_file_location('diffusers', str(diffusers_path.resolve()))
    sys.modules['diffusers'] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules['diffusers'])
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    sys.path.insert(0, str(sd_dir))
    from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer, ImageRewardScorer, OneStepGenerationScorer
    return StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer, ImageRewardScorer, OneStepGenerationScorer

# =========================
# Scorer Factory
# =========================
def get_scorer(backend, scorer_name, BrightnessScorer, CompressibilityScorer, CLIPScorer=None, ImageNetScorer=None, ImageRewardScorer=None, OneStepGenerationScorer=None):
    """Return the appropriate scorer instance for the backend and scorer name."""
    if scorer_name == 'brightness':
        return BrightnessScorer(dtype=torch.float32)
    elif scorer_name == 'compressibility':
        return CompressibilityScorer(dtype=torch.float32)
    elif scorer_name == 'clip' and backend == 'sd':
        return CLIPScorer(dtype=torch.float32)
    elif scorer_name == 'imagereward' and backend == 'sd':
        return ImageRewardScorer(dtype=torch.float32)
    elif scorer_name == 'onestepgeneration' and backend == 'sd':
        return OneStepGenerationScorer(dtype=torch.float32)
    elif scorer_name == 'imagenet' and backend == 'edm':
        return ImageNetScorer(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown or invalid scorer '{scorer_name}' for backend '{backend}'")

# =========================
# AYS
# =========================
def gen_ays_sd15_timesteps(num_steps: int = 50):
    """
    用 AYS 给的 SD1.5 10 步锚点做对数线性插值到 num_steps+1 个“节点”，
    返回降序的 timestep 索引列表（不包含最后的 0，方便直接传 scheduler）。
    """
    import numpy as np
    anchors = np.array([999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0], dtype=np.float64)

    xs = np.linspace(0, 1, len(anchors))
    ys = np.log(anchors[::-1] + 1)      # log 域插值（+1 防止 log(0)）
    xs_new = np.linspace(0, 1, num_steps + 1)  # 包含终点 0 的节点数是 steps+1
    ys_new = np.interp(xs_new, xs, ys)
    ts = np.exp(ys_new) - 1
    ts = ts[::-1]                        # 降序，大→小
    ts = np.clip(np.round(ts).astype(int), 0, 999).tolist()

    # scheduler.timesteps 通常不包含最后的 0（0 是终点而非迭代步）
    if ts and ts[-1] == 0:
        ts = ts[:-1]
    # 去重（防止相邻重复）
    dedup = []
    last = None
    for v in ts:
        if v != last:
            dedup.append(v)
            last = v
    return dedup



# =========================
# Main Logic
# =========================
def main():
    # -----------
    # CLI Arguments
    # -----------
    parser = argparse.ArgumentParser(
        description='Unified Diffusion Image Generator (EDM/SD)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--backend', type=str, choices=['edm', 'sd'], required=True, help='Backend: edm or sd')
    parser.add_argument('--scorer', type=str, choices=['brightness', 'compressibility', 'clip', 'imagenet', 'imagereward', 'onestepgeneration'], required=True, help='Scorer name')
    parser.add_argument('--method', type=str, default='naive', help='Sampling method (naive, rejection, beam, mcts, zero_order, eps_greedy)')
    parser.add_argument('--prompt', type=str, default='YOUR PROMPT HERE', help='Prompt for SD')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: auto)')
    # Master params (with SD defaults)
    parser.add_argument('--N', type=int, default=4, help='Master param N')
    parser.add_argument('--lambda_', type=float, default=0.15, help='Master param lambda')
    parser.add_argument('--eps', type=float, default=0.4, help='Master param eps')
    parser.add_argument('--K', type=int, default=20, help='Master param K')
    parser.add_argument('--B', type=int, default=2, help='Master param B')
    parser.add_argument('--S', type=int, default=8, help='Master param S')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    # New parameters
    parser.add_argument('--rollout_steps', type=int, default=0,
    help='>0 启用多步前瞻评分：每个候选额外前瞻的步数 r（建议 2~3）。')
    parser.add_argument('--rollout_gap', type=int, default=1,
    help='前瞻步的“跳步间隔”，gap=2 表示每次跳过 1 个 timestep。')
    parser.add_argument('--downscale_factor', type=int, default=4,
    help='前瞻末尾解码后的下采样比例；4 表示边长/4。')
    parser.add_argument('--rollout_only_mid', action='store_true',
    help='只在中间 σ 区间做前瞻，其他步仍用即时 x̂0 打分（省算力）。')

    #Top-M , auto-rollout
    parser.add_argument('--refine_threshold', type=float, default=0.015,
        help="Only trigger Top-M re-eval if (top1 - top2) < threshold; set 0 to disable")
    parser.add_argument('--refine_top_m', type=int, default=3,
        help="How many top candidates to re-evaluate when triggered")
    parser.add_argument('--refine_r', type=int, default=2,
        help="Rollout steps used for Top-M re-eval")
    parser.add_argument('--refine_gap', type=int, default=3,
        help="Gap used for Top-M re-eval")
    parser.add_argument('--refine_downscale', type=int, default=1,
        help="Downscale for Top-M re-eval (1 = no downscale)")

    #AYS
    parser.add_argument('--ays_enable', action='store_true',
    help="Use Align-Your-Steps schedule for SD.")
    parser.add_argument('--ays_steps', type=int, default=50,
    help="Number of inference steps when AYS is enabled (e.g., 50).")

    #eps last k boost
    parser.add_argument('--final_last_k', type=int, default=6,
    help='Increase exploration epsilon on the last K steps (linear ramp).')
    parser.add_argument('--final_eps_boost', type=float, default=2.0,
    help='Boost factor for epsilon at the very last step (2.0 => up to 3x).')

    #clip eval k get best in last step
    parser.add_argument('--final_topk', type=int, default=0,
    help='在最后一步按“环内分数”粗排，取 Top-K 再用 CLIP 终选（0 关闭）')
    parser.add_argument('--final_eval', type=str, default=None)
    parser.add_argument('--final_eval_views', type=int, default=1)
    parser.add_argument('--final_eval_center_crop', action='store_true')

    args = parser.parse_args()

    # -----------
    # Validation
    # -----------
    if args.backend == 'sd' and args.scorer == 'imagenet':
        raise ValueError('imagenet scorer is only available for edm backend')
    if args.backend == 'edm' and args.scorer == 'clip':
        raise ValueError('clip scorer is only available for sd backend')

    # -----------
    # SD Backend
    # -----------
    if args.backend == 'sd':
        StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer, ImageRewardScorer, OneStepGenerationScorer= import_sd()
        scorer = get_scorer('sd', args.scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer, ImageRewardScorer=ImageRewardScorer, OneStepGenerationScorer=OneStepGenerationScorer)

        model_id = "runwayml/stable-diffusion-v1-5"
        local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        local_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=local_scheduler,
            torch_dtype=torch.float16,
        ).to(args.device)
        #add method
        # 绑定到当前 scheduler，确保签名里有 `timesteps`（让 retrieve_timesteps 的检查通过）
        local_pipe.scheduler.set_timesteps = MethodType(_set_timesteps_with_custom, local_pipe.scheduler)

        method = args.method
        MASTER_PARAMS = {
            'N': args.N,
            'lambda': args.lambda_,
            'eps': args.eps,
            'K': args.K,
            'B': args.B,
            'S': args.S,
        }
        MASTER_PARAMS.update({
            'rollout_steps': args.rollout_steps,
            'rollout_gap': args.rollout_gap,
            'downscale_factor': args.downscale_factor,
            'rollout_only_mid': args.rollout_only_mid,
        })
        MASTER_PARAMS.update({
            'refine_threshold': args.refine_threshold,
            'refine_top_m': args.refine_top_m,
            'refine_r': args.refine_r,
            'refine_gap': args.refine_gap,
            'refine_downscale': args.refine_downscale,
        })
        MASTER_PARAMS.update({
            'final_topk': args.final_topk,
            'final_last_k': args.final_last_k,
            'final_eps_boost': args.final_eps_boost,
        })



        best_result, best_score = None, float('-inf')
        for _ in range(MASTER_PARAMS['N'] if method == "rejection" else 1):
          if args.ays_enable :
            # 生成 50 步的 AYS 时刻索引（降序，不含 0）
            ays_ts = gen_ays_sd15_timesteps(num_steps=50)
            print(f"[AYS] custom timesteps ({len(ays_ts)}): head={ays_ts[:5]} ... tail={ays_ts[-5:]}", flush=True)

            result, score = local_pipe(
                prompt=args.prompt,
                num_inference_steps=len(ays_ts),    # 与 timesteps 长度一致
                timesteps=ays_ts,                   # ★ 关键：把 AYS timesteps 直接传入
                score_function=scorer,
                method=method,
                params=MASTER_PARAMS,
            )
          else:
            
            result, score = local_pipe(
                prompt=args.prompt,
                num_inference_steps=50,
                score_function=scorer,
                method=method,
                params=MASTER_PARAMS,
            )

            
            

          if score > best_score:
              best_result, best_score = result, score

        outname = args.output or f"sd_{method}_{args.scorer}.png"
        best_result.images[0].save(outname)
        # ===== Final evaluation (optional) =====
        if args.final_eval is not None and args.final_eval.lower() == 'clip':
            # 支持多视角与可选中心裁剪的 CLIP 终评

            def _to_uint8_tensor(pil_img):
                arr = np.array(pil_img.convert("RGB"), copy=False)  # H,W,3 uint8
                return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W uint8

            def _gen_views(img_pil, n_views: int, center_crop: bool):
                views = []
                base = img_pil
                if center_crop:
                    # 先等边填充为正方形，再做中心裁剪/缩放由 CLIPScorer 内部处理
                    side = max(img_pil.size)
                    pad_w = side - img_pil.width
                    pad_h = side - img_pil.height
                    base = ImageOps.expand(img_pil, border=(0, 0, pad_w, pad_h), fill=0)
                views.append(base)

                # 四角轻裁（提高稳健性）
                if n_views >= 4:
                    w, h = img_pil.size
                    crops = [
                        img_pil.crop((0, 0, int(0.9 * w), int(0.9 * h))),
                        img_pil.crop((int(0.1 * w), 0, w, int(0.9 * h))),
                        img_pil.crop((0, int(0.1 * h), int(0.9 * w), h)),
                        img_pil.crop((int(0.1 * w), int(0.1 * h), w, h)),
                    ]
                    views.extend(crops)

                # 水平翻转对应的视角
                if n_views >= 8:
                    views.extend([v.transpose(Image.FLIP_LEFT_RIGHT) for v in views[:4]])

                # 截至到 n_views
                return views[:max(1, n_views)]

            n_views = getattr(args, "final_eval_views", 1)
            use_center = bool(getattr(args, "final_eval_center_crop", False))

            scorer_eval = CLIPScorer(dtype=torch.float32)
            img = Image.open(outname).convert('RGB')
            views = _gen_views(img, n_views=n_views, center_crop=use_center)

            scores = []
            for v in views:
                t = _to_uint8_tensor(v)
                s = scorer_eval(images=[t], prompts=[args.prompt], timesteps=None)
                s = float(s.item() if torch.is_tensor(s) else s)
                scores.append(s)

            if len(scores) == 1:
                print(f"[final-eval] CLIP score = {scores[0]:.6f}", flush=True)
            else:
                mean = float(np.mean(scores)); std = float(np.std(scores))
                print(f"[final-eval] CLIP multi-view (n={len(scores)}) = {mean:.6f} ± {std:.6f}", flush=True)

        print(f"\n[SD] Saved: {outname}\n")


    # -----------
    # EDM Backend
    # -----------
    elif args.backend == 'edm':
        dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer, ImageRewardScorer, OneStepGenerationScorer = import_edm()
        scorer = get_scorer('edm', args.scorer, BrightnessScorer, CompressibilityScorer, ImageNetScorer=ImageNetScorer, ImageRewardScorer=None, OneStepGenerationScorer=None)

        # EDM defaults
        model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
        network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
        num_images = 1
        gridw = gridh = 1
        latents = torch.randn([num_images, 3, 64, 64])
        class_labels = torch.eye(1000)[torch.randint(1000, size=[num_images])]
        device = torch.device(args.device)
        num_steps = 18

        # EDM method mapping
        from edm.main import SamplingMethod, generate_image_grid
        method_map = {
            'naive': SamplingMethod.NAIVE,
            'rejection': SamplingMethod.REJECTION_SAMPLING,
            'beam': SamplingMethod.BEAM_SEARCH,
            'mcts': SamplingMethod.MCTS,
            'zero_order': SamplingMethod.ZERO_ORDER,
            'eps_greedy': SamplingMethod.EPS_GREEDY,
        }
        if args.method not in method_map:
            raise ValueError(f"Unknown method: {args.method}")
        sampling_method = method_map[args.method]
        sampling_params = {'scorer': scorer}

        # Add master params if relevant for method
        if args.method in ['rejection', 'zero_order', 'eps_greedy', 'beam', 'mcts']:
            if args.N is not None:
                sampling_params['N'] = args.N
            if args.K is not None:
                sampling_params['K'] = args.K
            if args.lambda_ is not None:
                sampling_params['lambda_param'] = args.lambda_
            if args.eps is not None:
                sampling_params['eps'] = args.eps
            if args.B is not None:
                sampling_params['B'] = args.B
            if args.S is not None:
                sampling_params['S'] = args.S

        outname = args.output or f"edm_{args.method}_{args.scorer}.png"
        generate_image_grid(
            network_pkl,
            outname,
            latents,
            class_labels,
            seed=args.seed,
            gridw=gridw,
            gridh=gridh,
            device=device,
            num_steps=num_steps,
            S_churn=40,
            S_min=0.05,
            S_max=50,
            S_noise=1.003,
            sampling_method=sampling_method,
            sampling_params=sampling_params,
        )
        print(f"\n[EDM] Saved: {outname}\n")

if __name__ == '__main__':
    main()