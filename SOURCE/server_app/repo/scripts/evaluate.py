import cv2
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim


def evaluate_repair(restored_dir, gt_dir, mask_dir=None):
    """评测修复质量：PSNR, SSIM"""
    restored_dir = Path(restored_dir)
    gt_dir = Path(gt_dir)
    
    results = []
    for gt_path in sorted(gt_dir.glob('*.png')):
        rest_path = restored_dir / gt_path.name
        if not rest_path.exists():
            continue
        
        gt = cv2.imread(str(gt_path))
        restored = cv2.imread(str(rest_path))
        
        # 确保尺寸一致
        if gt.shape != restored.shape:
            restored = cv2.resize(restored, (gt.shape[1], gt.shape[0]))
        
        psnr_val = compute_psnr(gt, restored)
        ssim_val = compute_ssim(gt, restored, channel_axis=2)
        
        result = {'file': gt_path.name, 'psnr': psnr_val, 'ssim': ssim_val}
        
        # 如果有mask，计算mask内外分别的指标
        if mask_dir:
            mask_path = Path(mask_dir) / gt_path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask_binary = mask > 127
                # mask内区域的PSNR
                if np.any(mask_binary):
                    gt_masked = gt.copy()
                    rest_masked = restored.copy()
                    gt_masked[~mask_binary] = 0
                    rest_masked[~mask_binary] = 0
                    result['psnr_in_mask'] = compute_psnr(gt_masked, rest_masked)
        
        results.append(result)
    
    # 汇总
    if results:
        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        print(f"Average PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--restored', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--mask', default=None)
    args = parser.parse_args()
    
    evaluate_repair(args.restored, args.gt, args.mask)
