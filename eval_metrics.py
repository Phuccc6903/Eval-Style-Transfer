from MambaST.evaluation.eval_artfid import compute_art_fid, compute_cfsd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate metrics for style transfer results")
    parser.add_argument('--target_dir', type=str, required=True, help='Path to stylized images (e.g., data/results/AdaAttN)')
    parser.add_argument('--style_dir', type=str, required=True, help='Path to style images (e.g., data/sty)')
    parser.add_argument('--content_dir', type=str, required=True, help='Path to content images (e.g., data/cnt)')

    args = parser.parse_args()    

    # tar = 'data/results/AdaAttN'
    # sty = 'data/sty'
    # cnt = 'data/cnt'
    batch_size = 1
    device = 'cuda'
    mode = 'art_fid_inf'
    content_metric = 'lpips'
    num_workers = 8
    artfid, fid, lpips, lpips_gray = compute_art_fid(args.target_dir,
                                                    args.style_dir,
                                                    args.content_dir,
                                                    batch_size,
                                                    device,
                                                    mode,
                                                    content_metric,
                                                    num_workers)

    cfsd = compute_cfsd(args.target_dir,
                        args.content_dir,
                        batch_size,
                        device,
                        num_workers)

    print('ArtFID:', artfid, 'FID:', fid, 'LPIPS:', lpips, 'LPIPS_gray:', lpips_gray)
    print('CFSD:', cfsd)