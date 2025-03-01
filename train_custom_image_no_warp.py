import sys
import torch
import random
import numpy as np
from pathlib import Path
from train import parse_args, SimpleTrainer2d

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    image_path = Path(args.dataset) / '002_03.jpg'
    
    trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)

    if args.model_path is None:
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        print("PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(psnr, ms_ssim, training_time, eval_time, eval_fps))
    else:
        psnr, ms_ssim = trainer.test()
        print("PSNR:{:.4f}, MS-SSIM:{:.4f}".format(psnr, ms_ssim))

    print("this is the save path", args.model_path)

if __name__ == "__main__":
    main(sys.argv[1:])