import sys
import torch
import torch.nn as nn
import random
import numpy as np
from pathlib import Path
from train import parse_args, SimpleTrainer2d
from ifmorph.model import from_pth
from copy import deepcopy
from ifmorph.util import warp_points
from ifmorph.diff_operators import jacobian

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
    #image_path = Path(args.dataset) / '002_03.jpg'
    
    trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, iterations=args.iterations, model_name=args.model_name, args=args, model_path=args.model_path)

    if args.model_path is None:
        psnr, ms_ssim, training_time, eval_time, eval_fps = trainer.train()
        print("PSNR:{:.4f}, MS-SSIM:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(psnr, ms_ssim, training_time, eval_time, eval_fps))
    else:
        #warp_net = from_pth("/home/jpsml/ifmorph/results/001_002/weights.pth", w0=1, ww=1, device=torch.device("cuda:0"))
        #warp_net = from_pth("/home/joaopaulolima/ifmorph/results/001_002/weights.pth", w0=1, ww=1, device=torch.device("cuda:0"))
        warp_net = from_pth("/home/vitor/Documents/doc/GaussianImage/warping_models/weights.pth", w0=1, ww=1, device=torch.device("cuda:0"))
        warped_trainer = deepcopy(trainer)
        t1 = 0
        t2 = 1
        n_frames = 101
        times = np.arange(t1, t2, (t2 - t1) / n_frames)
        for t in times:
            warp_net = warp_net.eval()
            wpoints, coords = warp_points(warp_net, trainer.gaussian_model.get_xyz[:, [1, 0]], t)
            jac = jacobian(wpoints.unsqueeze(0), coords)[0].squeeze(0)
            s = torch.linalg.norm(jac, dim=1)[:, [1, 0]]
            u = jac[:, :, 0]
            theta = torch.atan2(u[:, 1], u[:, 0]).unsqueeze(-1)
            warped_trainer.gaussian_model._xyz = nn.Parameter(torch.atanh(wpoints[:, [1, 0]]))
            #warped_trainer.gaussian_model.scaling_inc = s
            #warped_trainer.gaussian_model.rotation_inc = theta
            warped_trainer.gaussian_model.scaling_inc = torch.ones_like(s)
            warped_trainer.gaussian_model.rotation_inc = torch.zeros_like(theta)

            warped_trainer.test(warped_trainer.image_name + "_fitting_{:.2f}.png".format(t))
            torch.save(warped_trainer.gaussian_model.state_dict(), warped_trainer.log_dir / "gaussian_model_{:.2f}.pth.tar".format(t))

if __name__ == "__main__":
    main(sys.argv[1:])