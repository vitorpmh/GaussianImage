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

    image_path_0 = Path(args.dataset) / '001_03.jpg'
    image_path_1 = Path(args.dataset) / '002_03.jpg'

    model_path_0 = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/gaussian_model.pth.tar'
    model_path_1 = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/002_03/gaussian_model.pth.tar'
    
    trainer_0 = SimpleTrainer2d(image_path=image_path_0, num_points=args.num_points, iterations=args.iterations, model_name=args.model_name, args=args, model_path=model_path_0)
    trainer_1 = SimpleTrainer2d(image_path=image_path_1, num_points=args.num_points, iterations=args.iterations, model_name=args.model_name, args=args, model_path=model_path_1)
    
    #warp_net = from_pth("/home/jpsml/ifmorph/results/001_002/weights.pth", w0=1, ww=1, device=torch.device("cuda:0"))
    warp_net = from_pth("/home/vitor/Documents/doc/GaussianImage/warping_models/weights.pth", w0=1, ww=1, device=torch.device("cuda:0"))

    morphed_trainer = deepcopy(trainer_0)

    t1 = 0
    t2 = 1
    n_frames = 101
    times = np.arange(t1, t2, (t2 - t1) / n_frames)

    for t in times:
        warp_net = warp_net.eval()

        wpoints, coords = warp_points(warp_net, trainer_0.gaussian_model.get_xyz[:, [1, 0]], t)
        jac = jacobian(wpoints.unsqueeze(0), coords)[0].squeeze(0)
        s = torch.linalg.norm(jac, dim=1)[:, [1, 0]]
        #s, _ = s.max(dim=1, keepdim=True)
        s = s.mean(dim=1, keepdim=True)
        u = jac[:, :, 0]
        theta = torch.atan2(u[:, 1], u[:, 0]).unsqueeze(-1)
        xyz_0 = torch.atanh(wpoints[:, [1, 0]])
        opacity_0 = trainer_0.gaussian_model.get_opacity * (1 - t)
        scaling_inc_0 = s
        rotation_inc_0 = theta

        wpoints, coords = warp_points(warp_net, trainer_1.gaussian_model.get_xyz[:, [1, 0]], t - 1)
        jac = jacobian(wpoints.unsqueeze(0), coords)[0].squeeze(0)
        s = torch.linalg.norm(jac, dim=1)[:, [1, 0]]
        #s, _ = s.max(dim=1, keepdim=True)
        s = s.mean(dim=1, keepdim=True)
        u = jac[:, :, 0]
        theta = torch.atan2(u[:, 1], u[:, 0]).unsqueeze(-1)
        xyz_1 = torch.atanh(wpoints[:, [1, 0]])
        opacity_1 = trainer_1.gaussian_model.get_opacity * t
        scaling_inc_1 = s
        rotation_inc_1 = theta

        morphed_trainer.gaussian_model._xyz = nn.Parameter(torch.cat((xyz_0, xyz_1)))
        morphed_trainer.gaussian_model._opacity = torch.cat((opacity_0, opacity_1))
        morphed_trainer.gaussian_model._scaling = nn.Parameter(torch.cat((trainer_0.gaussian_model._scaling, trainer_1.gaussian_model._scaling)))
        morphed_trainer.gaussian_model._rotation = nn.Parameter(torch.cat((trainer_0.gaussian_model._rotation, trainer_1.gaussian_model._rotation)))
        morphed_trainer.gaussian_model._features_dc = nn.Parameter(torch.cat((trainer_0.gaussian_model._features_dc, trainer_1.gaussian_model._features_dc)))
        morphed_trainer.gaussian_model.scaling_inc = nn.Parameter(torch.cat((scaling_inc_0, scaling_inc_1)))
        #morphed_trainer.gaussian_model.rotation_inc = nn.Parameter(torch.cat((rotation_inc_0, rotation_inc_1)))
        #morphed_trainer.gaussian_model.scaling_inc = nn.Parameter(torch.cat((torch.ones_like(scaling_inc_0), torch.ones_like(scaling_inc_1))))
        morphed_trainer.gaussian_model.rotation_inc = nn.Parameter(torch.cat((torch.zeros_like(rotation_inc_0), torch.zeros_like(rotation_inc_1))))

        morphed_trainer.test("morph_{:.2f}.png".format(t))

if __name__ == "__main__":
    main(sys.argv[1:])