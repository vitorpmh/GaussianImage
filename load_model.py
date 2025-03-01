
from train import  parse_args,SimpleTrainer2d
from ifmorph.diff_operators import jacobian
import torch.nn as nn
from ifmorph.util import warp_points
from ifmorph.model import from_pth
from copy import deepcopy
from tqdm import tqdm
from gaussianimage_rs import GaussianImage_RS
from gsplat import project_gaussians_2d_scale_rot
from gsplat.project_gaussians_2d import project_gaussians_2d
import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from gsplat.rasterize_sum import rasterize_gaussians_sum
from pygame_interface import pygame_interface
from train import image_path_to_tensor
from utils import *
from tqdm import tqdm
import random
import torchvision.transforms as transforms
from gaussianimage_cholesky import GaussianImage_Cholesky
from einops import rearrange
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

image_path = '/home/vitor/Documents/doc/GaussianImage/dataset/neutral_front/001_03.jpg'
#model_path = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_Cholesky_50000_30000/001_03/gaussian_model.pth.tar'
model_path = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/gaussian_model.pth.tar'
device = torch.device("cuda:0")
num_points = 30_000
BLOCK_H, BLOCK_W = 16, 16
gt_image = image_path_to_tensor(image_path).to(device)
H, W = gt_image.shape[2], gt_image.shape[3]


# gaussian_model = GaussianImage_Cholesky(
#     loss_type="L2", 
#     opt_type="adan", 
#     num_points=num_points, 
#     H=H, 
#     W=W, 
#     BLOCK_H=BLOCK_H, 
#     BLOCK_W=BLOCK_W, 
#     device=device, 
#     lr=1e-3, 
#     quantize=False).to(device)


# gaussian_model = GaussianImage_RS(
#     loss_type="L2", 
#     opt_type="adan", 
#     num_points=num_points, 
#     H=H, 
#     W=W, 
#     BLOCK_H=BLOCK_H, 
#     BLOCK_W=BLOCK_W, 
#     device=device, 
#     lr=1e-3, 
#     quantize=False).to(device)


# print(f"loading model path:{model_path}")
# checkpoint = torch.load(model_path, map_location=device)
# model_dict = gaussian_model.state_dict()
# pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# gaussian_model.load_state_dict(model_dict)


# gaussian_model.eval()
# with torch.no_grad():
#     out = gaussian_model()
#     out_rgb = rearrange(out['render'], '1 c h w -> h w c').cpu().numpy()
    


    # ### ploting the first gaussian:
    # mu = gaussian_model.get_xyz[0].detach().cpu().numpy()

    # L_11,L_21,L_22  = gaussian_model.get_cholesky_elements[0].detach().cpu().numpy()
    # L = np.array([[L_11, 0], [L_21, L_22]])

    # Sigma = L @ L.T

    # x = np.linspace(-5, 10, 100)
    # y = np.linspace(-5, 10, 100)
    # X, Y = np.meshgrid(x, y)

    # Z = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    # inv_Sigma = np.linalg.inv(Sigma)
    # Z_exp = np.einsum('...i,ij,...j->...', Z, inv_Sigma, Z)
    # Z_pdf = np.exp(-0.5 * Z_exp) / (2 * np.pi * np.sqrt(np.linalg.det(Sigma)))

    # plt.contourf(X, Y, Z_pdf, levels=50, cmap='viridis')
    # plt.colorbar(label='Probability Density')
    # plt.title('2D Gaussian Distribution')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
        

def plot_conics_plotly():
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
        gaussian_model.get_xyz, 
        gaussian_model.get_cholesky_elements, 
        gaussian_model.H, 
        gaussian_model.W, 
        gaussian_model.tile_bounds)

    H = gaussian_model.H  
    W = gaussian_model.W  


    fig = go.Figure()
    min_ = gaussian_model.get_features.min()
    max_ = gaussian_model.get_features.max()
    rgb = torch.floor(((gaussian_model.get_features-min_)/(max_-min_)) * 255).detach().cpu().numpy().astype(np.uint8)

    amount_to_plot = 30_000

    for i, (xy, radius, conic, rgb) in tqdm(enumerate(zip(xys, radii, conics,rgb)), total=amount_to_plot):
        if i>amount_to_plot:break

        # Get center (x, y) and conic parameters [A, B, C]
        mu_x, mu_y = xy.detach().cpu().numpy()  # Gaussian center
        A, B, C = conic.detach().cpu().numpy()  # Conic parameters

        # Compute the semi-major and semi-minor axes using the conic equation
        # The eigenvalues of the matrix defined by [A, B, C] give the axes lengths of the ellipse.
        # We first construct the conic matrix.
        conic_matrix = np.array([[A, B], [B, C]])
        
        # Eigen decomposition to get the axes of the ellipse
        eigvals, eigvecs = np.linalg.eig(conic_matrix)

        # Eigenvalues are the inverses of the squared axes lengths (semi-major and semi-minor)
        semi_major = np.sqrt(1 / eigvals[0])  # Larger axis
        semi_minor = np.sqrt(1 / eigvals[1])  # Smaller axis

        # Ellipse orientation (rotation angle)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        # Create points for the ellipse
        theta = np.linspace(0, 2 * np.pi, 100)
        x_ellipse = semi_major * np.cos(theta)
        y_ellipse = semi_minor * np.sin(theta)

        # Rotate the ellipse based on the orientation angle
        x_rot = x_ellipse * np.cos(angle) - y_ellipse * np.sin(angle)
        y_rot = x_ellipse * np.sin(angle) + y_ellipse * np.cos(angle)

        # Translate the ellipse to its center (mu_x, mu_y)
        x_final = mu_x + x_rot
        y_final = mu_y + y_rot

        
        r,g,b = rgb
        fig.add_trace(go.Scatter(
            x=x_final,
            y=y_final,
            mode='lines',
            fill='togaussian_model', 
            fillcolor=f'rgba({r}, {g}, {b}, 1)',  
            line=dict(color=f'rgba({r}, {g}, {b}, 1)', width=2),
            name=f'Gaussian {i+1}',  
        ))

    fig.update_layout(
        title="2D Gaussian Projections on HxW Grid",
        xaxis_title="X",
        yaxis_title="Y",
        showlegend=True,
        width=1200,
        height=900,
        xaxis=dict(range=[0, W]),  
        yaxis=dict(
            range=[0, H],  
            autorange='reversed'  
        ), 
    )

    # Show the interactive plot
    fig.show()

from errno import EEXIST
from os import makedirs, path
import os
from plyfile import PlyData, PlyElement

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def save_ply(path):
    # mkdir_p(os.path.dirname(path))
    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            gaussian_model.get_xyz, 
            gaussian_model.get_scaling, 
            gaussian_model.get_rotation,
            gaussian_model.H, 
            gaussian_model.W, 
            gaussian_model.tile_bounds)
    

    xyz = xys.detach().cpu().numpy()
    xyz = np.concat([xyz,np.zeros((xys.shape[0],1))],axis=-1)
    normals = np.zeros((xys.shape[0],45)) #+ 1e-3
    f_dc = gaussian_model.get_features.detach().contiguous().cpu().numpy() #+ 1e-8 #.transpose(1, 2).flatten(start_dim=1)
    # f_rest = _features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = np.zeros_like(f_dc) #+ 1
    opacities = gaussian_model.get_opacity.detach().cpu().numpy()
    scale = gaussian_model.get_scaling.detach().cpu().numpy() #/ 2
    rotation = gaussian_model.get_rotation.detach().cpu().numpy()


    # testing...
    scale_add = np.zeros((xys.shape[0],1)) #+ 1e-8
    scale = np.concat([scale,scale_add],axis=-1)

    R = np.array([
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ], dtype=np.float32) for theta in rotation.flatten()
    ], dtype=np.float32)
    q = np.zeros((R.shape[0], 4))
    #(30000,1)
    # Compute the trace of the matrix
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Small epsilon value to avoid division by zero
    epsilon = 1e-8

    # Quaternion conversion
    for i, mat in enumerate(R):
        trace = mat[0, 0] + mat[1, 1] + mat[2, 2]
        
        if trace > epsilon:  # Check if the trace is sufficiently large
            q_w = 0.5 * np.sqrt(1 + trace)
            q_x = (mat[2, 1] - mat[1, 2]) / (4 * q_w)
            q_y = (mat[0, 2] - mat[2, 0]) / (4 * q_w)
            q_z = (mat[1, 0] - mat[0, 1]) / (4 * q_w)
            q[i] = [q_w, q_x, q_y, q_z]
        else:
            # Special case when the trace is near zero (possible 180-degree rotation)
            # You can handle this by using a different formula, like from the matrix diagonal
            q_w = np.sqrt(max(0.5 * (1 + mat[0, 0] + mat[1, 1] + mat[2, 2]),0))
            q_x = np.sqrt(max(0.5 * (1 + mat[0, 0] - mat[1, 1] - mat[2, 2]),0)) if q_w != 0 else 0
            q_y = np.sqrt(max(0.5 * (1 - mat[0, 0] + mat[1, 1] - mat[2, 2]),0)) if q_w != 0 else 0
            q_z = np.sqrt(max(0.5 * (1 - mat[0, 0] - mat[1, 1] + mat[2, 2]),0)) if q_w != 0 else 0
            q[i] = [q_w, q_x, q_y, q_z]


    rotation = q + 1e-8
    # def construct_list_of_attributes():
    #     l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    #     # All channels except the 3 DC
    #     for i in range(gaussian_model.get_features.shape[0]*gaussian_model.get_features.shape[1]):
    #         l.append('f_dc_{}'.format(i))
    #     for i in range(np.zeros_like(f_dc).shape[0]*np.zeros_like(f_dc).shape[1]):
    #         l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(gaussian_model.get_scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(gaussian_model.get_rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     return l
    # dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]



    dtype_full = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('nx', 'f4'),
        ('ny', 'f4'),
        ('nz', 'f4'),
        ('f_dc_r', 'f4'),
        ('f_dc_g', 'f4'),
        ('f_dc_b', 'f4'),
        ('f_rest_0','f4'),
        ('f_rest_1','f4'),
        ('f_rest_2','f4'),
        ('f_rest_3','f4'),
        ('f_rest_4','f4'),
        ('f_rest_5','f4'),
        ('f_rest_6','f4'),
        ('f_rest_7','f4'),
        ('f_rest_8','f4'),
        ('f_rest_9','f4'),
        ('f_rest_10','f4'),
        ('f_rest_11','f4'),
        ('f_rest_12','f4'),
        ('f_rest_13','f4'),
        ('f_rest_14','f4'),
        ('f_rest_15','f4'),
        ('f_rest_16','f4'),
        ('f_rest_17','f4'),
        ('f_rest_18','f4'),
        ('f_rest_19','f4'),
        ('f_rest_20','f4'),
        ('f_rest_21','f4'),
        ('f_rest_22','f4'),
        ('f_rest_23','f4'),
        ('f_rest_24','f4'),
        ('f_rest_25','f4'),
        ('f_rest_26','f4'),
        ('f_rest_27','f4'),
        ('f_rest_28','f4'),
        ('f_rest_29','f4'),
        ('f_rest_30','f4'),
        ('f_rest_31','f4'),
        ('f_rest_32','f4'),
        ('f_rest_33','f4'),
        ('f_rest_34','f4'),
        ('f_rest_35','f4'),
        ('f_rest_36','f4'),
        ('f_rest_37','f4'),
        ('f_rest_38','f4'),
        ('f_rest_39','f4'),
        ('f_rest_40','f4'),
        ('f_rest_41','f4'),
        ('f_rest_42','f4'),
        ('f_rest_43','f4'),
        ('f_rest_44','f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'),
        ('scale_1', 'f4'),
        ('scale_2', 'f4'),
        ('rot_1', 'f4'),
        ('rot_2', 'f4'),
        ('rot_3', 'f4'),
        ('rot_4', 'f4'),
    ] #* xyz.shape[0]
    elements = np.empty((xyz.shape[0],len(dtype_full)), dtype=dtype_full)
    # Concatenate all attributes
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)

    # Reshape the attributes array so that each row corresponds to the 16 attributes per vertex
    attributes = attributes.reshape(-1, len(dtype_full))

    # Now, elements can be directly assigned from attributes as tuples
    elements = np.empty(attributes.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def plot_conics_matplotlib(
    gaussian_model, 
    crop = False,
    square = None,
    plot_original = True,
    original_path = None,
    amount_to_plot = 1000 , 
    scaling_factor=2,
    filepath="gaussian_projections.png",
    figsize = (1440,960),
    dpi = 300):

    assert crop is True and square is not None


    xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            gaussian_model.get_xyz, 
            gaussian_model.get_scaling * gaussian_model.scaling_inc * scaling_factor, 
            gaussian_model.get_rotation * gaussian_model.scaling_inc, 
            gaussian_model.H, 
            gaussian_model.W, 
            gaussian_model.tile_bounds)
    if plot_original:
        original_image_path = original_path + f'morph_0.{int(filepath.split("_")[-1][:-4]):02d}.png'
    

    os.makedirs('/'.join(filepath.split('/')[:-1]), exist_ok=True)

    # Set up the figure
    if plot_original:
        fig, ax = plt.subplots(2,1,figsize=(figsize[0] / dpi, figsize[1] / dpi), dpi=dpi)
        original_image = plt.imread(original_image_path)
        if crop:
            x_min, y_min = square[0]  # top-left
            x_max, y_max = square[1]  # bottom-right
            original_image = original_image[y_min:y_max, x_min:x_max, :]

        plt.subplot(2,1,1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
    else:
        fig, ax = plt.subplots(1,1,figsize=(figsize[0] / 100, figsize[1] / 100), dpi=300)
        if crop:
            x_min, y_min = square[0]  # top-left
            x_max, y_max = square[1]  # bottom-right
    


    with torch.no_grad():
        pix_coord = torch.stack(torch.meshgrid(torch.arange(gaussian_model.W), torch.arange(gaussian_model.H), indexing='xy'), dim=-1).to(xys.device)
        

        if crop:
            selected_gaussians = (
                ((xys[:,1] > y_min) * (xys[:,1] < y_max)) * 
                ((xys[:,0] > x_min) * (xys[:,0] < x_max))
            )
            out_img = rasterize_gaussians_sum_torch(
                xys[selected_gaussians][:amount_to_plot], 
                radii[selected_gaussians][:amount_to_plot], 
                conics[selected_gaussians][:amount_to_plot], 
                gaussian_model.get_features[selected_gaussians][:amount_to_plot], 
                gaussian_model.get_opacity[selected_gaussians][:amount_to_plot], 
                depths[selected_gaussians][:amount_to_plot], 
                gaussian_model.H, 
                gaussian_model.W,
                pix_coord )
            out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
            out_img = out_img * 255

            out_img = out_img[y_min:y_max, x_min:x_max, :]
        else:
            out_img = rasterize_gaussians_sum_torch(
                xys[:amount_to_plot], 
                radii[:amount_to_plot], 
                conics[:amount_to_plot], 
                gaussian_model.get_features[:amount_to_plot], 
                gaussian_model.get_opacity[:amount_to_plot], 
                depths[:amount_to_plot], 
                gaussian_model.H, 
                gaussian_model.W,
                pix_coord )
            out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
            out_img = out_img * 255
        if plot_original:
            plt.subplot(2,1,2)
        plt.imshow(out_img.detach().cpu().numpy().astype(np.uint8))
        plt.title('Rasterizer')
        plt.axis('off')
    
    
    
    
    try:
        for a_ in ax:
            a_.axis('off')
            a_.set_aspect(1)
    except:
        ax.axis('off')

    fig.tight_layout()
    plt.subplots_adjust(left=00, right=1, top=1, bottom=0)
    plt.savefig(filepath, bbox_inches='tight')
        
    plt.close(fig)  # Close the figure to free memory


    
def main(argv):
    args = parse_args(argv)

    img_path = "/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/001_03_fitting_0.99.png"
    img_path = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/morph_0.61.png'
    selected_indices = pygame_interface(img_path)


    square = selected_indices


    save = '/home/vitor/Documents/doc/GaussianImage/videos/'
    os.makedirs(save, exist_ok=True)

    image_path_0 = '/home/vitor/Documents/doc/GaussianImage/dataset/neutral_front/001_03.jpg'
    image_path_1 = '/home/vitor/Documents/doc/GaussianImage/dataset/neutral_front/002_03.jpg'

    model_path_0 = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/gaussian_model.pth.tar'
    model_path_1 = '/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/002_03/gaussian_model.pth.tar'

    trainer_0 = SimpleTrainer2d(image_path=image_path_0, num_points=30000, iterations=50000, model_name='GaussianImage_RS', args=args, model_path=model_path_0)
    trainer_1 = SimpleTrainer2d(image_path=image_path_1, num_points=30000, iterations=50000, model_name='GaussianImage_RS', args=args, model_path=model_path_1)
        

    warp_net = from_pth("/home/vitor/Documents/doc/GaussianImage/warping_models/weights.pth", w0=1, ww=1, device=device)
    morphed_trainer = deepcopy(trainer_0)

    t1 = 0
    t2 = 1
    n_frames = 100
    times = np.arange(t1, t2, (t2 - t1) / n_frames)

    for idx,t in enumerate(times):
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
        #morphed_trainer.rotation_inc = nn.Parameter(torch.cat((rotation_inc_0, rotation_inc_1)))
        #morphed_trainer.scaling_inc = nn.Parameter(torch.cat((torch.ones_like(scaling_inc_0), torch.ones_like(scaling_inc_1))))
        morphed_trainer.gaussian_model.rotation_inc = nn.Parameter(torch.cat((torch.zeros_like(rotation_inc_0), torch.zeros_like(rotation_inc_1))))

        
        
        
        # plot_conics_matplotlib(
        #     morphed_trainer.gaussian_model,
        #     crop=args.crop,
        #     square=square,
        #     scaling_factor= args.scaling_factor,
        #     amount_to_plot=args.n,
        #     filepath=save+f'/{args.n}_{args.output_for}/'+f'test_{idx}.png',
        #     plot_original = True,
        #     original_path='/home/vitor/Documents/doc/GaussianImage/checkpoints/frll_neutral_front/GaussianImage_RS_50000_30000/001_03/',
        #     figsize = (1080,1920),
        #     dpi = 300)


    generate_video(save,args.n,args.output_for,output_video_path=args.out_video_path)

    ##  python load_model.py --n 4000 --output_for "top_hair" --plot_original True --scaling_factor 0.5 --crop True --out_video_path top_hair.mp4

if __name__ == "__main__":
    main(sys.argv[1:])
    # args = parse_args(sys.argv[1:])
    # save = '/home/vitor/Documents/doc/GaussianImage/videos/'
    # generate_video(save,args.n,args.output_for,output_video_path=args.out_video_path)