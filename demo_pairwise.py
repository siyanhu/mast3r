                elif rt_count >= demo_frames_limit[0] and rt_count < demo_frames_limit[1]:
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images

# visualize a few matches
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl

import sys
import time
import re
sys.path.append('/home/siyanhu/Gits/mast3r')
import root_file_io as fio


def sort_file_paths(file_paths):
    def extract_number(file_path):
        # Extract the file name from the path
        file_name = file_path.split('/')[-1]
        # Extract the number from the file name
        match = re.search(r'\d+', file_name)
        if match:
            return int(match.group())
        return 0  # Return 0 if no number is found

    return sorted(file_paths, key=extract_number)


def current_timestamp(micro_second=False):
    t = time.time()
    if micro_second:
        return int(t * 1000 * 1000)
    else:
        return int(t * 1000)


def pairwise(model, img_pth0, img_pth1, check_vis=True, save_dir=''):
    (im0dir, im0name, im0ext) = fio.get_filename_components(img_pth0)
    (im1dir, im1name, im1ext) = fio.get_filename_components(img_pth1)
    images = load_images([img_pth0, img_pth1], size=512)

    start_time = current_timestamp(micro_second=False)

    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    end_time = current_timestamp(micro_second=False)

    n_viz = 20
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)

    if len(matches_im0) < 1:
        return -1, im0name, im1name

    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
    image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

    viz_imgs = []
    for i, view in enumerate([view1, view2]):
        rgb_tensor = view['img'] * image_std + image_mean
        viz_imgs.append(rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
    img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    # pl.show(block=True)
    if check_vis == False and fio.file_exist(save_dir):
        save_path = fio.createPath(fio.sep, [save_dir], '_'.join([im0name, im1name]) + '.png')
        pl.savefig(save_path)
    # else:
    #     pl.show(block=True)
    pl.close()
    return float(end_time - start_time), im0name, im1name


if __name__ == '__main__':

    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    demo_frames_limit = [20, -1]

    db_tag = '7scenes'
    combo = 'train_full_byorder_85/images'
    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'data', db_tag])
    scene_dirs = fio.traverse_dir(data_dir, full_path=True, towards_sub=False)
    scene_dirs = fio.filter_if_dir(scene_dirs, filter_out_target=False)
    image_paths = []

    model_name = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    for scn_dth_raw in scene_dirs:
        (scndir, scnname, scnext) = fio.get_filename_components(scn_dth_raw)
        scn_dth = fio.createPath(fio.sep, [scn_dth_raw, combo])
        seq_dirs = fio.traverse_dir(scn_dth, full_path=True, towards_sub=False)
        seq_dirs = fio.filter_if_dir(seq_dirs, filter_out_target=False)

        pw_ts_save_path = fio.createPath(fio.sep, [fio.getParentDir(), 'outputs', 'pairwise', db_tag, scnname], 
                                         'ts_log' + '-'.join([str(demo_frames_limit[0]), str(demo_frames_limit[1])]) + '.txt')
        
        rt_count = 0
        useful_count = 0
        rt_ts_diff_sum = 0.0
        for seq_dth in seq_dirs:
            (seqdir, seqname, seqext) = fio.get_filename_components(seq_dth)
            full_pths = fio.traverse_dir(seq_dth, full_path=True, towards_sub=False)
            img_pths_raw = fio.filter_folder(full_pths, filter_out=False, filter_text='color')
            img_pths_raw = fio.filter_ext(img_pths_raw, filter_out_target=False, ext_set=fio.img_ext_set)
            img_pths = sort_file_paths(img_pths_raw)

            pw_save_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'outputs', 'pairwise', db_tag, scnname, seqname])
            fio.ensure_dir(pw_save_dir)

            for i in range (len(img_pths) - 1):
                if rt_count < demo_frames_limit[0]:
                    ts_diff, im0name, im1name = pairwise(model, img_pths[i], img_pths[i + 1], check_vis=False, save_dir='')
                    rt_count += 1
                else:
                    if (demo_frames_limit[1] == -1) or (rt_count < demo_frames_limit[1]):
                        ts_diff, im0name, im1name = pairwise(model, img_pths[i], img_pths[i + 1], check_vis=False, save_dir='')
                        if ts_diff == -1:
                            continue
                        rt_ts_diff_sum += ts_diff
                        content = ','.join([seqname, im0name, im1name, str(ts_diff)]) + '\n'
                        # print(content)
                        rt_count += 1
                        useful_count += 1
                        fio.save_str_to_txt(content, pw_ts_save_path, mode='a+')
                    else:
                        break

        fio.save_str_to_txt(str(rt_ts_diff_sum/useful_count), pw_ts_save_path, mode='a+')