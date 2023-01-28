import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
import os

def train(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []
    lr_path=os.path.join(args.images_dir,'Train/LR/')
    hr_path=os.path.join(args.images_dir,'Train/HR/')
    for image_path in sorted(glob.glob('{}/*'.format(hr_path))):
        hr = pil_image.open(image_path).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        hr=hr.swapaxes(1,2)
        hr=hr.swapaxes(0,1)
        hr_patches.append(hr)

    hr_patches = np.array(hr_patches)
    
    for image_path in sorted(glob.glob('{}/*'.format(lr_path))):
        lr = pil_image.open(image_path).convert('RGB')
        lr = np.array(lr).astype(np.float32)
        lr=lr.swapaxes(1,2)
        lr=lr.swapaxes(0,1)
        lr_patches.append(lr)
    
    lr_patches = np.array(lr_patches)
    print(lr_patches.shape)
    print(hr_patches.shape)
        
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    lr_path=os.path.join(args.images_dir,'Eval/LR/')
    hr_path=os.path.join(args.images_dir,'Eval/HR/')
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(hr_path)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr = np.array(hr).astype(np.float32)
        hr=hr.swapaxes(1,2)
        hr=hr.swapaxes(0,1)
        hr_group.create_dataset(str(i), data=hr)
    
    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(lr_path)))):
        lr = pil_image.open(image_path).convert('RGB')
        lr = np.array(lr).astype(np.float32)
        lr=lr.swapaxes(1,2)
        lr=lr.swapaxes(0,1)
        lr_group.create_dataset(str(i), data=lr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='data')
    parser.add_argument('--output-path', type=str, default='SROP_method/img_datasets/img_datasets.h5')
    parser.add_argument('--patch-size', type=int, default=120)
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)