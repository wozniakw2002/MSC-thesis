import kagglehub
import shutil
import os
from pathlib import Path
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch


def get_data_from_Kaggle(dataset, destination_path):
    path = kagglehub.dataset_download(dataset)
    shutil.copytree(path, destination_path, dirs_exist_ok = True)
    print("Dataset copied successfully.")
    shutil.rmtree(path)
    print("Cache deleted.")

def divide_into_img_gt(path, img_name = 'images', gt_name = 'matrices', img_ext = '.jpg',
                       gt_ext = '.mat'):
    images = os.path.join(path, img_name)
    gt = os.path.join(path, gt_name)

    os.makedirs(images, exist_ok=True)
    os.makedirs(gt, exist_ok=True)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isfile(file_path):
            if file.endswith(img_ext):
                shutil.move(file_path, os.path.join(images, file))
            elif file.endswith(gt_ext):
                shutil.move(file_path, os.path.join(gt, file))

def concat_folders(path, folders_name = 'images_part', out_name = 'images'):
    target_folder = os.path.join(path, out_name)
    os.makedirs(target_folder, exist_ok=True)

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)

        if os.path.isdir(folder_path) and folder.startswith(folders_name):

            for file in os.listdir(folder_path):
                src = os.path.join(folder_path, file)
                dst = os.path.join(target_folder, file)

                shutil.move(src, dst)
            os.rmdir(folder_path)


def rename_folder(path, name):
    folder = Path(path)
    folder.rename(folder.parent / name)

def rename_files_in_folder(folder, prefix='ann_'):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if os.path.isfile(path):
            os.rename(path, os.path.join(folder, prefix + file))

def rename_files_sequentialy(path, folder='ground-truth', prefix='ann_'):
    dir_list = os.listdir(path)
    if folder not in dir_list:
        for dir in dir_list:
            new_path = os.path.join(path, dir)
            if os.path.isdir(new_path):
                rename_files_sequentialy(new_path,folder,prefix)
    else:
        new_path = os.path.join(path,folder)
        rename_files_in_folder(new_path, prefix)

def rename_folders_sequentialy(path, sufix = '_data', new_name = None):
    dir_list = os.listdir(path)
    for dir in dir_list:
        new_path = os.path.join(path,dir)
        if dir.endswith(sufix):
            if new_name == None:
                new_name = dir.removesuffix(sufix)
            rename_folder(new_path, new_name)
        elif os.path.isdir(new_path):
            rename_folders_sequentialy(new_path, sufix, new_name)

def flatten_folders(path):
    parent_name = os.path.basename(path)
    subfolders = [f for f in os.listdir(path) 
                if os.path.isdir(os.path.join(path, f))]

    for sub in subfolders:
        sub_path = os.path.join(path, sub)
        new_folder = f"data\{parent_name}_{sub}"
        shutil.move(sub_path, new_folder)
    shutil.rmtree(path)

def split_test_val_test(path):
    images_folder = os.path.join(path, "images")
    matrices_folder = os.path.join(path, "matrices")
    txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]

    for txt_file in txt_files:
        txt_path = os.path.join(path, txt_file)
        folder_name = os.path.splitext(txt_file)[0]
        target_folder = os.path.join(path, folder_name)
        os.makedirs(os.path.join(target_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_folder, "matrices"), exist_ok=True)
        numbers = []
        with open(txt_path, "r") as f:
            for line in f:
                numbers.append(line.strip().split()[0])
        
        for file in os.listdir(images_folder):
            if any(num in file for num in numbers):
                shutil.copy2(os.path.join(images_folder, file),
                            os.path.join(target_folder, "images", file))
        for file in os.listdir(matrices_folder):
            if any(num in file for num in numbers):
                shutil.copy2(os.path.join(matrices_folder, file),
                            os.path.join(target_folder, "matrices", file))
    shutil.rmtree(images_folder)
    shutil.rmtree(matrices_folder)



def create_density_map(image, points, leaf_size = 64, k=5, beta = 0.3):
    h, w = image.shape[:2]
    density = torch.zeros((h, w), dtype=torch.float32)

    if len(points) == 0:
        return density

    pts = np.array(points)
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leaf_size)
    distances, _ = tree.query(pts, k=min(k+1, len(points)))

    for i, (x, y) in enumerate(pts):
        x, y = int(x), int(y)

        if len(points) > 1:
            sigma = beta * np.mean(distances[i][1:])
        else:
            sigma = 10

        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1

        grid = torch.arange(size).float() - size // 2
        x_grid, y_grid = torch.meshgrid(grid, grid, indexing='ij')
        kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
        kernel /= kernel.sum()

        x1 = max(0, x - size // 2)
        y1 = max(0, y - size // 2)
        x2 = min(w, x + size // 2 + 1)
        y2 = min(h, y + size // 2 + 1)

        kx1 = max(0, size // 2 - x)
        ky1 = max(0, size // 2 - y)
        kx2 = kx1 + (x2 - x1)
        ky2 = ky1 + (y2 - y1)
        if (x2 > x1) and (y2 > y1) and (kx2 > kx1) and (ky2 > ky1):
            density[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2]
        else:
            print(f"Skipped invalid slice at point {(x, y)} with sigma={sigma}")

    return density


def create_density_maps_in_folders(path, ann_first = True, mat_format = '.mat', is_shanghai = False):
    dir_list = os.listdir(path)
    if 'matrices' not in dir_list:
        for dir in dir_list:
            new_path = os.path.join(path, dir)
            if os.path.isdir(new_path):
                create_density_maps_in_folders(new_path, ann_first, mat_format, is_shanghai)
    else:
        images_path = os.path.join(path,'images')
        matrices_path = os.path.join(path,'matrices')
        maps_path = os.path.join(path,'maps')
        os.makedirs(maps_path, exist_ok=True)
        images_list = os.listdir(images_path)
        maps_list = os.listdir(maps_path)
        for image in images_list:
            image_path = os.path.join(images_path, image)
            image_view = plt.imread(image_path)
            matrix_file_name = image.replace('.jpg', '').split('.')
            matrix_file_name.insert(0, 'ann_') if ann_first else matrix_file_name.insert(1, '_ann')
            if is_shanghai:
                matrix_file_name.insert(1, 'GT_')
            matrix_file_name = ''.join(matrix_file_name)
            if image.replace('.jpg', '.npy') in maps_list:
                continue
            print(matrix_file_name)
            matrix_path = os.path.join(matrices_path, matrix_file_name)
            if mat_format == '.mat':
                if is_shanghai:
                    points = scipy.io.loadmat(matrix_path)['image_info'][0,0][0,0][0]
                else:
                    points = scipy.io.loadmat(matrix_path)['annPoints']
            else:
                points = []
                print(matrix_path)
                matrix_path = matrix_path + '.txt'
                print(matrix_path)
                with open(matrix_path, "r") as f:
                    for line in f:
                        numbers = np.int32(line.split())
                        points.append(numbers[:2])

                points = np.array(points)
            density_map = create_density_map(image_view, points)
            map_name = image.split('.')[0]
            map_path = os.path.join(maps_path, map_name)
            np.save(map_path, density_map)