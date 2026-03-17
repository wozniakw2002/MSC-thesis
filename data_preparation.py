import kagglehub
import shutil
import os
from pathlib import Path


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