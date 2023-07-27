import os
from tqdm import tqdm
import time


def get_folder_info(folder_path):
    folder_info = []
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        if os.path.isdir(dir_path):
            creation_time = os.path.getctime(dir_path)
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time))
            folder_size = get_folder_size(dir_path)
            formatted_size = format_size(folder_size)
            folder_info.append((dir_path, formatted_time, formatted_size))

    sorted_info = sorted(folder_info, key=lambda x: x[1])  # 按文件夹大小排序

    with open("folder_info.txt", "w") as f:
        for info in sorted_info:
            f.write(f"Folder: {info[0]}\n")
            f.write(f"Creation Time: {info[1]}\n")
            f.write(f"Size: {info[2]}\n")
            f.write("\n")


def get_folder_size(folder_path):
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    return total_size


def format_size(size):
    if size < 1024:
        return f"{size} bytes"
    elif size < 1024 * 1024:
        return f"{size / 1024:.3f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.3f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.3f} GB"


# 指定文件夹路径
folder_path = "/homes/gws/xihc20/.cache/huggingface/datasets/text/"

# 调用函数获取文件夹信息并输出到文件
get_folder_info(folder_path)
