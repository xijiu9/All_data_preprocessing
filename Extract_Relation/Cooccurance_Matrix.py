import json
import time
import numpy as np

from tqdm import tqdm, trange

# # # # 读取ALL_Object_dict.json
with open("/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Exact_intersection/ALL_Object_dict.json") as f:
    Object_dict = json.load(f)

# 读取ALL_Subject_dict.json
with open("/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Extract_Relation/Words_PMID/Exact_intersection/ALL_Subject_dict.json") as f:
    Subject_dict = json.load(f)
#
# import IPython
# IPython.embed()
#
# print("finish loading")
# print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")
#
Object_dict_filtered = {key: value for key, value in Object_dict.items() if len(value) > 0}
Subject_dict_filtered = {key: value for key, value in Subject_dict.items() if len(value) > 0}

# print(f"after filter Object {len(Object_dict_filtered)}, Subject {len(Subject_dict_filtered)}")
# # 获取ALL_Object_dict和ALL_Subject_dict的大小
# object_size = len(Object_dict_filtered)
# subject_size = len(Subject_dict_filtered)
# #
# print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")
# #
# PMID_dict = {str(pmid): {"sub": [], "obj": []} for pmid in range(37000000)}
# #
# subject_map, subject_inverse_map, object_map, object_inverse_map = {}, {}, {}, {}
#
# for i, sub in tqdm(enumerate(Subject_dict_filtered.keys()), total=len(Subject_dict_filtered)):
#     subject_map[i] = sub
#     subject_inverse_map[sub] = i
#     for pmid in Subject_dict_filtered[sub]:
#         PMID_dict[pmid]["sub"].append(i)

# for j, obj in tqdm(enumerate(Object_dict_filtered.keys()), total=len(Object_dict_filtered)):
#     object_map[j] = obj
#     object_inverse_map[obj] = j
#     for pmid in Object_dict_filtered[obj]:
#         PMID_dict[pmid]["obj"].append(j)

# import IPython
# IPython.embed()

# Subject_lists = [lst for sublist in Subject_dict.values() for lst in sublist]
# Subject_listss = set(Subject_lists)
# Object_lists = [lst for sublist in Object_dict.values() for lst in sublist]
# Object_listss = set(Object_lists)
#
# print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")

# Subject_intersection = []
# for key, value in tqdm(Subject_dict.items()):
#     if set(value).intersection(Object_listss):
#         Subject_intersection.append([key, len(set(value).intersection(Object_listss))])
#
# Object_intersection = []
# for key, value in tqdm(Object_dict.items()):
#     if set(value).intersection(Subject_listss):
#         Object_intersection.append([key, len(set(value).intersection(Subject_listss))])

# import IPython
# IPython.embed()

# def time_convert(time):
#     year, month, day = int(time[0:4]), int(time[5:7]), int(time[8:10])
#
#     if year < 2020:
#         return "q_2019"
#     elif year > 2022:
#         return "q_2023"
#     else:
#         return f"q_{year}_{(month - 1) // 3 + 1}"

# all_time_map = {}
# all_time_list = []
# for year in [2020, 2021, 2022]:
#     for month in [1, 2, 3, 4]:
#         time_feature = f"q_{year}_{month}"
#         all_time_map[time_feature] = []
#         all_time_list.append(time_feature)
# all_time_map["q_2019"] = []
# all_time_map["q_2023"] = []

# for key in trange(0, 37000000, 5000):
#     try:
#     # if True:
#         with open(
#                 f"../Fetch_Date_Info/data_map/{str(5000 * (key // 5000)).zfill(8)}-{str(5000 * (key // 5000 + 1)).zfill(8)}.json") as f:
#             time_map = json.load(f)
#             for pmid in time_map.keys():
#                 time_feature = time_convert(time_map[f"{pmid}"]["full"]["EDAT"])
#                 all_time_map[time_feature].append(pmid)
#     except:
#         print(f"can not load {str(5000 * (key // 5000)).zfill(8)}-{str(5000 * (key // 5000 + 1)).zfill(8)}.json")
#
# with open("Co_occurance/all_time_map.json", "w") as f:
#     json.dump(all_time_map, f)

with open('subject_map.json', 'r') as file:
    subject_map = json.load(file)
with open('subject_inverse_map.json', 'r') as file:
    subject_inverse_map = json.load(file)
with open('object_map.json', 'r') as file:
    object_map = json.load(file)
with open('object_inverse_map.json', 'r') as file:
    object_inverse_map = json.load(file)

with open("Co_occurance/all_time_map.json", "r") as f:
    all_time_map = json.load(f)

print("finish all time map")

with open("Co_occurance/PMID_search.json", "r") as f:
    PMID_dict = json.load(f)
# 创建一个初始值为0的矩阵

matrix_count_2021_4 = np.load(f"Co_occurance/co_occurance_matrix_q_2021_4.npy")
matrix_count_2022_1 = np.load(f"Co_occurance/co_occurance_matrix_q_2022_1.npy")

print(len(all_time_map["q_2022_1"]))
zero_len_list = []
posi_len_list = []
for pmid in tqdm(all_time_map["q_2022_1"]):
    subs, objs = PMID_dict[pmid]["sub"], PMID_dict[pmid]["obj"]
    # print(len(subs), len(objs), pmid)
    if len(subs) + len(objs) == 0:
        zero_len_list.append(pmid)
    else:
        posi_len_list.append(len(subs) + len(objs))
    # user_input = input("请输入：")

    # if "@" in user_input:
    #     for sub in subs:
    #         print(subject_map[str(sub)])
    #     print('-' * 20)
    #     for obj in objs:
    #         print(object_map[str(obj)])
    #
    #     import IPython
    #     IPython.embed()

with open("/m-ent1/ent1/xihc20/ALL_DATA_PREPROCESSING/Divide_Into_Months/pmids_in_q_2022_01.json") as f:
    pmid_2022_1 = json.load(f)

percent = [5, 10, 20, 40, 80, 160]
thres = [0.01, 0.1, 0.2, 0.5, 0.9]
print(len(zero_len_list))

# for per in percent:
#     for thr in thres:
#         cnt = 0
#         for pmid in tqdm(all_time_map["q_2022_1"]):
#             subs, objs = PMID_dict[pmid]["sub"], PMID_dict[pmid]["obj"]
#             feature = [1000000000]
#             for sub in subs:
#                  for obj in objs:
#                      feature.append(matrix_count_2021_4[sub, obj] / matrix_count_2022_1[sub, obj])
#             feature = sorted(feature)
#             # print(feature, '\n' * 5)
#             if feature[len(feature) // per] < thr:
#                     cnt += 1
#         print("all time map", per, thr, cnt)
#
#         cnt = 0
#         for pmid in tqdm(pmid_2022_1):
#             subs, objs = PMID_dict[pmid]["sub"], PMID_dict[pmid]["obj"]
#             feature = [1000000000]
#             for sub in subs:
#                  for obj in objs:
#                      feature.append(matrix_count_2021_4[sub, obj] / matrix_count_2022_1[sub, obj])
#             feature = sorted(feature)
#             # print(feature, '\n' * 5)
#             if feature[len(feature) // per] < thr:
#                 cnt += 1
#         print("pmid 2022 1", per, thr, cnt)

import IPython
IPython.embed()

matrix_count = np.zeros((subject_size, object_size))

# for pmid in tqdm(all_time_map["q_2019"]):
#     subobj = PMID_dict[pmid]
#     for sub_idx in subobj["sub"]:
#         for obj_idx in subobj["obj"]:
#             matrix_count[sub_idx, obj_idx] += 1

# np.save("Co_occurance/co_occurance_matrix_q_2019.npy", matrix_count)
# import IPython
# IPython.embed()

# for time_q in all_time_list:
#     for pmid in tqdm(all_time_map[time_q]):
#         subobj = PMID_dict[pmid]
#         for sub_idx in subobj["sub"]:
#             for obj_idx in subobj["obj"]:
#                 matrix_count[sub_idx, obj_idx] += 1
#
#     np.save(f"Co_occurance/co_occurance_matrix_{time_q}.npy", matrix_count)

# 遍历 Object_dict_filtered 的值
# for i, obj_value in tqdm(enumerate(Object_dict_filtered.values()), total=len(Object_dict_filtered)):
#     # 遍历 Subject_dict_filtered 的值
#     for j, sub_value in enumerate(Subject_dict_filtered.values()):
#         # 计算交集的大小
#         intersection_size = len(set(obj_value).intersection(set(sub_value)))
#         # 在 matrix 的 [i, j] 位置加上交集的大小
#         matrix[i, j] += intersection_size


# 保存矩阵到文件
# np.save("Co_occurance/co_occurance_matrix.npy", matrix_count)
# with open("Co_occurance/PMID_search.json", "w") as f:
#     json.dump(PMID_dict, f, indent=4)

# import IPython
# IPython.embed()
# import numpy as np
#
# # 生成随机矩阵
# matrix = np.random.randint(1, 11, size=(3000, 3000))
#
# # 保存矩阵到文件
# np.save("random_matrix.npy", matrix)
#
# import psutil
# print(psutil.Process().memory_info().rss / (1024 * 1024))

