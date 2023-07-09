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

print(f"after filter Object {len(Object_dict_filtered)}, Subject {len(Subject_dict_filtered)}")
# 获取ALL_Object_dict和ALL_Subject_dict的大小
object_size = len(Object_dict_filtered)
subject_size = len(Subject_dict_filtered)
# #
# print("The current time is:", time.strftime("%H:%M:%S", time.localtime()), "start loading Pubmed")
# #
PMID_dict = {str(pmid): {"sub": [], "obj": []} for pmid in range(37000000)}
# #
subject_map, subject_inverse_map, object_map, object_inverse_map = {}, {}, {}, {}

for i, sub in tqdm(enumerate(Subject_dict_filtered.keys()), total=len(Subject_dict_filtered)):
    subject_map[i] = sub
    subject_inverse_map[sub] = i
    for pmid in Subject_dict_filtered[sub]:
        PMID_dict[pmid]["sub"].append(i)

for j, obj in tqdm(enumerate(Object_dict_filtered.keys()), total=len(Object_dict_filtered)):
    object_map[j] = obj
    object_inverse_map[obj] = j
    for pmid in Object_dict_filtered[obj]:
        PMID_dict[pmid]["obj"].append(j)

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

def time_convert(time):
    year, month, day = int(time[0:4]), int(time[5:7]), int(time[8:10])

    if year < 2020:
        return "q_2019"
    elif year > 2022:
        return "q_2023"
    else:
        return f"q_{year}_{(month - 1) // 3 + 1}"

all_time_map = {}
all_time_list = []
for year in [2020, 2021, 2022]:
    for month in [1, 2, 3, 4]:
        time_feature = f"q_{year}_{month}"
        all_time_map[time_feature] = []
        all_time_list.append(time_feature)
all_time_map["q_2019"] = []
all_time_map["q_2023"] = []

for key in trange(0, 37000000, 5000):
    try:
    # if True:
        with open(
                f"../../Fetch_Date_Info/data_map/{str(5000 * (key // 5000)).zfill(8)}-{str(5000 * (key // 5000 + 1)).zfill(8)}.json") as f:
            time_map = json.load(f)
            for pmid in time_map.keys():
                time_feature = time_convert(time_map[f"{pmid}"]["full"]["EDAT"])
                all_time_map[time_feature].append(pmid)
    except:
        print(f"can not load {str(5000 * (key // 5000)).zfill(8)}-{str(5000 * (key // 5000 + 1)).zfill(8)}.json")

with open('subject_map.json', 'w') as file:
    json.dump(subject_map, file)

# 写入 subject_inverse_map.json
with open('subject_inverse_map.json', 'w') as file:
    json.dump(subject_inverse_map, file)

# 写入 object_map.json
with open('object_map.json', 'w') as file:
    json.dump(object_map, file)

# 写入 object_inverse_map.json
with open('object_inverse_map.json', 'w') as file:
    json.dump(object_inverse_map, file)

# 写入 all_time_map.json
with open("all_time_map.json", "w") as file:
    json.dump(all_time_map, file)

with open("PMID_search.json", "r") as f:
    json.dump(PMID_dict, f)


# with open('subject_map.json', 'r') as file:
#     subject_map = json.load(file)
# with open('subject_inverse_map.json', 'r') as file:
#     subject_inverse_map = json.load(file)
# with open('object_map.json', 'r') as file:
#     object_map = json.load(file)
# with open('object_inverse_map.json', 'r') as file:
#     object_inverse_map = json.load(file)
#
# with open("Co_occurance/all_time_map.json", "r") as f:
#     all_time_map = json.load(f)
#
# print("finish all time map")
#
# with open("Co_occurance/PMID_search.json", "r") as f:
#     PMID_dict = json.load(f)

matrix_count = np.zeros((subject_size, object_size))

for pmid in tqdm(all_time_map["q_2019"]):
    subobj = PMID_dict[pmid]
    for sub_idx in subobj["sub"]:
        for obj_idx in subobj["obj"]:
            matrix_count[sub_idx, obj_idx] += 1

np.save("co_occurance_matrix_q_2019.npy", matrix_count)
import IPython
IPython.embed()

for time_q in all_time_list:
    for pmid in tqdm(all_time_map[time_q]):
        subobj = PMID_dict[pmid]
        for sub_idx in subobj["sub"]:
            for obj_idx in subobj["obj"]:
                matrix_count[sub_idx, obj_idx] += 1

    np.save(f"co_occurance_matrix_{time_q}.npy", matrix_count)
