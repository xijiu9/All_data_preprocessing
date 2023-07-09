import torch
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead
)
import logging
import time
from decoder import Decoder
import argparse
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm, trange
import glob
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train_mode', type=str, required=True, default='m', choices=['m', 'q', 'h'],
                    help='Divide training data into every month, 3 months or 6 months')
p_args = parser.parse_args()



with open("../Extract Relation/Triple_list_after_2020.json", "r") as f:
    Triple_list = json.load(f)

Rank_dir = sorted(glob.glob(f"Rank_result/{p_args.train_mode}/*/*/*/*/Rank_dict.json"))

def flatten(lst):
    # print(11111)
    result = []
    for item in lst:
        if isinstance(item, str) and item.startswith('[') and item.endswith(']'):
            try:
                eval_lst = eval(item)
            except:
                print(item)
                eval_lst = [item[1:-1]]
                print(eval_lst)
            result.extend(eval_lst)
        elif isinstance(item, str):
            result.append(item)
        elif isinstance(item, list):
            result.extend(flatten(item))
    return result

def get_months(start_date_str, end_date_str, train_mode=''):
    if train_mode == 'm':
        date_format = "%Y/%m/%d %H:%M"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        # Set boundaries for valid months
        min_month = datetime(year=2020, month=1, day=1)
        max_month = datetime(year=2023, month=2, day=28)

        # Calculate five months before and after the given dates
        start_month = start_date
        end_month = end_date

        # Create list of valid months
        valid_months = []
        current_month = start_month
        while current_month <= end_month:
            if current_month >= min_month and current_month <= max_month:
                month_str = "m_{}_{}".format(current_month.strftime("%Y"), int(current_month.strftime("%m")))
                valid_months.append(month_str)
            current_month += timedelta(days=30)

        if len(valid_months):
            return valid_months[0]
        else:
            return None

    elif train_mode == 'q':
        date_format = "%Y/%m/%d %H:%M"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        # Set boundaries for valid quarters
        min_quarter = datetime(year=2020, month=1, day=1)
        max_quarter = datetime(year=2023, month=1, day=1)

        # Calculate two quarters before and after the given dates
        start_quarter = start_date
        end_quarter = end_date

        # Create list of valid quarters
        valid_quarters = []
        current_quarter = start_quarter
        while current_quarter <= end_quarter:
            if current_quarter >= min_quarter and current_quarter < max_quarter:
                quarter_str = "q_{}_{}".format(current_quarter.strftime("%Y"), (current_quarter.month - 1) // 3 + 1)
                valid_quarters.append(quarter_str)
            current_quarter += timedelta(days=365 / 4)

        if len(valid_quarters):
            return valid_quarters[0]
        else:
            return None

    elif train_mode == 'h':
        date_format = "%Y/%m/%d %H:%M"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        # Set boundaries for valid half years
        min_half_year = datetime(year=2020, month=1, day=1)
        max_half_year = datetime(year=2023, month=1, day=1)

        # Calculate one half year before and after the given dates
        start_half_year = start_date
        end_half_year = end_date

        # Create list of valid half years
        valid_half_years = []
        current_half_year = start_half_year
        while current_half_year <= end_half_year:
            if current_half_year >= min_half_year and current_half_year < max_half_year:
                half_year_str = "h_{}_{}".format(current_half_year.strftime("%Y"),
                                                 1 if current_half_year.month <= 6 else 2)
                valid_half_years.append(half_year_str)
            current_half_year += timedelta(days=365 / 2)

        return valid_half_years[0]

prob_up_num, prob_down_num, prob_total_num = {}, {}, {}
rank_up_num, rank_down_num, rank_total_num = {}, {}, {}

# SHENG
SHENG_list = []
# NEW
# data_map_huge_dict = {}
# Object_dict_data_short = {}
# for pmid in trange(25000000, 37000000, 5000):
#     try:
#         with open(f"../Fetch_Date_Info/data_map/{5000 * (int(pmid) // 5000)}-{5000 * (int(pmid) // 5000 + 1)}.json") as f:
#             time_map = json.load(f)
#
#             for pd in time_map.keys():
#                 data_map_huge_dict[str(pd)] = time_map[str(pd)]["time"]
#     except:
#         print("can not load {}".format(pmid))

for rank_list_dir in Rank_dir:
    print(rank_list_dir)
    with open(rank_list_dir, "r") as f:
        rank_list = json.load(f)

        for triple in Triple_list:
            first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
            Triple_feature = first_relation["sub"] + ' & ' + first_relation["obj"]

            if Triple_feature == rank_list_dir.split('/')[-2]:
                break

        sub, obj = first_relation["sub"], first_relation["obj"]
        obj_list = [obj.lower(), obj.lower().replace("-", " "), obj.lower().replace("-", " - "),
                    obj.lower().replace(".", " . "), obj.lower().replace(".", " "),
                    obj.lower().replace("'", " ' "), obj.lower().replace("'", " ")]

        obj_syn = [triple["object"]]
        if triple["object_synonym"] is not None:
            obj_syn.extend(triple["object_synonym"])
        obj_syn = flatten(obj_syn)

        obj_syn_list = []
        for obj_s in obj_syn:
            # print(obj_s)
            obj_syn_list.append(obj_s.lower())
            obj_syn_list.append(obj_s.lower().replace("-", " "))
            obj_syn_list.append(obj_s.lower().replace("-", " - "))
            obj_syn_list.append(obj.lower().replace(".", " . "))
            obj_syn_list.append(obj.lower().replace(".", " "))
            obj_syn_list.append(obj.lower().replace("'", " ' "))
            obj_syn_list.append(obj.lower().replace("'", " "))
            
        obj_syn_list = list(set(obj_syn_list))

        # NEW
        # data_short = triple["data_short"]
        # if data_short not in Object_dict_data_short.keys():
        #     with open(f"../Extract Relation/Extract Result/e_result/{data_short}/Object_dict.json") as f:
        #         Object_dict = json.load(f)
        #     Object_dict_data_short[data_short] = Object_dict
        # else:
        #     Object_dict = Object_dict_data_short[data_short]
        # print("finish pmid - object")
        # obj_pmid = sorted(Object_dict[obj.lower()])
        # obj_pmid[:] = [item for item in obj_pmid if int(item) > 25000000]
        # object_occur_month = {}
        # for pmid in tqdm(obj_pmid):
        #     object_time = data_map_huge_dict[pmid]
        #
        #     object_time = get_months(object_time, object_time, p_args.train_mode)
        #     if object_time is not None:
        #         if object_time not in object_occur_month.keys():
        #             object_occur_month[object_time] = [pmid]
        #         else:
        #             object_occur_month[object_time].append(pmid)
        #     else:
        #         pass
        #         # print()
        #         # print(data_map_huge_dict[pmid], pmid, triple["data_short"])
        # print(object_occur_month.keys())


        month_list, prob_list, idx_list = [], [], []
        for month, rank_l in rank_list.items():
            # prob_sum = 0
            # # for prob in list(rank_l.values())[:-2]:
            # for prob in list(rank_l.values()):
            #     prob_sum += prob

            prob_syn = 0
            for idx, (pred, prob) in enumerate(rank_l.items()):
                if pred in obj_list:
                # if pred in obj_syn_list:

                    if month not in month_list:
                        month_list.append(month)
                    # prob_list.append(prob / prob_sum)
                    # idx_list.append(idx)
                    prob_syn = prob_syn + prob
                    # print(pred, prob)
            if prob_syn > 0:
                prob_list.append(prob_syn)
                # prob_list.append(prob_syn / len(object_occur_month[month])) # NEW

        try:
        # if True:
            time_list = [triple["extract relation pairs"][x]["time"] for x in list(triple["extract relation pairs"].keys())[:-2]]
            time_list = [get_months(t, t, p_args.train_mode) for t in time_list]

            # print(time_list)
            # print(month_list)

            midpoint_list = [month_list.index(t) - 0.5 for t in time_list]

            for mid in midpoint_list:
                if triple["data_short"] not in prob_up_num.keys():
                    prob_up_num[triple["data_short"]] = 0
                    prob_down_num[triple["data_short"]] = 0
                    prob_total_num[triple["data_short"]] = 0

                if prob_list[int(mid - 0.5)] < prob_list[int(mid + 0.5)]:
                    prob_up_num[triple["data_short"]] += 1
                    prob_total_num[triple["data_short"]] += 1
                else:
                    prob_down_num[triple["data_short"]] += 1
                    prob_total_num[triple["data_short"]] += 1

                SHENG_list.append([sub, triple["relation_prompt"], obj, prob_list[int(mid - 0.5)], prob_list[int(mid + 0.5)]])




            # for mid in midpoint_list:
            #     if triple["data_short"] not in rank_up_num.keys():
            #         rank_up_num[triple["data_short"]] = 0
            #         rank_down_num[triple["data_short"]] = 0
            #         rank_total_num[triple["data_short"]] = 0
            #
            #     if idx_list[int(mid - 0.5)] < idx_list[int(mid + 0.5)]:
            #         rank_up_num[triple["data_short"]] += 1
            #         rank_total_num[triple["data_short"]] += 1
            #     else:
            #         rank_down_num[triple["data_short"]] += 1
            #         rank_total_num[triple["data_short"]] += 1
        #
        except:
            print("HAVE PROBLEM!")


        # Create a figure and two subplots
        fig, ax1 = plt.subplots(figsize=(len(month_list) + 3, 5))

        # Create the first plot
        color1, color2 = '#2ca02c', '#9467bd'
        ax1.plot(month_list, prob_list, color=color1)
        ax1.set_xlabel('Labels')
        ax1.set_ylabel('Values 1', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        # x_label_occur = []
        # for month in month_list:
        #     x_label_occur.append(str(len(object_occur_month[month])))
        # ax1.set_xlabel(" ".join(x_label_occur))

        # Create the second plot
        # ax2 = ax1.twinx()
        # ax2.plot(month_list, idx_list, color=color2)
        # ax2.set_ylabel('Values 2', color=color2)
        # ax2.tick_params(axis='y', labelcolor=color2)

        for mid in midpoint_list:
            plt.axvline(x=mid, linestyle='--', color='r')

        ax1.set_title(Triple_feature)

        # Show the plot
        os.makedirs("Rank_plot/{}/{}".format(p_args.train_mode, triple["data_short"]), exist_ok=True)
        plt.savefig("Rank_plot/{}/{}/{} - rank.png".format(p_args.train_mode, triple["data_short"], Triple_feature))

        plt.close()


for key in prob_up_num.keys():
    print(key, prob_up_num[key], prob_down_num[key], prob_up_num[key] + prob_down_num[key])

uu, dd = 0, 0
for u, d in zip(prob_up_num.values(), prob_down_num.values()):
    uu = uu + u
    dd = dd + d
print(uu, dd)

pp1, pp2 = [], []
with open("SHENG.txt", "w") as f:
    for x, r, y, p1, p2 in SHENG_list:
        f.write(f"{x} | {r} | {y} | {p1} | {p2} \n")
        pp1.append(p1)
        pp2.append(p2)

with open("SHENG_p1.json", "w") as f:
    json.dump(pp1, f)
with open("SHENG_p2.json", "w") as f:
    json.dump(pp2, f)

