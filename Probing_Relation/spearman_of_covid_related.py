import torch
from transformers import (
    AutoConfig,
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
import torch.multiprocessing as mp
import warnings
import random

warnings.filterwarnings("ignore", message="your warning message here")
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(1)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def print_predictions(sentence, preds_probs):
    k = min(len(preds_probs), 100)
    # print(f"Top {k} predictions")
    print("-------------------------")
    print(f"Rank\tProb\tPred")
    print("-------------------------")
    for i in range(k):
        preds_prob = preds_probs[i]
        print(f"{i + 1}\t{preds_prob[1]:.10f}\t{preds_prob[0]}")

    print("-------------------------")
    # print("\n")
    print("Top1 prediction sentence:")
    print(f"\"{sentence.replace('[Y]', preds_probs[0][0])}\"")


# def get_months(start_date_str, end_date_str, train_mode=''):
#     if train_mode == 'm':
#         date_format = "%Y/%m/%d %H:%M"
#         start_date = datetime.strptime(start_date_str, date_format)
#         end_date = datetime.strptime(end_date_str, date_format)
#
#         # Set boundaries for valid months
#         min_month = datetime(year=2020, month=1, day=1)
#         max_month = datetime(year=2023, month=2, day=28)
#
#         # Calculate five months before and after the given dates
#         start_month = start_date - timedelta(days=30 * 5)
#         end_month = end_date + timedelta(days=30 * 5)
#
#         # Create list of valid months
#         valid_months = []
#         current_month = start_month
#         while current_month <= end_month:
#             if current_month >= min_month and current_month <= max_month:
#                 month_str = "m_" + current_month.strftime("%Y_%-m")
#                 valid_months.append(month_str)
#             current_month += timedelta(days=30)
#
#     return valid_months

def get_months(start_date_str, end_date_str, train_mode=''):
    if train_mode == 'm':
        date_format = "%Y/%m/%d %H:%M"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        # Set boundaries for valid months
        min_month = datetime(year=2020, month=1, day=1)
        max_month = datetime(year=2023, month=2, day=28)

        # Calculate five months before and after the given dates
        start_month = start_date - timedelta(days=30 * 5)
        end_month = end_date + timedelta(days=30 * 5)

        # Create list of valid months
        valid_months = []
        current_month = start_month
        while current_month <= end_month:
            if current_month >= min_month and current_month <= max_month:
                month_str = "m_{}_{}".format(current_month.strftime("%Y"), int(current_month.strftime("%m")))
                valid_months.append(month_str)
            current_month += timedelta(days=30)

        return valid_months

    elif train_mode == 'q':
        date_format = "%Y/%m/%d %H:%M"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        # Set boundaries for valid quarters
        min_quarter = datetime(year=2020, month=1, day=1)
        max_quarter = datetime(year=2023, month=1, day=1)

        # Calculate two quarters before and after the given dates
        start_quarter = start_date - timedelta(days=365 / 4 * 2)
        end_quarter = end_date + timedelta(days=365 / 4 * 2)

        # Create list of valid quarters
        valid_quarters = []
        current_quarter = start_quarter
        while current_quarter <= end_quarter:
            if current_quarter >= min_quarter and current_quarter < max_quarter:
                quarter_str = "q_{}_{}".format(current_quarter.strftime("%Y"), (current_quarter.month - 1) // 3 + 1)
                valid_quarters.append(quarter_str)
            current_quarter += timedelta(days=365 // 4)

        return valid_quarters

    elif train_mode == 'h':
        date_format = "%Y/%m/%d %H:%M"
        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        # Set boundaries for valid half years
        min_half_year = datetime(year=2020, month=1, day=1)
        max_half_year = datetime(year=2023, month=1, day=1)

        # Calculate one half year before and after the given dates
        start_half_year = start_date - timedelta(days=365 / 2)
        end_half_year = end_date + timedelta(days=365 / 2)

        # Create list of valid half years
        valid_half_years = []
        current_half_year = start_half_year
        while current_half_year <= end_half_year:
            if current_half_year >= min_half_year and current_half_year < max_half_year:
                half_year_str = "h_{}_{}".format(current_half_year.strftime("%Y"),
                                                 1 if current_half_year.month <= 6 else 2)
                valid_half_years.append(half_year_str)
            current_half_year += timedelta(days=365 // 2)

        return valid_half_years

def shuffled_string(my_list, my_string, my_sample_num):
    # 使用相同的种子值对列表进行打乱
    random.seed(hash(my_string))
    random.shuffle(my_list)
    return my_list[:my_sample_num]

def sort_dict(dictionary):
    sorted_dict = {}
    for key in sorted(dictionary.keys()):
        value = dictionary[key]
        if isinstance(value, dict):
            sorted_dict[key] = sort_dict(value)
        elif isinstance(value, list):
            sorted_dict[key] = sorted(value)
        else:
            sorted_dict[key] = value
    return sorted_dict

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", required=True)
    parser.add_argument("--init_method", choices=['independent', 'order'], default='order')
    parser.add_argument("--iter_method", choices=['none', 'order', 'confidence'], default='none')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--train_mode', type=str, required=True, default='m', choices=['m', 'q', 'h'],
                        help='Divide training data into every month, 3 months or 6 months')

    data_origin = "P2176"
    long_or_short = "short"
    p_args = parser.parse_args()

    with open(f"../Extract_Relation/COVID_related/{data_origin}_triple.json", "r") as f:
        Triple = json.load(f)
        Triple_list = [Triple]
    with open("../Divide_Into_Months/pmids_in_q_2020_01.json", "r") as f:
        pmid_list = json.load(f)

    mp.set_start_method('spawn')

    model_dict = {}

    overlap_object = {}

    for idx, triple in enumerate(Triple_list):

        # if list(triple["extract relation pairs"].keys())[0] not in pmid_list:
        #     continue

        min_time, max_time = "2020/01/01 00:00", "2022/12/31 23:59"  # NEW
        intermediate_months = get_months(min_time, max_time, p_args.train_mode)

        triple_order_dict_subject = {}
        triple_order_dict_object = {}

        data_short = triple["data_short"]
        print(f"data short is {data_short}")
        # if '2' not in data_short and 'ctd' not in data_short:
        #     pass
        # else:
        #     continue

        first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
        Triple_feature = first_relation["sub"][0] + ' & ' + first_relation["obj"][0]

        print(Triple_feature)

        sample_size = 0

        with open(f"../Extract_Relation/COVID_related/{data_origin}_{long_or_short}_version.json", "r") as f:
            covid_Object_list = json.load(f)
        with open(f"../Extract_Relation/COVID_related/{data_origin}_until_q_2021_2_object.json", "r") as f:
            covid_cooccur_list = json.load(f)
        with open(f"../Extract_Relation/COVID_related/{data_origin}_most_common_object.json", "r") as f:
            common_Object_list = json.load(f)

        # IMPORTANT
        # sample_Object_list = covid_Object_list + common_Object_list[:len(covid_Object_list)]
        # sample_Object_list = covid_Object_list
        sample_Object_list = covid_cooccur_list
        # sample_Object_list = common_Object_list[:len(covid_Object_list)]

        for month in ["q_2019_4"] + intermediate_months:
        # for month in ["q_2021_2"]:
        # for month in ["q_2021_4/unlearn_rho_0.1_alpha_0.2",]:
        # for month in ["debug_descent_first_multi_triple_3000_steps_seed_42_sgd_momentum_no_nestrove_adamW_ascent_1_descent_1_step"]:
        # for month in ["debug_descent_first_single_triple_300_steps_seed_42_sgd_momentum_no_nestrove_adamW_ascent_1_descent_1_step_q_2022_1"]:
        # for month in ["q_2022_1_ascent_SGD_3"]:
        # for month in ["GSAM_q_2022_2_only_10_epoch_rho_0.2_alpha_0.4"]:
        #     model_path = f"../../SHENG/after_EMNLP_result/{month}/"
        #     model_path = f"../../SHENG/result/{month}/pretrain"
        # for month in ["q_2021_2", "q_2021_3", "q_2021_4"]:
        #     model_path = f"../../SHENG/BioBERTresult_line_by_line/{month}/pretrain/"
        #     model_path = f"../../SHENG/Retrain_BioBERT/{month}/pretrain/"
        #     model_path = f"../../SHENG/Retrain_BioBERT_unlearn/{month}/"
            model_path = f"../../SHENG/Only_Use_COVID/{month}/pretrain/"
        #     model_path = f"../../SHENG/Only_Use_COVID/{month}/unlearn_from_q_2021_2_to_q_2019_4_rho_0.05_alpha_0.4/"
            # subdirs = sorted(
            #     [d for d in os.listdir(model_path) if  # and 20000 > int(d.split("_")[-1]) > 10000  and "epoch" in d
            #      os.path.isdir(os.path.join(model_path, d)) and int(d.split("_")[-1]) < 10000],
            #     key=lambda x: int(x.split("_")[-1]))
            # print("subdir is ", subdirs)
            # for model_step in subdirs:
            # print(subdirs)
            # for model_step in subdirs:
            # for model_step in ["epoch_0", "epoch_1", "epoch_2"]:
            for model_step in ["no_use"]:
                print(model_step)

                if (month + "/" + model_step) not in model_dict.keys():
                    print(
                        f'load model {data_short} at month {month} and step {model_step}, triple feature {Triple_feature}')
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    config = AutoConfig.from_pretrained(model_path, use_fast=False)
                    lm_model = AutoModelWithLMHead.from_pretrained(model_path, config=config)
                    # lm_model = AutoModelWithLMHead.from_pretrained(model_path + model_step, config=config)

                    # IMPORTANT
                    # if torch.cuda.is_available():
                    #     lm_model = lm_model.cuda()

                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if info.used < 12 * 1024 * 1024 * 1024:
                        model_dict[f"{month}/{model_step}"] = [tokenizer, lm_model]
                    else:
                        print("CUDA OUT OF MEMORY")
                else:
                    print(
                        f'Have model {data_short} at month {month} and step {model_step}, triple feature {Triple_feature}')
                    tokenizer, lm_model = model_dict[f"{month}/{model_step}"]

                # make sure this is only an evaluation
                lm_model.eval()
                for param in lm_model.parameters():
                    param.grad = None

                decoder = Decoder(
                    model=lm_model,
                    tokenizer=tokenizer,
                    init_method=p_args.init_method,
                    iter_method=p_args.iter_method,
                    MAX_ITER=p_args.max_iter,
                    BEAM_SIZE=p_args.beam_size,
                    verbose=False,
                    batch_size=p_args.batch_size
                )

                # Subject probing
                triple_order_dict_subject[f"{month}/{model_step}"] = {}

                for i in range(0, len(sample_Object_list), p_args.batch_size):
                    batch = sample_Object_list[i:i + p_args.batch_size]
                    probe_texts = []
                    for probe_text in batch:
                        probe_texts.append(probe_text.lower())

                    # if probe_text.lower() != first_relation["obj"][0] or "/" in probe_text:  # NEW
                    #     continue
                    for _ in range(p_args.batch_size - len(probe_texts)):
                        probe_texts.append(batch[-1])

                    first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
                    text = triple["relation_prompt"].replace("[X]", "covid-19") # IMPORTANT
                    # text = "[X] has relation with [Y].".replace("[X]", first_relation["sub"][0])

                    print(text,probe_texts)
                    all_preds_probs = decoder.decode([text for _ in probe_texts],
                                                     probe_texts=probe_texts)  # topk predictions

                    # print(all_preds_probs)
                    for idx, preds_probs in enumerate(all_preds_probs):
                        triple_order_dict_subject[f"{month}/{model_step}"][probe_texts[idx]] = float(preds_probs[1])

                triple_order_dict_subject[f"{month}/{model_step}"] = dict(
                    sorted(triple_order_dict_subject[f"{month}/{model_step}"].items(), key=lambda item: item[1]))

            os.makedirs(f"Rank_result/{p_args.train_mode}/{data_short}/{Triple_feature}", exist_ok=True)
            triple_order_dict_subject = sort_dict(triple_order_dict_subject)

            with open(f"Rank_result/{p_args.train_mode}/{data_short}/{Triple_feature}/Rank_dict_Subject.json", "w") as f:
                json.dump(triple_order_dict_subject, f, indent=4)

            with open(f"Rank_result/{p_args.train_mode}/{data_short}/{Triple_feature}/Triple_save.json", "w") as f:
                json.dump(triple, f, indent=4)



if __name__ == '__main__':
    main()
