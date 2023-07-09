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
import numpy as np
import torch

warnings.filterwarnings("ignore", message="your warning message here")
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

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


def main(time_feature='q_2022_1'):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", required=True)
    parser.add_argument("--init_method", choices=['independent', 'order'], default='order')
    parser.add_argument("--iter_method", choices=['none', 'order', 'confidence'], default='none')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--train_mode', type=str, required=True, default='m', choices=['m', 'q', 'h'],
                        help='Divide training data into every month, 3 months or 6 months')

    p_args = parser.parse_args()

    with open(f"../Get_Indices/zero_indices_{time_feature}_only_1000000.json", "r") as f:
        zero_indices = json.load(f)
    with open(f"../Get_Indices/positive_indices_{time_feature}.json", "r") as f:
        nonzero_indices = json.load(f)

    sample_size = 200000

    np.random.seed(p_args.seed)
    torch.manual_seed(p_args.seed)

    # 打乱 nonzero_indices 数组顺序
    np.random.shuffle(nonzero_indices)
    np.random.shuffle(zero_indices)

    # 从打乱后的数组中选择百分之一的样本
    random_sample_1 = nonzero_indices[:sample_size]
    random_sample_2 = zero_indices[:sample_size]
    random_sample = random_sample_1 + random_sample_2
    np.random.shuffle(random_sample)
    print(f"seed {p_args.seed}, ", random_sample[0:10])

    # 根据样本的第一位构建字典
    result_dict = {}
    for index in random_sample:
        key = index[0]
        if key in result_dict:
            result_dict[key].append(index[1])
        else:
            result_dict[key] = [index[1]]

    probing_result = {}

    for month in [time_feature]:
        model_path = f"../../../SHENG/result/{month}/pretrain/"

        print(f'load model at month {month}')
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        config = AutoConfig.from_pretrained(model_path, use_fast=False)
        lm_model = AutoModelWithLMHead.from_pretrained(model_path, config=config)
        if torch.cuda.is_available():
            lm_model = lm_model.cuda()

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

        # universal_prompt = "[X] [Y]."
        universal_prompt = "[X] has relation with [Y]."
        # universal_prompt = "[X] has very strong relation with [Y]."
        for idx, (sub_id, obj_list) in tqdm(enumerate(result_dict.items()), total=len(result_dict)):
            sub_text = subject_map[str(sub_id)]
            text = universal_prompt.replace("[X]", sub_text)
            probing_result[sub_id] = {}

            for i in range(0, len(obj_list), p_args.batch_size):
                batch = obj_list[i:i + p_args.batch_size]
                probe_texts = []

                for probe_id in batch:
                    probe_text = object_map[str(probe_id)]
                    probe_texts.append(probe_text)

                for _ in range(p_args.batch_size - len(probe_texts)):
                    probe_texts.append(probe_texts[-1])

                # import IPython
                # IPython.embed()

                all_preds_probs = decoder.decode([text for _ in probe_texts],
                                                 probe_texts=probe_texts)  # topk predictions

                for (obj, prob), actual_obj in zip(all_preds_probs, probe_texts):
                    probing_result[sub_id][object_inverse_map[actual_obj]] = float(prob)

            if idx % 2000 == 100:
                os.makedirs(f"control_probing_matrix_{time_feature}", exist_ok=True)
                with open(f"control_probing_matrix_{time_feature}/LM_learn.json", "w") as f:
                    json.dump(probing_result, f, indent=4)
                    # print(probing_result)
                print(f"save result at time {idx}")
        with open(f"control_probing_matrix_{time_feature}/LM_learn.json", "w") as f:
            json.dump(probing_result, f, indent=4)
            # print(probing_result)
        print(f"save result at time {idx}")
    # import IPython
    # IPython.embed()

if __name__ == '__main__':
    with open('../Cooccurance_Matrix/subject_map.json', 'r') as file:
        subject_map = json.load(file)
    with open('../Cooccurance_Matrix/subject_inverse_map.json', 'r') as file:
        subject_inverse_map = json.load(file)
    with open('../Cooccurance_Matrix/object_map.json', 'r') as file:
        object_map = json.load(file)
    with open('../Cooccurance_Matrix/object_inverse_map.json', 'r') as file:
        object_inverse_map = json.load(file)

    main(time_feature='q_2022_4')
