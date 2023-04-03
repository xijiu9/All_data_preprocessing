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
import torch.multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", message="your warning message here")

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

    p_args = parser.parse_args()

    with open("../Extract_Relation/Triple_list_after_2020.json", "r") as f:
        Triple_list = json.load(f)
    # with open("Summarize/Object_list.json", "r") as f:
    #     Object_list = json.load(f)

    mp.set_start_method('spawn')

    rank_dict = {}
    model_dict = {}

    for triple in Triple_list:

        min_time, max_time = triple["extract relation pairs"]["min time"], triple["extract relation pairs"]["max time"]

        min_time, max_time = "2020/01/01 00:00", "2022/12/31 23:59"  # NEW
        intermediate_months = get_months(min_time, max_time, p_args.train_mode)

        triple_order_dict = {}

        data_short = triple["data_short"]

        if data_short not in rank_dict.keys():
            rank_dict[data_short] = {}

        first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
        Triple_feature = first_relation["sub"][0] + ' & ' + first_relation["obj"][0]
        with open(f"../Extract_Relation/Co_occurance/{data_short}/{Triple_feature}/Co_occur.json", "r") as f:
            Co_occur_list = json.load(f)

        for month in intermediate_months:
            # if month not in rank_dict[data_short].keys():
            if True:  # NEW
                triple_order_dict[f"{month}"] = {}
                model_path = f"../../SHENG/result/{month}/pretrain/"

                if month not in model_dict.keys():
                    print(f'load model {data_short} at month {month}, triple feature {Triple_feature}')
                    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                    lm_model = AutoModelWithLMHead.from_pretrained(model_path)
                    if torch.cuda.is_available():
                        lm_model = lm_model.cuda()

                    model_dict[month] = [tokenizer, lm_model]
                else:
                    print(f'Have model {data_short} at month {month}, triple feature {Triple_feature}')
                    tokenizer, lm_model = model_dict[month]

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

                triple_order_dict[f"{month}"][first_relation["obj"][0]] = {}

                for i in trange(0, len(Co_occur_list), p_args.batch_size):
                    sub_batch = Co_occur_list[i:i + p_args.batch_size]
                    for _ in range(p_args.batch_size - len(sub_batch)):
                        sub_batch.append(sub_batch[-1])

                    probe_texts = [first_relation["obj"][0] for _ in range(p_args.batch_size)]

                    first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
                    texts = [triple["relation_prompt"].replace("[X]", sub_) for sub_ in sub_batch]

                    all_preds_probs = decoder.decode(texts, probe_texts=probe_texts)  # topk predictions

                    for idx, preds_probs in enumerate(all_preds_probs):
                        triple_order_dict[f"{month}"][first_relation["obj"][0]][sub_batch[idx]] = float(preds_probs[1])

                rank_dict[data_short][month] = triple_order_dict[month]

            os.makedirs(f"Co_occur_result/{p_args.train_mode}/{data_short}/{Triple_feature}", exist_ok=True)
            with open(f"Co_occur_result/{p_args.train_mode}/{data_short}/{Triple_feature}/Rank_dict.json", "w") as f:
                json.dump(triple_order_dict, f, indent=4)

            with open(f"Co_occur_result/{p_args.train_mode}/{data_short}/{Triple_feature}/Triple_save.json", "w") as f:
                json.dump(triple, f, indent=4)  # NEW
            # time.sleep(2)


if __name__ == '__main__':
    main()
