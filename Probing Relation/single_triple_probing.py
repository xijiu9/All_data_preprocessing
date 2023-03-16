import torch
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead
)

from decoder import Decoder
import argparse
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def print_predictions(sentence, preds_probs):
    k = min(len(preds_probs),100)
    # print(f"Top {k} predictions")
    print("-------------------------")
    print(f"Rank\tProb\tPred")
    print("-------------------------")
    for i in range(k):
        preds_prob = preds_probs[i]
        print(f"{i+1}\t{preds_prob[1]:.10f}\t{preds_prob[0]}")

    print("-------------------------")
    # print("\n")
    print("Top1 prediction sentence:")
    print(f"\"{sentence.replace('[Y]',preds_probs[0][0])}\"")

def get_months(start_date_str, end_date_str):
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
            month_str = "m_" + current_month.strftime("%Y_%-m")
            valid_months.append(month_str)
        current_month += timedelta(days=30)

    return valid_months

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", required=True)
    parser.add_argument("--init_method", choices=['independent','order'], default='order')
    parser.add_argument("--iter_method", choices=['none','order','confidence'], default='none')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    args = parser.parse_args()

    with open("../Extract Relation/Triple_list_after_2020.json", "r") as f:
        Triple_list = json.load(f)
    with open("Summarize/Object_list.json", "r") as f:
        Object_list = json.load(f)

    for triple in Triple_list:
        min_time, max_time = triple["extract relation pairs"]["min time"], triple["extract relation pairs"]["max time"]
        intermediate_months = get_months(min_time, max_time)

        triple_order_dict = {}

        for month in intermediate_months:
            triple_order_dict[f"{month}"] = {}
            model_path = f"../../SHENG/result/{month}/pretrain/"
            print(f'load model {model_path}')
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            lm_model = AutoModelWithLMHead.from_pretrained(model_path)
            if torch.cuda.is_available():
                lm_model = lm_model.cuda()

            # make sure this is only an evaluation
            lm_model.eval()
            for param in lm_model.parameters():
                param.grad = None

            decoder = Decoder(
                model=lm_model,
                tokenizer=tokenizer,
                init_method=args.init_method,
                iter_method=args.iter_method,
                MAX_ITER=args.max_iter,
                BEAM_SIZE=args.beam_size,
                verbose=False
            )


            for probe_text in tqdm(Object_list):
                first_relation = triple["extract relation pairs"][list(triple["extract relation pairs"].keys())[0]]
                text = triple["relation_prompt"].replace("[x]", first_relation["sub"])

                all_preds_probs = decoder.decode(text, probe_text=probe_text) # topk predictions
                preds_probs = all_preds_probs[0][0]

                triple_order_dict[f"{month}"][preds_probs[0]] = preds_probs[1]

        break



if __name__ == '__main__':
    main()
