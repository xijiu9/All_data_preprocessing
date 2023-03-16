import torch
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead
)

from decoder import Decoder
import argparse
import os

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

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--text", required=True)
    parser.add_argument("--model_name_or_path", default='bert-base-uncased')
    parser.add_argument("--init_method", choices=['independent','order'], default='order')
    parser.add_argument("--iter_method", choices=['none','order','confidence'], default='none')
    parser.add_argument("--max_iter", type=int, default=10)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--probe_text", type=str, default='', required=True)
    args = parser.parse_args()

    print(f'load model {args.model_name_or_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    lm_model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path)
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

    while True:
        text = input("Please enter input (e.g., Flu has symptom such as [Y].):\n")
        if "[Y]" not in text:
            print("[Warning] Please type in the proper format.\n")
            continue

        all_preds_probs = decoder.decode(text, probe_text=args.probe_text) # topk predictions
        preds_probs = all_preds_probs[0]

        print_predictions(text, preds_probs)

        print("\n")


if __name__ == '__main__':
    main()
