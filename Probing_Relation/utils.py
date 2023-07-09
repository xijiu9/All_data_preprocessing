import re
import string
import collections

# https://github.com/huggingface/transformers/blob/758ed3332b219dd3529a1d3639fa30aa4954e0f3/src/transformers/data/metrics/squad_metrics.py
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def convert_2d_list_to_1d(l):
    return [j for sub in l for j in sub]

def convert_1d_list_to_2d(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]