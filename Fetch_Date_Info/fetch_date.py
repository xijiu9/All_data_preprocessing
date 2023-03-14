import random

from Bio import Entrez
from datetime import datetime
import re
import time
import json
import multiprocessing
import os

def fetch_pubmed_records(pubmed_ids):
    Entrez.email = "your.email@domain.com"
    id_list = ",".join(pubmed_ids)
    handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
    return handle.read().strip().split("\n\n")

def get_paper_release_times(records):
    release_times = {}
    for record in records:
        medline_dict = dict([re.split(r'\s*-\s*', line.strip(), maxsplit=1) for line in record.splitlines() if line.strip() and "-" in line])

        if "EDAT" in medline_dict:
            release_times[medline_dict["PMID"]] = {"time": medline_dict["EDAT"] if len(medline_dict["EDAT"]) >= len(medline_dict["CRDT"]) \
                                                                        else medline_dict["CRDT"],
                                                   "title broken": medline_dict["TI"] if "TI" in medline_dict.keys() else medline_dict["BTI"],
                                                   "full": medline_dict}
        else:
            print("wrong for {}".format(records))
            return None
        # print(medline_dict)
        # exit(0)
    return release_times

# 30000000 + i * step_size, 30000000 + (i + 1) * step_size
def download(start, end):
    try:
        # pubmed_ids = [str(end - 1)]
        pubmed_ids = [str(x) for x in range(start, end)]
        records = fetch_pubmed_records(pubmed_ids)
        release_times = get_paper_release_times(records)

        if release_times:
            print(time.asctime(time.localtime()), len(release_times.keys()),
                  "add map, {:0>8} - {:0>8}".format(min(release_times.keys()), max(release_times.keys())))
            # with open('data_map/{:0>8}-{:0>8}.json'.format(start, end), 'r') as f:
            #     file = json.load(f)
            # file["{}".format(end - 1)] = release_times["{}".format(end - 1)]
            # with open('data_map/{:0>8}-{:0>8}.json'.format(start, end), 'w') as f:
            #     json.dump(file, f)
            with open('data_map/{:0>8}-{:0>8}.json'.format(start, end), 'w') as f:
                json.dump(release_times, f)
    except:
        print("!" * 100, "invalid {:0>8} - {:0>8}".format(start, end))

# download(30009916, 30009917)
# cnt = 0
# for filelist in sorted(os.listdir("data_map")):
#     start, end = int(filelist[:8]), int(filelist[9:17])
#     download(start, end)

max_article = 37000000
step_size = 5000
proc = {}
proc_num = 10

Success, Fail = 0, 1
while Fail > 0:
    Start, End = 0, step_size
    Success, Fail = 0, 0
    while End < max_article:
        if not os.path.exists("data_map/{:0>8}-{:0>8}.json".format(Start, End)):
            Fail += 1
        else:
            Success += 1

        Start += step_size
        End += step_size
    print("{} Success, {} Fail, {} in total".format(Success, Fail, max_article // step_size))

    Start, End = 0, step_size
    while End < max_article:
        cur_num_proc = 0
        while cur_num_proc < proc_num:
            if not os.path.exists("data_map/{:0>8}-{:0>8}.json".format(Start, End)):
                print("Try to load {:0>8}-{:0>8}".format(Start, End))
                proc[cur_num_proc] = multiprocessing.Process(target=download, args=(Start, End))
                cur_num_proc += 1

                # print(proc_num)

            Start += step_size
            End += step_size

        time.sleep(2)

        for i in range(proc_num):
            proc[i].start()

        for i in range(proc_num):
            proc[i].join()


