import json

def count_common_elements(list1, list2):
    count = 0
    for element in list1:
        if element in list2:
            count += 1
    return count

time_list_const = ['q_2020_1', 'q_2020_2', 'q_2020_3', 'q_2020_4', 'q_2021_1', 'q_2021_2', 'q_2021_3', 'q_2021_4', 'q_2022_1', 'q_2022_2', 'q_2022_3', 'q_2022_4']

for time_before in time_list_const:
    with open(f"{time_before}_pmids.json", "r") as f:
        refined_pmid = json.load(f)

    with open(f"../../Divide_Into_Months/pmids_in_{time_before[:-1]}0{time_before[-1]}.json", "r") as f:
        new_pmid = json.load(f)

    overlap_number = count_common_elements(new_pmid, refined_pmid)
    print(f"{overlap_number} / {len(new_pmid)} in {time_before}")