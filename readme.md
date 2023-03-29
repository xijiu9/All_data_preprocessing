### Divide_Into_Months:

#### process_pubmed.py
    use the original .offset file, get the PMID, title, abstract and save them into
    ./data/pubmed_abstract.json

#### refine_the_PMID.py
    Use together with Fetch_Date_Info. 

    Use ./data/pubmed_abstract.json, find the article of English and the right PMID. Discard
    those who are Chinese, French, etc. Discard the article with wrong PMID.
    
    Results are saved in ./data/refined_pubmed_abstract.json and ./data/discard_pubmed_abstract.json
#### divide_into_month.py

    Use data/refined_pubmed_abstract.json and Fetch_Date_Info, divide the title and abstract into
    year, month, quater, and half of a year.


### Extract Relation

#### save_and_load_triple.py
    From BioLAMA and MedLAMA get the info of triples and save them into ./Triple_result

    We seperately deal with each type of relation, and save a Triple_list.json, a Object_list.json containing all
    the possible Object in this relation, and Subject_list.json containing all the possible Subject in this relation.

#### search_for_triples.py
    Use ./Divide_Into_Months/data/refined_pubmed_abstract.json and ./Triple_result.

    We divide the Object and Subject into single words and search for the lower case in pubmed articles. 
    We save the pmid of each single word into a dict, and store them in ./Extract Result/e_result.




### Fetch_Date_Info

#### fetch_date.py
    From web fetch the time info, title, and other information and save them in ./data_map, 
    each group have 5000 pmid.


### Probing Relation