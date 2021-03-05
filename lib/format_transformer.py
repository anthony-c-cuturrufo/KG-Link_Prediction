import pandas as pd
import numpy as np
'''
takes list of csvs and returns triplets of [h,r,t]. Assumes csv if formatted like:
Head  Tail  Relation1  Relation2 ...
h1    h2    1          0
h1    h3    0          1
where 1 specifies (h,r,t)
'''
def transform_kg_csv_to_triplets(list_of_csvs, entities = [], relations = []):
    triplets = []
    valid_entity = lambda e : True if entities == [] else e in entities
    for file in list_of_csvs: 
        df_file = pd.read_csv(file)
        row = df_file.values.tolist()
        col_names = df_file.columns
        #if no specified relations then use all
        if relations == []: relations = col_names[2:]
        for r in row: 
            for i in range(2, len(r)):
                if r[i] == 1 and col_names[i] in relations and valid_entity(r[0]): 
                    triplets.append([r[0], col_names[i], r[1]])
    return triplets

'''
Takes in list of triples formatted like (h,r,t) and ratio of dataset for training, and validation
Note: testign ratio will be 1 - (train_ratio + validation_ratio)
Note: must create folder called train 
'''
def partition_dataset(triples, dataset_name, train_ratio, validation_ratio):
    num_triples = len(triples)
    seed = np.arange(num_triples)
    np.random.shuffle(seed)

    train_cnt = int(num_triples * (train_ratio))
    valid_cnt = int(num_triples * (validation_ratio))
    train_set = seed[:train_cnt]
    train_set = train_set.tolist()
    valid_set = seed[train_cnt:train_cnt+valid_cnt].tolist()
    test_set = seed[train_cnt+valid_cnt:].tolist()
    
    train_fp = "notebooks/train/{name}_train.tsv".format(name=dataset_name)
    validation_fp = "notebooks/train/{name}_valid.tsv".format(name=dataset_name)
    test_fp = "notebooks/train/{name}_test.tsv".format(name=dataset_name)

    
    with open(train_fp, 'w+') as f:
        for idx in train_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))
        
    with open(validation_fp, 'w+') as f:
        for idx in valid_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))

    with open(test_fp, 'w+') as f:
        for idx in test_set:
            f.writelines("{}\t{}\t{}\n".format(triples[idx][0], triples[idx][1], triples[idx][2]))