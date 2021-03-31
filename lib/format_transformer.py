import pandas as pd
import numpy as np   
from csv import reader
import csv
import pickle
import argparse

'''
iterates through csvs and finds all unique nodes and relations and creates dictionary like:
{
    'node1': 1
    'node2' : 2
    ...
    'relation1' : 100
    'relation2' : 101
}
and then saves to file [kg_dict.pickle]
Precondition: assumes valid paths to csvs from the python root
'''
def create_kg_dictionary(list_of_csvs):
    words = []
    for file in list_of_csvs:
        with open(file) as csvfile:
            kgreader = csv.reader(csvfile, delimiter=',')
            for i,row in enumerate(kgreader):
                if i == 0: words = words + row[2:]
                else:
                    words += row[:2] 
    unique_words = np.unique(np.array(words)).tolist()
    kg_dict = {k: v for v, k in enumerate(unique_words)}
    
    with open('kg_dict.pickle', 'wb') as handle:
        pickle.dump(kg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return kg_dict
'''
takes list of csvs and returns triplets of [h,r,t]. Assumes csv if formatted like:
Head  Tail  Relation1  Relation2 ...
h1    h2    1          0
h1    h3    0          1
where 1 specifies (h,r,t)
'''
def transform_kg_csv_to_triplets(list_of_csvs, entities = [], relations = [], relation_sample_ratio=1):
    with open('kg_dict.pickle', 'rb') as handle:
        kg_dict = pickle.load(handle)
    triplets = []
    valid_entity = lambda e : True if entities == [] else e in entities
    for file in list_of_csvs: 
        df_file = pd.read_csv(file).sample(frac = relation_sample_ratio)
        row = df_file.values.tolist()
        col_names = df_file.columns
        #if no specified relations then use all
        if relations == []: relations = col_names[2:]
        for r in row: 
            for i in range(2, len(r)):
                if r[i] == 1 and col_names[i] in relations and valid_entity(r[0]): 
                    triplets.append([kg_dict[r[0]], kg_dict[col_names[i]], kg_dict[r[1]]])
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
            
'''takes in dataset name and returns a dictionary containing "train" for training set, "valid" for validation set,
"test" for test set, "num_nodes" for the number of unique nodes as a head, "num_rels" for number of unique relations
in total dataset
Precondition: assumes there as a directory in notebooks/train/ which contains the dataset stored in {dataset_name}_{train,valid,test}.tsv
'''
def load_data(dataset_name): 
    dataset = {}
    train_fp = "notebooks/train/{name}_train.tsv".format(name=dataset_name)
    validation_fp = "notebooks/train/{name}_valid.tsv".format(name=dataset_name)
    test_fp = "notebooks/train/{name}_test.tsv".format(name=dataset_name)
    
    for d_part in ["train", "valid", "test"]:
        fp = "notebooks/train/{name}_{dataset_part}.tsv".format(name=dataset_name, dataset_part=d_part)
        
        with open(train_fp, 'r') as read_obj:
            csv_reader = reader(read_obj, delimiter = "\t")
            dataset[d_part] = [[int(x) for x in rec] for rec in csv.reader(read_obj, delimiter='\t')]
            dataset[d_part] = np.array(dataset[d_part])
    all_of_data = np.vstack((dataset["train"],dataset["test"],dataset["valid"]))
    df = pd.DataFrame(all_of_data, columns=["h","r","t"])
    nodes = np.append(df.h.values, df.t.values)
    dataset["num_nodes"] = len(np.unique(nodes))
    dataset["num_rels"] = len(np.unique(df.r.values))
    return dataset

def main(args):
    dict_list_csv = ['data/v1_res/relation/DDires.csv','data/v1_res/relation/DGres.csv','data/v1_res/relation/DiGres.csv','data/v1_res/relation/GGres.csv']
    drkg_file = 'data/v1_res/relation/DDires.csv'

    if args.create_dict == 1:
        dct = create_kg_dictionary(dict_list_csv)
    triples = transform_kg_csv_to_triplets([drkg_file], relation_sample_ratio=args.sample_ratio)
    print("partitioning dataset")
    partition_dataset(triples, args.dataset, args.train_ratio, args.validation_ratio)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format Transfomer')
    parser.add_argument("--create-dict", type=int, default=0,
            help="set to 1 to recreate kg_dict")
    parser.add_argument("--sample-ratio", type=float, default=.05,
            help="ratio of dataset to sample between 0 and 1")
    parser.add_argument("--dataset", type=str, default="WANGKG",
            help="string name of dataset")
    parser.add_argument("--train-ratio", type=float, default=.9,
            help="ratio of dataset for training")
    parser.add_argument("--validation-ratio", type=float, default=.05,
            help="ratio of dataset for validation, it will test on the rest")
    print("running main..")
    args = parser.parse_args()
    print(args)
    main(args)
    
    print("finished")
    
    
    
    
    
    
    
    
    
    