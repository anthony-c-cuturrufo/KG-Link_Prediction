import pandas as pd
import numpy as np   
from csv import reader
import csv
import pickle
import argparse
import os

'''
Creates and saves files nodes_dict and rels_dict from [triplets] which are dictionary mapping from
node name to a unique id. [triplets] is structured like [[h1,r1,t1],...]
'''
def create_dicts_from_triplets(triplets):
    print("creating nodes and relation vocabulary")
    df = pd.DataFrame(triplets, columns=['h','r','t'])
    unique_nodes = np.unique(np.concatenate((df.h.values,df.t.values)))
    unique_rels = np.unique(df.r.values)
    nodes_dict = {k: v for v, k in enumerate(unique_nodes)}
    rels_dict = {k: v for v, k in enumerate(unique_rels)}

    with open('nodes_dict.pickle', 'wb') as handle:
        pickle.dump(nodes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('rels_dict.pickle', 'wb') as handle:
        pickle.dump(rels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print("finished creating nodes and relation vocabulary")
    new_triplets = []
    for t in triplets:
        new_t = [0,0,0]
        for i,x in enumerate(t):
            if i == 1:
                new_t[i] = rels_dict[x]
            else:
                new_t[i] = nodes_dict[x]
        new_triplets.append(new_t)                             
    return new_triplets

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
    node_names = []
    rel_names = []
    for file in list_of_csvs:
        with open(file) as csvfile:
            kgreader = csv.reader(csvfile, delimiter=',')
            for i,row in enumerate(kgreader):
                if i == 0: rel_names = rel_names + row[2:]
                else:
                    node_names += row[:2] 
    unique_nodes = np.unique(np.array(node_names)).tolist()
    unique_rels = np.unique(np.array(rel_names)).tolist()
    nodes_dict = {k: v for v, k in enumerate(unique_nodes)}
    rels_dict = {k: v for v, k in enumerate(unique_rels)}
    
    with open('nodes_dict.pickle', 'wb') as handle:
        pickle.dump(nodes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('rels_dict.pickle', 'wb') as handle:
        pickle.dump(rels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return nodes_dict, rels_dict
'''
takes list of csvs and returns triplets of [h,r,t]. Assumes csv if formatted like:
Head  Tail  Relation1  Relation2 ...
h1    h2    1          0
h1    h3    0          1
where 1 specifies (h,r,t)
'''
def transform_kg_csv_to_triplets(list_of_csvs, entities = [], relations = [], relation_sample_ratio=1, is_dict = False):
    nodes_dict = {}
    rels_dict = {}
    if is_dict:
        with open('nodes_dict.pickle', 'rb') as handle:
            nodes_dict = pickle.load(handle)
        with open('rels_dict.pickle', 'rb') as handle:
            rels_dict = pickle.load(handle)
        
    triplets = []
    valid_entity = lambda e : True if entities == [] else e in entities
    for file in list_of_csvs: 
        val_relations = relations
        df_file = pd.read_csv(file).sample(frac = relation_sample_ratio)
        row = df_file.values.tolist()
        col_names = df_file.columns
        #if no specified relations then use all
        if relations == []: val_relations = col_names[2:]
        for r in row: 
            for i in range(2, len(r)):
                if r[i] == 1 and col_names[i] in val_relations and valid_entity(r[0]):
                    if is_dict:
                        triplets.append([nodes_dict[r[0]], rels_dict[col_names[i]], nodes_dict[r[1]]])
                    else:
                        triplets.append([r[0], col_names[i], r[1]])
    if not is_dict:
        triplets = create_dicts_from_triplets(triplets)
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
        
        with open(fp, 'r') as read_obj:
            csv_reader = reader(read_obj, delimiter = "\t")
            dataset[d_part] = [[int(x) for x in rec] for rec in csv.reader(read_obj, delimiter='\t')]
            dataset[d_part] = np.array(dataset[d_part])
    all_of_data = np.vstack((dataset["train"],dataset["test"],dataset["valid"]))
    df = pd.DataFrame(all_of_data, columns=["h","r","t"])
    nodes = np.append(df.h.values, df.t.values)
    dataset["num_nodes"] = len(np.unique(nodes))
    dataset["num_rels"] = len(np.unique(df.r.values))
    return dataset

'''
Read in three .tsv files [filenames] for training, validation, and testing and converts string node names into integers using an vocabulary index map. Returns the resulting embedded [triplets]. Files must be names ..._train for training set, ..._valid for validation set, ..._test for testing set
'''
def embed_partitioned_data(filenames, name):    
    node_vocab = {}
    relation_vocab = {}
    
    curr_node_index = 0
    curr_relation_index = 0

    for file in filenames: 
        data_type = ""
        triplets = []
        df_file = pd.read_csv(file, header=None, delimiter='\t')
        row = df_file.values.tolist()
        for r in row:
            trip = [0,0,0]
            for i, element in enumerate(r):
                if i == 1: #element is a relation 
                    if element in relation_vocab:
                        trip[i] = relation_vocab[element]
                    else:
                        relation_vocab[element] = curr_relation_index
                        curr_relation_index += 1
                        trip[i] = relation_vocab[element]
                else: #element is a node  
                    if element in node_vocab:
                        trip[i] = node_vocab[element]
                    else:
                        node_vocab[element] = curr_node_index
                        curr_node_index += 1
                        trip[i] = node_vocab[element]
            triplets.append(trip)
        if "_train" in file:
            data_type = "_train"
        elif "_valid" in file:
            data_type = "_valid"
        elif "_test" in file:
            data_type = "_test"
        else:
            raise ValueError("Filenames need to be specified with _train, _valid, and _test in respective filenames")
            
        new_filename = "notebooks/train/" + name + data_type + ".tsv"
        if os.path.exists(new_filename):
            raise ValueError("Files already exists. Only writes embedded files to files that do not currently exist. Try again with different --dataset argument")
        with open(new_filename, "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(triplets)
    return triplets
        
def main(args):
    dir_files = [f for f in os.listdir(args.data_path) if not (f.startswith('.') or f in args.exclude_files)]
    dir_files = [args.data_path + "/" + f for f in dir_files]
    if args.already_partitioned == 1: 
        embed_partitioned_data(dir_files, args.dataset) 
    else: 
        is_dict = args.create_dict == 0
        val_relations = args.valid_relations if args.valid_relations != "" else []
        triples = transform_kg_csv_to_triplets(dir_files, relations=val_relations, relation_sample_ratio=args.sample_ratio, is_dict=is_dict)
        print("partitioning dataset")
        partition_dataset(triples, args.dataset, args.train_ratio, args.validation_ratio)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Format Transfomer')
    parser.add_argument("--data-path", type=str, default="data/v1_res/relation",
            help="path to folder with knowledge graphs, default=data/v1_res/relation")
    parser.add_argument('--exclude-files', nargs='*', type=str, default="",
            help="list files you wish to exlude from training")
    parser.add_argument('--valid-relations', nargs='*', type=str, default="",
            help="list all the relations you want to train with, default is all of them")
    parser.add_argument("--create-dict", type=int, default=1,
            help="set to 0 if vocabulary has already been created, default is 1")
    parser.add_argument("--sample-ratio", type=float, default=.05,
            help="ratio of dataset to sample between 0 and 1, default is .05")
    parser.add_argument("--dataset", type=str, default="WANGKG",
            help="string name of dataset")
    parser.add_argument("--train-ratio", type=float, default=.9,
            help="ratio of dataset for training, default=.9")
    parser.add_argument("--validation-ratio", type=float, default=.05,
            help="ratio of dataset for validation, it will test on the rest, default=.05")
    parser.add_argument("--already-partitioned", type=int, default=0,
            help="set to 1 if we are training on already partitioned dataset (train, validation, test)")
    print("running main..")
    args = parser.parse_args()
    print(args)
    main(args)
    
    print("finished")
    
    
    
    
    
    
    
    
    
    