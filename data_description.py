import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse
from os import listdir
import os.path
import pickle

file_suffix = ".csv"

def qcut_encode(column, df, b=5):
    """
    Encode the column of the dataframe with qcut binning and returns a mapping

    column: the column to be encoded
    df: the dataframe
    b: number of bins to cut the column

    Return a mapping as a result of the qcut
    """
    while b > 1:
        try:
            mapping = pd.qcut(df[column].value_counts(), b) 
            break
        except ValueError as e:
            print("Unable to use bin =", b ,"to qcut the column", column, ", trying b-1...")
            b -= 1
    if b==1:
        raise Exception("Unable to qcut the column",column)
    return mapping.cat.codes.to_dict() # convert the categorical accessor to a dictionary

def write_to(file, df, meta="", append = True, index_col = None):
    """
    Write the dataframe to a file

    file: path to the file
    df: dataframe to write to
    meta: any meta information to be included in the end of the file
    append: boolean for whether to overwrite the file or append
    index_col: a boolean to set the index option 
    """
    m = 'a' if append else 'w'
    i = True if index_col else False
    df.to_csv(file, header = True, mode = m, index = i)
    with open(file, 'a') as f:
            f.write(meta)
    print("Successfully write to ",file,"!")

def load_offices():
    """
    Create a code map for Adobe offices with incremental coding scheme starting from 1. The possible offices are in offices.txt.

    Return a dictionary for the office map with keys all lower cased, ex. office['chicago'] = 2
    """
    with open('offices.txt', 'r') as file:
        cities = [l.lower().strip().split("/") for l in file.readlines()] 
        m = {}
        i = 1
        for c in cities:
            if not len(c) == 1: # use case: Midrand/Johannesburg should be encoded with the same number
                for k in c:
                    m[k] = max(i - 1, 1)
            else:
                m[c[0]] = i
            i+=1
        with open('offices.pickle', 'wb') as p:
            pickle.dump(m, p, 0)
    return m

def load_types(df, col, filename = ""):
    """
    Append code map in filename.pickle/col.pickle or create one if it doesn't exist yet, the pickle code map is a dictionary itself, with mapper[column value] maps to encoded value. The encoding is basically incremental from 1, with mapper['max'] as the max encoded value so far.

    df: the dataframe that contains an okta csv logs
    col: the column to be encoded
    filename: same as col if not specified, otherwise a string for the name of the code map
    """
    keys = df[col].unique()
    filename = col if len(filename) == 0 else filename
    if not os.path.isfile(filename+'.pickle'):
        with open(filename+'.pickle', 'wb') as f:
            mapper = {j: i+1 for i, j in enumerate(keys)}
            mapper['max'] = len(keys) # next max+i+1
            pickle.dump(mapper, f, 0)
    else:
        with open(filename+'.pickle', 'rb') as f:
            mapper = pickle.load(f)
        with open(filename+'.pickle', 'ab') as f:
            max_i = mapper['max']
            k = 0
            for j in keys:
                if not j in mapper:
                    mapper[j] = max_i+k+1
                    k+=1
            mapper['max'] = max_i+len(keys)
            pickle.dump(mapper, f, 0)
    return mapper

def main(files):
    """
    Encode the features in each log and output the encoded okta csv with prefix "sys_"
    
    Columns being encoded:
    - 'agent' -> 'agent code' with qcut binning, a system leve encoding
    - 'src_city' -> 'src_city' with 1-1 mapping based on offices.txt, check offices.pickle for code map
    - 'user_work_city' -> 'user_work_city' with 1-1 mapping based on offices.txt, check offices.pickle for code map
    - 'action.objectType' -> 'action.objectType' with 1-1 mapping, check action.objectType.pickle for code map
    - 'eventtype' -> 'eventtype' with 1-1 mapping, check eventtype.pickle for code map
    - 'published' -> 'published' with hour as the encoding
    And new encoded columns being added are:
    - 'same_city': 1 if src_city == user_work_city else 0
    
    files: list of file paths
    """
    offices_map = load_offices()
    offices = offices_map.keys()

    def office_helper(r):
        """
        Encode the src_city and user_work_city based on offices.txt and offices.pickle, and 0 for unknown.

        r: the row of the dataframe
        """
        if pd.isnull(r):
            return 0
        for o in offices: #san jose/san francisco?
            if o in r.lower():
                return offices_map[o]
        return 0

    def city_helper(r):
        """
        Encode the same_city column based on src_city and user_work_city.

        r: the row of the dataframe
        """
        if r['src_city'] == 0 or r['user_work_city'] == 0:
            return 0
        if r['src_city'] == r['user_work_city']:
            return 1
        return 2

    for file in files:
        directory, f = os.path.split(file)
        df = pd.read_csv(file, index_col = 0, dtype=object, encoding = "cp1252") #MAKE SURE eventId is set as index during read_raw.py(or the first column for all named_okta_*.csv files)

        # Create mappings for the columns
        action_objectType_map = load_types(df, 'action.objectType')
        eventtype_map = load_types(df, 'eventtype')
        action_map = load_types(df, 'action')
        
        # Encode the columns with the mappings
        for col in df.columns.values:
            if col == 'agent':
                mapping = qcut_encode(col, df)       
                df['agent code'] = df['agent'].apply(lambda x: mapping[x]+1 if pd.notnull(x) else 0)
            elif 'city' in col:
                df[col] = df[col].apply(office_helper)
            elif 'objectType' in col:
                df[col] = df[col].apply(lambda x: action_objectType_map[x] if pd.notnull(x) else 0)
            elif col == 'eventtype':
                df[col] = df[col].apply(lambda x: eventtype_map[x] if pd.notnull(x) else 0)
            elif col == 'action':
                df[col] = df[col].apply(lambda x: action_map[x] if pd.notnull(x) else 0)
            elif col == 'published':
                df[col] = pd.to_datetime(df[col]).dt.hour
            # Add your new column logic here:

        df['same_city'] = df.apply(city_helper, axis = 1)
        write_to(os.path.join(directory, "sys_encoded_"+f), df, index_col = 'eventId')

if __name__ == "__main__":
    """
    Run the script to encode named okta logs at a system level, meaning that all users share the same encoding scheme at this stage. The code maps are pickle files.

    arg:
        -p prefix of named raw csv data, ex. named_okta_
        -d relative directory of the raw csv data from this script directory, ex. data\
        -f the file name of a specific raw csv data without the .csv suffix, ex. okta_8_11
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store',
                    dest='directory',
                    default = '',
                    help='The directory for this read operation, ex. data\\')
    parser.add_argument('-p', action='store',
                    dest='prefix',
                    help='Read all raw files started with a given prefix under the directory')
    parser.add_argument('-f', action='store',
                        dest='file',
                        help='Only read a specific raw file with a given file name(without suffix)')
    results = parser.parse_args()
    root = os.path.abspath(os.path.dirname(__file__))
    if (results.prefix and results.file) or not (results.prefix or results.file):
        parser.error("Specify either -p or -f, but NOT both!")

    if len(results.directory) == 0:
        files = [f for f in listdir() if f.endswith(file_suffix) and f.startswith(results.prefix)] if results.prefix else [results.file+file_suffix]
    else:
        files = [os.path.join(root, results.directory, f) for f in listdir(results.directory) if f.endswith(file_suffix) and f.startswith(results.prefix)] if results.prefix else [results.directory+"\\"+results.file+file_suffix]
    main(files)
