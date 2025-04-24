# *************** Data Loading/Saving Tools ***************

import os
import numpy as np
import pandas as pd
from langchain.embeddings.openai import OpenAIEmbeddings


def identify_and_load_file(filename: str) -> dict:
    # Implementing Vector Embedding Search for Similarity
    embedding = OpenAIEmbeddings()
    dirpath, dirnames, filenames = os.walk(top=os.getcwd()+'/bb')
    if len(dirpath[2]) == 0: return 'No file found'
    elif filename not in dirpath[2]:
        embeddings = {dirfilename:embedding.embed_query(dirfilename) for dirfilename in dirpath[2]}
        similarity_scores = [(np.dot(embedding.embed_query(filename), embeddings[dirfilename]), dirfilename) 
                            for dirfilename in dirpath[2]]
        closest_filename = sorted(similarity_scores, reverse=True)[0][1]
        filename = closest_filename
    for func in [load_csv, load_excel, load_parquet, load_tsv, load_fwf]:
        dataframe = func(filename)
        if type(dataframe) == pd.DataFrame: break
    return dataframe


def save_data_to_file(dataframe: pd.DataFrame, filename: str) -> dict:
    dataframe.to_csv(r'./out/'+filename, index=False)
    return 'File saved successfully.'


def load_csv(filename: str) -> dict:
    """Check if .CSV file is present, load it into the dataframe and return it"""
    try:
        df = pd.read_csv(r'./bb/'+filename)
    except:
        return 'Error loading file'
    #return f'Verified the file as CSV and loaded it: {df}'
    return df


def load_excel(filename: str) -> dict:
    """Check if .XLS or .XLSX file is present, load (the first sheet of) it into the dataframe and return it"""
    try:
        df = pd.read_excel(r'./bb/'+filename)
    except:
        return 'Error loading file'
    #return f'Verified the file as Excel and loaded it: {df}'
    return df

def load_parquet(filename: str) -> dict:
    """Check if .PARQUET file is present, load it into the dataframe and return it"""
    try:
        df = pd.read_parquet(r'./bb/'+filename)
    except:
        return 'Error loading file'
    #return f'Verified the file as Parquet and loaded it: {df}'
    return df

def load_tsv(filename: str) -> dict:
    """Check if .TSV file is present, load it into the dataframe and return it"""
    try:
        df = pd.read_csv(r'./bb/'+filename, sep='\t', quoting=3)
    except:
        return 'Error loading file'
    #return f'Verified the file as TSV and loaded it: {df}'
    return df


def load_fwf(filename: str) -> dict: # Unknown usage
    """Check if Fixed width file (FWF) is present, load it into the dataframe and return it"""
    try:
        df = pd.read_fwf(r'./bb/'+filename)
    except:
        return 'Error loading file'
    #return f'Verified the file as FWF and loaded it: {df}'
    return df
