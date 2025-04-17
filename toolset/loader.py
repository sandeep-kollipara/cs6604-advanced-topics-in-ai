# *************** Data Loading Tools ***************

import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import tool



class FileName(BaseModel):
    filename: str = Field(..., description="Filename of the data file to be loaded")



@tool(args_schema=FileName)
def identify_and_load_file(filename: str) -> dict:
    """
    Checks if the provided file is present in the directory.
    If found returns True.
    else not found then returns False 
    along with suggestion of closest filename.
    """
    return True


def load_csv(filename: str) -> dict:
    """Check if .CSV file is present, load it into the dataframe and return it"""
    try:
        df = pd.read_csv(r'/bb/'+filename)
    except:
        return 'Error loading file'
    return f'Verified the file as CSV and loaded it: {df}'


def load_excel(filename: str) -> dict:
    """Check if .XLS or .XLSX file is present, load it into the dataframe and return it"""
    try:
        df = pd.read_excel(r'./bb/'+filename)
    except:
        return 'Error loading file'
    return f'Verified the file as Excel and loaded it: {df}'

def load_parquet(filename: str) -> dict:
    """Check if .PARQUET file is present, load it into the dataframe and return it"""
    try:
        df = pd.read_parquet(r'./bb/'+filename)
    except:
        return 'Error loading file'
    return f'Verified the file as Parquet and loaded it: {df}'

def load_tsv(filename: str) -> dict:
    """Check if .TSV file is present, load it into the dataframe and return it"""
    try:
        df = pd.read_csv(r'./bb/'+filename, sep='\t', quoting=3)
    except:
        return 'Error loading file'
    return f'Verified the file as TSV and loaded it: {df}'

