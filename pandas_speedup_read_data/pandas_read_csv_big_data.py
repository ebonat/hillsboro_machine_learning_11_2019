

# 50 times faster data loading for Pandas: no problem
# Loading irregular data into Pandas using C++
# https://blog.esciencecenter.nl/irregular-data-in-pandas-using-c-88ce311cb9ef
# GitHub
# https://github.com/TICCLAT/ticcl-output-reader

# TRY IT THIS!
# I usually parse once and save a pickled version of DataFrame and then pd.read_pickle for the next loads. 
# (Works nice when loading multiple times the same file)!


import os
import sys
import time

import numpy as np         
import pandas as pd        
import pathlib as pl  

# from pathlib import Path, PureWindowsPath

np.set_printoptions(suppress=True)   

def get_code_runtime(start_time):    
    end_time = time.clock()
    diff_time = end_time - start_time
    code_runtime = time.strftime("%H:%M:%S", time.gmtime(diff_time)) 
    print("program runtime: {}".format(code_runtime))
    
def main1():
#     how long to read in pandas (> 30 seconds)
    start_time = time.clock()
    csv_file_path = r"E:\Visual WWW\Python\05 DATA PROJECTS\2015 Flight Delays and Cancellations\flights_11M.csv"    
    df = pd.read_csv(filepath_or_buffer=csv_file_path, low_memory=False)
    print("df rows count: {}".format(len(df)))
    get_code_runtime(start_time)
    
#     convert to np array
#     start_time = time.clock()
#     np_array = np.array(df)  
#     print("np array rows count: {}".format(np_array.shape))
#     get_code_runtime(start_time)
    
def main2():        
#     how long to read in pandas (> 30 seconds)
    start_time = time.clock()
    print("loading pandas dataframe")
    csv_file_path = r"E:\Visual WWW\Python\05 DATA PROJECTS\2015 Flight Delays and Cancellations\flights_5.8M.csv"    
    df = pd.read_csv(filepath_or_buffer=csv_file_path, low_memory=False)
    print("df rows count: {}".format(len(df)))
    get_code_runtime(start_time)
     
#     how long to convert to pickle (> 5 seconds)
#     total runtime = 30 - 35 seconds    
    start_time = time.clock()
    print("writting pickle serialize file")
    project_path = pl.Path(os.path.dirname(os.path.abspath(__file__)))
    pkl_path = project_path / "pkl"
    flights_pickle = "flights.pkl"
    flights_pickle_path_name = pkl_path / flights_pickle
    pd.to_pickle(df, flights_pickle_path_name)
    get_code_runtime(start_time)

def main3():    
#     how long to read from pickle (< 4 seconds) - EXCELLEN RESULT 
    start_time = time.clock()
    print("reading pickle serialize file")
    project_path = pl.Path(os.path.dirname(os.path.abspath(__file__)))
    pkl_path = project_path / "pkl"
    flights_pickle = "flights.pkl"
    flights_pickle_path_name = pkl_path / flights_pickle
    df = pd.read_pickle(flights_pickle_path_name)
    print("df rows count: {}".format(len(df)))
    get_code_runtime(start_time)
        
# def main4():  
#     start_time = time.clock()
#     csv_file_path = r"E:\Visual WWW\Python\05 DATA PROJECTS\2015 Flight Delays and Cancellations\flights_5.8M.csv"    
#     data = np.fromfile(file=csv_file_path, sep=",")
#     print(data)
#     get_code_runtime(start_time)

if __name__ == '__main__':
#     main2()
#     main3()
    main4()

