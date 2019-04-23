# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd

# Takes input and output directories as arguments
parser=argparse.ArgumentParser()
parser.add_argument('--input', default=".", help='The file path of the unzipped DailyDialog dataset')
parser.add_argument('--output', default="./data", help='The file path of the output dataset')
parser.add_argument('--separator', default=r"[TRN]", help='The separator token between context turns')
parser.add_argument('--turns', default="1", help='The number of previous turns to include in the context')
args = parser.parse_args()
INPUT_PATH = args.input
OUTPUT_PATH = args.output
SEPARATOR = args.separator
CONTEXT_LEVEL = int(args.turns)

database_types = ["train", "dev", "test"]

# Make the output directory if it does not currently exist
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

for database_type in database_types:
    # Open file
    original_dataframe = pd.read_csv(OUTPUT_PATH + "/" + database_type+".txt", sep="\t")

    # Construct the context series
    if CONTEXT_LEVEL > 1:
        context_series = original_dataframe["turn1"] + " " + SEPARATOR + " " + original_dataframe["turn2"]
    elif CONTEXT_LEVEL == 1:
        context_series = original_dataframe["turn1"] 
    else:
        context_series = [""] * original_dataframe.shape[0]
    
    # Create the BERT-ready dataframe
    converted_dataframe = pd.DataFrame({"id": original_dataframe["id"],
                                       "dialogue": original_dataframe["turn3"],
                                       "context": context_series,
                                       "emotion": original_dataframe["label"],
                                       })

    # Output the dataframe
    converted_dataframe.to_csv(OUTPUT_PATH+"/"+database_type+".tsv", sep='\t', encoding="utf-8")
