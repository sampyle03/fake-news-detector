#data_preprocessor.py
"""
Function: Processes data from Politifact dataset, and saves it to a pkl file
Parameters: None
Returns: None
"""

import pandas as pd
import os
import csv

def process_data(data_path):
    # Load data from json file
    chosen_columns = ["id", "label", "statement", "subjects", "speaker", "speaker_job_title", "state_info", "party_affiliation","barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]
    unprocessed_data = pd.DataFrame(columns=chosen_columns)
    with open(data_path) as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for row in rd:
            unprocessed_data = pd.concat([unprocessed_data, pd.DataFrame([row], columns=chosen_columns)], ignore_index=True)
            print(unprocessed_data)
            input("Press Enter to continue...")

    # Drop unnecessary columns
"""    dropped_columns = unprocessed_data.drop(columns=['statement_date', "statement_source", "factcheck_date", "factcheck_analysis_link"])
    print(dropped_columns.columns)
    input("Press Enter to continue...")"""



current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "../data/train.tsv")
process_data(data_path)