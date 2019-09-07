import argparse

import pandas as pd
import os
from absl import logging
logging.set_verbosity(logging.INFO)
#works well only when objgraph outputs 50 items :(
from print_helper import print_info, print_error


def main(log_file_name):
    file = os.path.expanduser(log_file_name)

    key_value = dict()

    list_key_value = []
    with open(file, "r") as file:
        lines = file.readlines()
        found_list = False
        for line in lines:

            if "objgraph growth list start" in line and "print" not in line:
                found_list = True
                continue

            if found_list:
                obj_counts = line.split(" ")
                obj_counts = [value for value in obj_counts if len(value) > 0]
                key_value[obj_counts[0]] = obj_counts[-1].replace("+", "").replace("\n", "")
                # print_error(key_value)

            if "objgraph growth list end" in line and "print" not in line:
                found_list = False
                list_key_value.append(key_value)
                key_value = dict()


    df = pd.DataFrame(list_key_value).T
    print(df)
    df.to_csv(log_file_name.split(".")[0] + ".csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Objgraph log Parser  : ')

    parser.add_argument('-f', "--log_file_name", default=False, help="log_file_name")


    parsed_args = vars(parser.parse_args())

    main(log_file_name=parsed_args["log_file_name"])
