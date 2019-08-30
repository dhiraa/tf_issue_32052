import pandas as pd
import os

#works well only when objgraph outputs 50 items :(

def main():
    file = os.path.expanduser("log.txt")

    key_value = dict()

    list_key_value = []
    with open(file, "r") as file:
        lines = file.readlines()
        line_count = 0
        found_list = False
        for line in lines:

            if "Initial objects" in line or "Object growth" in line or "objgraph growth" in line:
                found_list = True
                continue

            if found_list and line_count <= 50:
                obj_counts = line.split(" ")
                obj_counts = [value for value in obj_counts if len(value) > 0]
                key_value[obj_counts[0]] = obj_counts[-1].replace("+", "").replace("\n", "")
                #print(key_value)
                line_count += 1

            if line_count == 50:
                found_list = False
                line_count = 0
                list_key_value.append(key_value)
                key_value = dict()


    df = pd.DataFrame(list_key_value).T
    print(df)
    df.to_csv("objgraph_tf_dataset_analysis.csv")

if __name__ == '__main__':
    main()
