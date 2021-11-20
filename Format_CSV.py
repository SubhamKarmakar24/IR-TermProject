import sys
import csv
import numpy as np

path_to = sys.argv[1]

parsed = []
c = 0

with open(path_to, 'r') as file1:
    csvreader = csv.reader(file1)
    header = next(csvreader)
    parsed.append(header)
    for row in csvreader:
        if c == 1:
            c = 0
            parsed.append(row)
        else:
            c = 1

if sys.argv[1] == "PB_22_rocchio_RF_metrics1.csv":
    create = "./PB_22_rocchio_RF_metrics1_formatted.csv"
elif sys.argv[1] == "PB_22_rocchio_RF_metrics2.csv":
    create = "./PB_22_rocchio_RF_metrics2_formatted.csv"
elif sys.argv[1] == "PB_22_rocchio_RF_metrics3.csv":
    create = "./PB_22_rocchio_RF_metrics3_formatted.csv"

open(create, 'w').close()
metrics = open(create, 'a')
metrics.write(parsed[0][0] + ',' + parsed[0][1])
for i in range(1, len(parsed)):
    metrics.write('\n' + parsed[i][0] + ',' + parsed[i][1])
