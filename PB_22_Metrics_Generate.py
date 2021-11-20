import sys
import csv

path_to_file1 = sys.argv[1]
path_to_file2 = sys.argv[2]
path_to_file3 = sys.argv[3]

parsed1 = []
parsed2 = []
parsed3 = []

print("Starting Task....")

with open(path_to_file1, 'r') as file1:
    csvreader = csv.reader(file1)
    header = next(csvreader)
    parsed1.append(header)
    for row in csvreader:
        parsed1.append(row)

with open(path_to_file2, 'r') as file2:
    csvreader = csv.reader(file2)
    header = next(csvreader)
    parsed2.append(header)
    for row in csvreader:
        parsed2.append(row)

with open(path_to_file3, 'r') as file3:
    csvreader = csv.reader(file3)
    header = next(csvreader)
    parsed3.append(header)
    for row in csvreader:
        parsed3.append(row)
        

if sys.argv[1] == "PB_22_rocchio_RF_evaluated_0.csv" and sys.argv[2] == "PB_22_rocchio_RF_evaluated_1.csv" and sys.argv[3] == "PB_22_rocchio_RF_evaluated_2.csv":
    create = "./PB_22_rocchio_RF_metrics.csv"
elif sys.argv[1] == "PB_22_rocchio_PsRF_evaluated_0.csv" and sys.argv[2] == "PB_22_rocchio_PsRF_evaluated_1.csv" and sys.argv[3] == "PB_22_rocchio_PsRF_evaluated_2.csv":
    create = "./PB_22_rocchio_PsRF_metrics.csv"

open(create, 'w').close()
metrics = open(create, 'a')
metrics.write("alpha,beta,gamma,mAP@20,NDCG@20")
metrics.write('\n' + '1' + ',' + '1' + ',' + '0.5' + ',' + parsed1[len(parsed1)-1][3] + ',' + parsed1[len(parsed1)-1][5])
metrics.write('\n' + '0.5' + ',' + '0.5' + ',' + '0.5' + ',' + parsed2[len(parsed2)-1][3] + ',' + parsed2[len(parsed2)-1][5])
metrics.write('\n' + '1' + ',' + '0.5' + ',' + '0' + ',' + parsed3[len(parsed3)-1][3] + ',' + parsed3[len(parsed3)-1][5])

print("Task completed")
