import sys
import csv
import numpy as np
from tabulate import tabulate

path_to_gold_standard = sys.argv[1]
path_to_ranked_list = sys.argv[2]

parsed = []
gold = []
relevance = {}

with open(path_to_ranked_list, 'r') as file1:
    csvreader = csv.reader(file1)
    header = next(csvreader)
    for row in csvreader:
        parsed.append(row)

with open(path_to_gold_standard, 'r') as file2:
    csvreader = csv.reader(file2)
    header = next(csvreader)
    for row in csvreader:
        gold.append(row)


output_parsed = {}
output_gold = {}

temp = []
k = 0
prev = parsed[0][0]
for i in range(len(parsed)):
    if k != 20 and prev == parsed[i][0]:
        k = k + 1
        temp.append(parsed[i][1])
    else:
        if prev != parsed[i][0]:
            output_parsed[parsed[i-1][0]] = temp
            temp = []
            k = 1
            prev = parsed[i][0]
            temp.append(parsed[i][1])
output_parsed[parsed[i][0]] = temp

temp = []
prev = gold[0][0]
for i in range(len(gold)):
    if prev == gold[i][0]:
        temp.append(gold[i][1])
    else:
        output_gold[gold[i-1][0]] = temp
        temp = []
        prev = gold[i][0]
        temp.append(gold[i][1])
output_gold[gold[i][0]] = temp


t = []
prev = gold[0][0]
for i in range(len(gold)):
    if prev == gold[i][0]:
        t.append(gold[i][1:3])
    else:
        relevance[gold[i-1][0]] = t
        t = []
        prev = gold[i][0]
        t.append(gold[i][1:3])
relevance[gold[i][0]] = t



counter = 0
s = 0
p10 = []
p20 = []
calc = []
for x, y in output_parsed.items():
    counter = 0
    calc = []
    if x in output_gold:
        for i in range(20):
            for j in range(len(output_gold[x])):
                if y[i] == output_gold[x][j]:
                    counter = counter + 1
                    break
            calc.append(counter/(i+1))
        s = 0
        for i in range(10):
            s = s + calc[i]
        p10.append(s/10)
        for i in range(10, 20):
            s = s + calc[i]
        p20.append(s/20)
    else:
        p10.append(0)
        p20.append(0)


scale10 = {}
scale20 = {}
found = 0
temp = []
temp1 = []
for x, y in output_parsed.items():
    if x in relevance:
        for i in range(10):
            found = 0
            for j in range(len(relevance[x])):
                if(y[i] == relevance[x][j][0]):
                    found = 1
                    temp.append(int(relevance[x][j][1]))
                    break
            if found == 0:
                temp.append(0)
        scale10[x] = temp
        temp = []
        for i in range(20):
            found = 0
            for j in range(len(relevance[x])):
                if(y[i] == relevance[x][j][0]):
                    found = 1
                    temp1.append(int(relevance[x][j][1]))
                    break
            if found == 0:
                temp1.append(0)
        scale20[x] = temp1
        temp1 = []
    else:
        temp = [0,0,0,0,0,0,0,0,0,0]
        scale10[x] = temp
        temp = []
        temp1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        scale20[x] = temp1
        temp1 = []


temp_sorted = []
scale10_sorted = {}
scale20_sorted = {}

for x, y in scale10.items():
    temp_sorted = []
    for i in range(len(y)):
        temp_sorted.append(y[i])
    temp_sorted.sort(reverse=True)
    scale10_sorted[x] = temp_sorted

temp_sorted = []
for x, y in scale20.items():
    temp_sorted = []
    for i in range(len(y)):
        temp_sorted.append(y[i])
    temp_sorted.sort(reverse=True)
    scale20_sorted[x] = temp_sorted



def discountedCumulativeGain(result):
    dcg = []
    for idx, val in enumerate(result): 
        numerator = 2**val - 1
        # add 2 because python 0-index
        denominator =  np.log2(idx + 2) 
        score = numerator/denominator
        dcg.append(score)
    return sum(dcg)

nan = 0

def normalizedDiscountedCumulativeGain(result, gold_standard): 
    dcg = discountedCumulativeGain(result)
    idcg = discountedCumulativeGain(gold_standard)
    if idcg!= 0:
        ndcg = dcg / idcg
    else:
        ndcg = nan
    return ndcg

ndgc10 = []
ndgc20 = []

for x, y in scale10.items():
    ndgc10.append(normalizedDiscountedCumulativeGain(y, scale10_sorted[x]))

for x, y in scale20.items():
    ndgc20.append(normalizedDiscountedCumulativeGain(y, scale20_sorted[x]))


final = []
temp = []
for i in range(len(p10)):
    temp.append(i+126)
    temp.append(p10[i])
    temp.append(p20[i])
    temp.append(ndgc10[i])
    temp.append(ndgc20[i])
    final.append(temp)
    temp = []


sap10 = 0
sap20 = 0
sndgc10 = 0
sndgc20 = 0

for i in range(len(p10)):
    sap10 = sap10 + p10[i]
    sap20 = sap20 + p20[i]
    if ndgc10[i] != nan:
        sndgc10 = sndgc10 + ndgc10[i]
    if ndgc20[i] != nan:
        sndgc20 = sndgc20 + ndgc20[i]

if sys.argv[2] == "PAT2_22_ranked_list_A.csv":
    create = "./PAT2_22_metrics_A.txt"
elif sys.argv[2] == "PAT2_22_ranked_list_B.csv":
    create = "./PAT2_22_metrics_B.txt"
elif sys.argv[2] == "PAT2_22_ranked_list_C.csv":
    create = "./PAT2_22_metrics_C.txt"

open(create, 'w').close()
metrics = open(create, 'a')
metrics.write(tabulate(final, headers=["Query ID", "AP @ 10", "AP @ 20", "NDGC @ 10", "NDGC @ 20"]))
metrics.write('\n\n')
metrics.write("Mean Average Precision @ 10 = " + str(sap10/len(p10)) + '\n')
metrics.write("Mean Average Precision @ 20 = " + str(sap20/len(p20)) + '\n')
metrics.write("Average Normalized Discounted Cumulative Gain @ 10 = " + str(sndgc10/len(ndgc10)) + '\n')
metrics.write("Average Normalized Discounted Cumulative Gain @ 20 = " + str(sndgc20/len(ndgc20)) + '\n')
metrics.close()

