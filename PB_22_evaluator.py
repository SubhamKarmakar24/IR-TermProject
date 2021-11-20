import sys
import csv
import numpy as np

path_to_gold_standard = sys.argv[1]
path_to_ranked_list = sys.argv[2]
path_to_queries = "./queries_22.txt"

parsed = []
gold = []
relevance = {}
queries = []

print("Starting Task....")

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

with open(path_to_queries, 'r') as file3:
    csvreader = csv.reader(file3)
    for row in csvreader:
        queries.append(row[0].split(" ", 1)[1])



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
    temp.append(queries[i])
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

cap10 = 0
cap20 = 0
cndgc10 = 0
cndgc20 = 0

for i in range(len(p10)):
    if p10[i] > 0.01:
        sap10 = sap10 + p10[i]
        cap10 = cap10 + 1
    if p20[i] > 0.01:
        sap20 = sap20 + p20[i]
        cap20 = cap20 + 1
    if ndgc10[i] != nan:
        sndgc10 = sndgc10 + ndgc10[i]
        cndgc10 = cndgc10 + 1
    if ndgc20[i] != nan:
        sndgc20 = sndgc20 + ndgc20[i]
        cndgc20 = cndgc20 + 1

if sys.argv[2] == "PB_22_ranked_list_RF_0.csv":
    create = "./PB_22_rocchio_RF_evaluated_0.csv"
elif sys.argv[2] == "PB_22_ranked_list_RF_1.csv":
    create = "./PB_22_rocchio_RF_evaluated_1.csv"
elif sys.argv[2] == "PB_22_ranked_list_RF_2.csv":
    create = "./PB_22_rocchio_RF_evaluated_2.csv"
elif sys.argv[2] == "PB_22_ranked_list_PsRF_0.csv":
    create = "./PB_22_rocchio_PsRF_evaluated_0.csv"
elif sys.argv[2] == "PB_22_ranked_list_PsRF_1.csv":
    create = "./PB_22_rocchio_PsRF_evaluated_1.csv"
elif sys.argv[2] == "PB_22_ranked_list_PsRF_2.csv":
    create = "./PB_22_rocchio_PsRF_evaluated_2.csv"

open(create, 'w').close()
metrics = open(create, 'a')
metrics.write('Query_ID,Query,AP@10,AP@20,NDCG@10,NDCG@20\n')
for i in range(len(p10)):
    metrics.write(str(final[i][0]) + ',' + str(final[i][1]) + ',' + str(final[i][2]) + ',' + str(final[i][3]) + ',' + str(final[i][4]) + ',' + str(final[i][5]) + '\n')
metrics.write(",," + str(sap10/cap10) + ',' + str(sap20/cap20) + ',' + str(sndgc10/cndgc10) + ',' + str(sndgc20/cndgc20) + '\n')
metrics.close()

print("Task completed")

