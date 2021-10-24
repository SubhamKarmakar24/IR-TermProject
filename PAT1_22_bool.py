import pickle
import sys
idx = {}
pic_file = sys.argv[1]
pickle_file = open(pic_file, "rb")
objec = pickle.load(pickle_file)
for j in objec:
    idx[j] = [x[0] for x in objec[j]]
map1 = {}
for g in objec:
    for c in objec[g]:
        map1[c[0]] = c[2]

q_file = sys.argv[2]
query_file = open(q_file, 'r')
x = query_file.read()
list_x = x.split('\n')
query = []
for i in list_x:
    query.append([i.split(' ')[0],' '.join(i.split(' ')[1:])])

def merge(arr1,arr2,n1,n2):
    ans = []
    i=0
    j=0
    while i<n1 and j<n2:
        if arr1[i]==arr2[j]:
            ans.append(arr1[i])
            i+=1
            j+=1
        elif arr1[i]<arr2[j]:
            i+=1
        else:
            j+=1
    return ans
bits = []
for u in query[:-1]:
    doc = {}
    for f in u[1].split(' '):
        if f in idx:
            doc[len(idx[f])] = idx[f]
    st = sorted(doc.keys())
    if len(st)==len(u[1].split(' ')):
        merged = doc[st[0]]
        for i in range(1,len(st)):
                merged = merge(merged,doc[st[i]],len(merged),len(doc[st[i]]))
        
    bits.append([u[0],[map1[x] for x in merged]])  
    
file1 = open('PAT1_22_results.txt','w')
for c in bits:
    if len(c[1])!=0:
        file1.write("{} : {}\n".format(c[0],', '.join(c[1])))
