#------Code for Task B2 of 'DEFAULT' project----
#------Written by Arghyadeep Bandyopadhyay (Roll No: 18EE10012)----
#------Importing the required libraries---------

import os
import sys
import pickle
import numpy as np
import csv
import math
from collections import Counter
from numpy import linalg as la

#------Function to obtain the words contained in each document------

def get_doc_contents(inverted_idx, num_docs, doc_idx_map):
    doc_content = []
    mapped_indices = {}     #Stores the inverse mapping of the doc_idx_map
    
    for i in range(num_docs):
        doc_content.append(list([]))

    for key,value in doc_idx_map.items():
        mapped_indices[value] = key
    
    for key,value in inverted_idx.items():
        cur_postings = value
        for doc,freq in cur_postings.items():
            if doc in mapped_indices.keys():
                doc_content[mapped_indices[doc]].append(key)

    return doc_content       


#-------Function to obtain the |V|-dimensional vector for a particular document-----

def get_doc_vectors(doc_words, doc_no, df, inverted_idx, term_idx):
    
    tf_list = []          #List of term frequencies
    counts = {}           #Dictionary mapping each term to its frequency in the document 
    
    for word in doc_words:
        temp = inverted_idx[word]
        tf_list.append(temp[doc_no])
        counts[word] = temp[doc_no]
        
    curvec = np.zeros(len(term_idx.keys()))       #Vector for the present document
        
    for word in doc_words:
        tf = 1 + math.log10(counts[word])
        curvec[term_idx[word]] = tf                #Scheme used here is lnc.ltc, so df=1
        
    return curvec

#-------Function to obtain the top 5 important words for a particular query-----

def get_topwords(doc_contents, df, inverted_idx, term_idx, doc_list, terms_list):

    topwords = []                 #Stores the top 5 important words for a particular query
    doc_vecs = []                 #Stores the normalised document vectors obtained from the given document list
    
    #Compute and store the normalised document vectors
    
    for i in range(len(doc_list)):
        doc_no = doc_list[i][1]
        cur_doc_no = doc_list[i][0]
        doc_vec = get_doc_vectors(doc_contents[cur_doc_no], doc_no, df, inverted_idx, term_idx)
        doc_vec_normalised = doc_vec/(la.norm(doc_vec))
        doc_vecs.append(doc_vec_normalised)

    centroid = np.zeros(len(doc_vecs[0]))         #Stores the centroid of the list of document vectors
    sparse_vec = []                               #Sparse representation of the centroid, stores non-zero entries and their indices
    
    for i in range(len(centroid)):
        sum = 0
        for j in range(len(doc_vecs)):
            sum += doc_vecs[j][i]
        avg = sum/(len(doc_vecs))
        centroid[i] = avg
        if centroid[i] != 0:
            sparse_vec.append([i,centroid[i]])
        
    sparse_vec_sorted = sorted(sparse_vec, key = lambda item:item[1], reverse=True)   #Sort in descending order of TF-IDF scores of words

    for i in range(len(sparse_vec_sorted)):
        word = terms_list[sparse_vec_sorted[i][0]]           #Obtain corresponding word from the index
        if word == 'bdnews' or word == 'com' or word == 'reuters' or word == 'said':
            continue                 #Words observed to be frequently occurring in most of the documents are ignored
        topwords.append(word)
        if len(topwords) == 5:
            break
    
    return topwords    


#-------Function to obtain the top 10 retrieved documents for each query------

def get_docs_by_query(ranked_list):

    file_names = []                        #Stores the file names of relevant documents for each query
    rows = []                              #Stores the rows of the ranked list csv file
    
    with open(ranked_list,'r') as input_file:
        csvreader = csv.reader(input_file)
        header = next(csvreader)
        for row in csvreader:
            rows.append(row)

    for i in range(0,len(rows), 50):
        temp = []
        for j in range(i,i+10,1):
            temp.append(rows[j])
        file_names.append(temp)

    return file_names    


#---------Function to write the csv file storing important words------

def create_file(query_ids, words, terms_list):

    csv_output = 'PB_22_important_words' + '.csv'     #File name
    headings = ['Query_ID', 'Words']
    
    with open(csv_output,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headings)

        for i in range(len(query_ids)):
            row = []
            row.append(query_ids[i])
            second_col = []

            for j in range(len(words[i])-1):
                second_col.append(words[i][j])
                
            second_col.append(words[i][len(words[i])-1])
            row.append(second_col)
            csvwriter.writerow(row)

            
#--------Main Function---------------

def main():

    data_dir = sys.argv[1]              #Path to en_BDNews24 folder
    doc_paths = []                      #List of paths for each document
    doc_ids = []                        #List of document names
    
    for root, subdirs, files in os.walk(data_dir):         #Walk through the en_BDNews24 folder and store all the file names
        for filename in files:
            file_path = os.path.join(root, filename)
            doc_paths.append(file_path)

    doc_paths.sort()
    doc_names = [os.path.basename(file_path) for file_path in doc_paths]    #Store the corresponding file names from the paths
    
    ranked_list = sys.argv[3]
    relevant_doc_names = get_docs_by_query(ranked_list)     #Obtaining top 10 retrieved documents for each query

    all_docs_list = set()               #Stores the set of all the relevant documents over all the queries
    for i in range(len(relevant_doc_names)):
        for j in range(len(relevant_doc_names[i])):
            all_docs_list.add(relevant_doc_names[i][j][1])

    all_docs_list = list(all_docs_list)
    #print("Number of docs = ",len(all_docs_list))
 
    docs_idx = {}                         #Stores a mapping from a document name to its corresponding index in the document list
    for i in range(len(all_docs_list)):
        docs_idx[all_docs_list[i]] = i

    doc_idx_map = {}                    #Stores a mapping from a document index in the relevant documents list to its corresponding index in                                        #the overall document list           
    for i in range(len(all_docs_list)):
        for j in range(len(doc_names)):
            if doc_names[j] == all_docs_list[i]:
                doc_idx_map[i] = j
                break
            
    print("\nReading inverted index file...")

    objects = []
    pickle_file = sys.argv[2]           #Path to the inverted index file 
    
    with (open(pickle_file, "rb")) as openfile:       #Read the file and store the inverted index using pickle
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
            
    inverted_idx = objects[0]

    #For each term, store a mapping between document number and term frequency in that document

    for key,value in inverted_idx.items():     
        temp_dict = {}
        for i in range(len(value)):
            temp_dict[value[i][0]] = value[i][1]
        inverted_idx[key] = temp_dict    

    
    term_list = [key for key,value in inverted_idx.items()]     #Store the vocabulary terms in a list
    term_idx = {}                                               #Store the index in term list for each term in the vocabulary
    for i in range(len(term_list)):
        term_idx[term_list[i]] = i
        
    df = {}                                                     #Store the document frequency for each term
    for key, value in inverted_idx.items():
        df[key] = len(value.keys())

    print("Obtaining contents of the documents...")
    
    #Get the documents as an array of list of terms in each document
    relevant_doc_contents = get_doc_contents(inverted_idx, len(all_docs_list), doc_idx_map)

    top_words_list = []                 #Stores the 5 most important words for each query

    print("Obtaining document vectors and top words...")
    
    for i in range(len(relevant_doc_names)):        #Iterate over all the queries
        cur_list = []

        #Obtain the indices for the relevant documents corresponding to current query
        for j in range(len(relevant_doc_names[i])):
            temp = docs_idx[relevant_doc_names[i][j][1]]
            cur_list.append([temp,doc_idx_map[temp]])
                    
        top_words = get_topwords(relevant_doc_contents, df, inverted_idx, term_idx, cur_list, term_list)
        top_words_list.append(top_words)
    
    query_ids = [relevant_doc_names[i][0][0] for i in range(len(relevant_doc_names))]

    print("Writing to csv file...")
    create_file(query_ids, top_words_list, term_list)


if __name__ == '__main__':
    main()
        
    
        
