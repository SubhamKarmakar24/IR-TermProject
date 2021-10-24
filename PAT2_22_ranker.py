#------Code for Task 2A of 'DEFAULT' project-------

#------Importing the required libraries---------

import os
import sys
import pickle
import nltk
import numpy as np
import string
import csv
import math
from collections import Counter
from numpy import linalg as la


#------Function to obtain the words contained in each document------

def get_doc_contents(inverted_idx, num_docs):
    doc_content = []

    for i in range(num_docs):
        doc_content.append(list([]))

    for key,value in inverted_idx.items():
        cur_postings = value
        for doc,freq in cur_postings.items():
            doc_content[doc].append(key)

    return doc_content       


#-------Function to obtain the |V|-dimensional vector for a particular document-----

def get_doc_vectors(doc_words, doc_no, df, scheme, inverted_idx, term_idx):
    
    tf_list = []          #List of term frequencies
    counts = {}           #Dictionary mapping each term to its frequency in the document 
    
    for word in doc_words:
        temp = inverted_idx[word]
        tf_list.append(temp[doc_no])
        counts[word] = temp[doc_no]
        
    #Precalculating certain parameters needed for the different weighting schemes
    
    tf_list = np.array(tf_list)    
    tf_mean = np.mean(tf_list)
    tf_max =  np.max(tf_list)
    scheme_b_denom = 1 + math.log10(tf_mean)     
    
     
    curvec = np.zeros(len(term_idx.keys()))       #Vector for the present document
        
    for word in doc_words:
        if scheme == 'A':
            tf = 1 + math.log10(counts[word])
        elif scheme == 'B':
            tf = ((1 + math.log10(counts[word])) / scheme_b_denom)
        elif scheme == 'C':
            tf = 0.5 + ((0.5*counts[word])/tf_max)
        
        curvec[term_idx[word]] = tf                #As value of document frequency component is 'n' in each scheme, so df=1
        
    return curvec


#------Function to obtain query vectors for all the queries-------

def get_query_vectors(queries, df, scheme, num_docs, term_idx):

    query_vecs = []                   #List to store the query vectors for all the queries

    for i in range(len(queries)):
        counter = Counter(queries[i][1:])                               #Store a map between each term and its frequency in the query
        tf_list = np.array([value for key,value in counter.items()])    #Store the list of term frequencies

        tf_mean = np.mean(tf_list)                                        
        tf_max =  np.max(tf_list)
        scheme_b_denom = 1 + math.log10(tf_mean)

        curvec = np.zeros(len(term_idx.keys()))           #Vector for present query
                          
        for key,value in term_idx.items():                #Iterate over the whole vocabulary
            
            if key in queries[i][1:]:                     #Check if current term is in the query
                if scheme == 'A':
                    tf = 1 + math.log10(counter[key])
                    cur_df = math.log10(num_docs/df[key])     
                elif scheme == 'B':
                    tf = ((1 + math.log10(counter[key])) / scheme_b_denom)
                    cur_df = max(0,math.log10((num_docs - df[key])/df[key]))
                elif scheme == 'C':
                    tf = 0.5 + ((0.5*counter[key])/tf_max)
                    cur_df = max(0,math.log10((num_docs - df[key])/df[key]))
                    
                curvec[value] = (tf*cur_df)

            else:
                curvec[value] = 0                   #Term not present in the query

        query_vecs.append(np.array(curvec))

    return query_vecs


#-------Function to get the cosine similarity metric corresponding to all documents for each query

def get_scores(all_doc_words, query_vecs, df, scheme, inverted_idx, term_idx):

    scores = {}              #Mapping between a query number/index and its list of cosine similarity scores for all documents             
    query_norms = []         #List of L2-norms for all the queries 
    sparse_query_vecs = []   #List of only the non zero entries of a query vector

    for j in range(len(query_vecs)):       #Compute and store the norms and the sparse query vectors
        
        scores[j] = []
        query_norms.append(la.norm(query_vecs[j]))
        
        cur_sparse = {}
        for i in range(len(query_vecs[j])):
            if query_vecs[j][i] != 0:
                cur_sparse[i] = query_vecs[j][i]      
                
        sparse_query_vecs.append(cur_sparse)        
        

    for i in range(len(all_doc_words)):              #Iterate over all the documents
        
        if len(all_doc_words[i]) == 0:               #For empty documents, store a score of 0 for all queries 
            for j in range(len(query_vecs)):
                scores[j].append(0)
            continue
        
        doc_vec = get_doc_vectors(all_doc_words[i],i,df,scheme,inverted_idx, term_idx)  
        cur_doc_norm = la.norm(doc_vec)        #Get document vector and its L2-norm for current document         

        for j in range(len(query_vecs)):       #Iterate over all queries
            dot_pdt = 0 
            for key,value in sparse_query_vecs[j].items():
                dot_pdt = dot_pdt + (value*doc_vec[key])             #Compute dot product between current query and current document

            scores[j].append(dot_pdt/(cur_doc_norm*query_norms[j]))   #Normalize and append score
        
    return scores


#--------Function to rank the documents for a particular query and return top 50 results------

def rank_by_query(query_id,score,doc_names):

    doc_score_map = {}                     #Dictionary storing the mapping between the document name and its corresponding score
    for i in range(len(score)):
        doc_score_map[doc_names[i]] = score[i]

    #Sorting the dictionary in descending order of the cosine similarity scores
    doc_score_map_new = dict(sorted(doc_score_map.items(), key = lambda item: item[1], reverse=True))
    
    rows = []           #List storing the query ID and corresponding top 50 document names             
    
    for key,value in doc_score_map_new.items():
        cur_row = [query_id, key]
        rows.append(cur_row)
        
        if len(rows) == 50:
            break

    return rows
   

#---------Function to write the csv file for a particular scheme------

def create_file(queries, scores, doc_ids, scheme):

    csv_output = 'PAT2_22_ranked_list_' + scheme + '.csv'     #File name
    headings = ['Query_ID', 'Document_ID']
    
    with open(csv_output,'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(headings)

        for key,value in scores.items():                            #Iterate over all queries         
            rows = rank_by_query(queries[key][0], value, doc_ids)   #Obtain top 50 documents for current query
            csvwriter.writerows(rows)                               #Write current query results to the file


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
    doc_ids = [os.path.basename(file_path) for file_path in doc_paths]   #Store the corresponding file names from the paths

    print("\nNumber of documents = ",len(doc_paths))
    
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
    doc_contents = get_doc_contents(inverted_idx, len(doc_paths))   
    
    print("Obtaining queries...")
    
    query_file_path = sys.argv[3]                        #Path to text file containing the pre-processed queries
    
    with open(query_file_path,'r') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    queries = [list(line.split(" ")) for line in lines]   #Store the queries as list of terms in each query
    
    print("\nTF-IDF Vectorization...\n")
    schemes = ['A','B','C']                #Symbols for the three schemes used for weighting-normalizing

    for scheme in schemes:
        
        print("\nScheme ",scheme)

        print("Obtaining query vectors...")
        query_vectors = get_query_vectors(queries, df,scheme,len(doc_paths),term_idx)

        print("Obtaining document vectors and Scoring...")
        scores = get_scores(doc_contents, query_vectors, df,scheme,inverted_idx, term_idx)

        print("Writing results to file...")
        create_file(queries, scores, doc_ids, scheme)

        
    
if __name__ == '__main__':
    main()
