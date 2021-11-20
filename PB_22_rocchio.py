import os
import sys
import pickle
import numpy as np
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

def get_doc_vectors(doc_words, doc_no, df, inverted_idx, term_idx):
    
    counts = {}           #Dictionary mapping each term to its frequency in the document 
    
    for word in doc_words:
        temp = inverted_idx[word]
        counts[word] = temp[doc_no]
        
    curvec = np.zeros(len(term_idx.keys()))       #Vector for the present document
        
    for word in doc_words:
        tf = 1 + math.log10(counts[word])
        curvec[term_idx[word]] = tf                #Scheme used here is lnc.ltc, so df=1
        
    return curvec

#-------Function to obtain a sparse vector representation for the |V|-dimensional vector of a particular document-----

def get_sparse_doc_vector(doc_words, doc_no, df, inverted_idx, term_idx):

    counts = {}           #Dictionary mapping each term to its frequency in the document 
    
    for word in doc_words:
        temp = inverted_idx[word]
        counts[word] = temp[doc_no]
        
    wholevec = np.zeros(len(term_idx.keys()))       #Vector for the present document
    curvec = {}                                     #Sparse representation as a dictionary 
    result = []                                     #Stores the sparse vector, the original vector and its L2-norm for a document
    
    for word in doc_words:
        tf = 1 + math.log10(counts[word])
        curvec[term_idx[word]] = tf                
        wholevec[term_idx[word]] = tf                
        
    
    result.append(curvec)
    result.append(wholevec)
    result.append(la.norm(wholevec))
    
    return result

#-------Function to obtain the relevant documents for each query for Pseudo Relevance Feedback-----

def get_docs_by_query_psrf(ranked_list):

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
            temp.append(rows[j])          #Top 10 ranked documents considered as relevant
        file_names.append(temp)

    return file_names

#-------Function to obtain the relevant and non-relevant documents for each query for Relevance Feedback-----

def get_docs_by_query_rf(ranked_list, gold_std):

    file_names = []                      #Stores top 20 documents in retrieved ranked list for each query
    rows_ranked = []                     #Stores the rows of the ranked list csv file
    rows_gold = []                       #Stores the rows of the gold standard csv file
    
    with open(ranked_list,'r') as input_file:
        csvreader = csv.reader(input_file)
        header = next(csvreader)
        for row in csvreader:
            rows_ranked.append(row)

    with open(gold_std,'r') as input_file:
        csvreader = csv.reader(input_file)
        header = next(csvreader)
        for row in csvreader:
            rows_gold.append(row)
            
    gold_qdmap = {}                      #Stores the document name and relevance judgement score from gold standard for each query
    gold_docnames = {}                   #Stores the document name from gold standard for each query
    
    for i in range(0,len(rows_ranked), 50):
        temp = []
        gold_qdmap[rows_ranked[i][0]] = []
        gold_docnames[rows_ranked[i][0]] = []
        for j in range(i,i+20,1):
            temp.append(rows_ranked[j])
        file_names.append(temp)

    cur_list = []                        #Stores the document names and relevance judgement scores for a particular query
    cur_docs = []                        #Stores the document names for a particular query
    prev = rows_gold[0][0]
    
    for i in range(len(rows_gold)):
        if prev == rows_gold[i][0]:
            cur_list.append(rows_gold[i][1:3])
            cur_docs.append(rows_gold[i][1])             #Previous and current rows correspond to same query
            
        else:
            gold_qdmap[rows_gold[i-1][0]] = cur_list
            gold_docnames[rows_gold[i-1][0]] = cur_docs   #Current row corresponds to a different query, store the previous query details
            cur_list = []
            cur_docs = []                               
            prev = rows_gold[i][0]
            cur_list.append(rows_gold[i][1:3])
            cur_docs.append(rows_gold[i][1])

    gold_qdmap[rows_gold[i][0]] = cur_list
    gold_docnames[rows_gold[i-1][0]] = cur_docs           #Store details for the last query

    rel_files = []                        #File names of relevant documents for each query
    nonrel_files = []                     #File names of non-relevant documents for each query

    for i in range(len(file_names)):
        
        query_num = file_names[i][0][0]
        rel_files.append([])
        nonrel_files.append([])
        if len(gold_docnames[query_num]) == 0:           #Query missing in gold standard, relevant and non-relevant sets considered null
            continue
        
        for j in range(len(file_names[i])):
            doc = file_names[i][j][1]
            if doc in gold_docnames[query_num]:
                idx = gold_docnames[query_num].index(doc)
                if gold_qdmap[query_num][idx][1] == '2':
                    rel_files[i].append(file_names[i][j])       #Relevance judgement score = 2, so a relevant document
                else:
                    nonrel_files[i].append(file_names[i][j])    #Relevacne judgement score = 1, non-relevant 
            else:
                nonrel_files[i].append(file_names[i][j])        #Document not in gold-standard, non-relevant 
                    
    all_files = []
    all_files.append(rel_files)
    all_files.append(nonrel_files)
    return all_files

#------Function to obtain query vectors for all the queries-------

def get_query_vectors(queries, df, num_docs, term_idx):

    query_vecs = []                   #List to store the query vectors for all the queries

    for i in range(len(queries)):
        counter = Counter(queries[i][1:])                 #Store a map between each term and its frequency in the query
        curvec = np.zeros(len(term_idx.keys()))           #Vector for present query
                          
        for key,value in term_idx.items():                #Iterate over the whole vocabulary
            
            if key in queries[i][1:]:                     #Check if current term is in the query
                tf = 1 + math.log10(counter[key])
                cur_df = math.log10(num_docs/df[key])     
                curvec[value] = (tf*cur_df)

            else:
                curvec[value] = 0                         #Term not present in the query

        query_vecs.append(curvec/la.norm(curvec))         #Normalise the query vector 

    return query_vecs

#------Function to obtain centroid of a set of documents-------

def get_centroid(doc_vecs):

    centroid = np.zeros(len(doc_vecs[0]))
        
    for i in range(len(centroid)):
        sum = 0
        for j in range(len(doc_vecs)):
            sum += doc_vecs[j][i]
        avg = sum/(len(doc_vecs))
        centroid[i] = avg

    return centroid

#------Function to obtain centroid of a set of relevant or non-relevant documents for each query-------

def get_all_centroids(queries, doc_idx, all_doc_words, inverted_idx, df, term_idx):

    centroid_list = []                     #Stores document set centroids for each query
    
    for i in range(len(queries)):
        #print("Query ",i)
        if len(doc_idx[i]) == 0:
            centroid_list.append(np.zeros(len(term_idx.keys())))    #Centroid taken as zero vector if document set is null
            continue
        
        doc_vecs = []                       #List of normalised document vectors for current query
        for j in range(len(doc_idx[i])):
            cur_doc = doc_idx[i][j]
            doc_vec = get_doc_vectors(all_doc_words[cur_doc], cur_doc, df, inverted_idx, term_idx)
            doc_vec_normalised = doc_vec/(la.norm(doc_vec))
            doc_vecs.append(doc_vec_normalised)

        centroid = get_centroid(doc_vecs)
        centroid_list.append(centroid)
        
    return centroid_list


#-------Function to get the cosine similarity metric corresponding to all documents for each query, corresponding to both Relevance and     #-------Pseudo Relevance Feedback and for all possible alpha,beta,gamma values------

def get_scores_feedback(all_doc_words, query_vecs, df, inverted_idx, term_idx, centroids, params):

    sparse_query_vecs = []   #List of only the non zero entries of a query vector
    
    all_query_norms_psrf = []   #List of L2-norms for all the queries for RF (all 3 alpha,beta,gamma combinations)
    all_query_norms_rf = []     #List of L2-norms for all the queries for PsRF (all 3 alpha,beta,gamma combinations)
    all_scores_psrf = []   #Mapping between a query number/index and its list of cosine similarity scores for all documents for PsRF
    all_scores_rf = []     #Mapping between a query number/index and its list of cosine similarity scores for all documents for PsRF

    #centroids[0] stores centroids of relevant documents of all queries for PsRF
    #centroids[1] stores centroids of relevant documents of all queries for RF
    #centroids[2] stores centroids of non-relevant documents of all queries for RF
    #params[0] stores alpha values, params[1] stores beta values and params[2] stores gamma values
    
    #Store the L2-norms of modified queries for both RF and PsRF
    
    for i in range(len(params[0])):
        all_scores_psrf.append(dict())
        all_scores_rf.append(dict())
        alpha = params[0][i]
        beta = params[1][i]
        gamma = params[2][i]
        query_norms_psrf = []
        query_norms_rf = []
        
        for j in range(len(query_vecs)):
            all_scores_psrf[i][j] = []
            all_scores_rf[i][j] = []
            query_norms_psrf.append(la.norm(alpha*query_vecs[j] + beta*centroids[0][j]))
            query_norms_rf.append(la.norm(alpha*query_vecs[j] + beta*centroids[1][j] - gamma*centroids[2][j]))
            
        all_query_norms_psrf.append(query_norms_psrf)
        all_query_norms_rf.append(query_norms_rf)

    allzero_chk_r = []                    #Checks if relevant document set for a query is empty(0) or not(1) for RF
    allzero_chk_nr = []                   #Checks if relevant document set for a query is empty(0) or not(1) for PsRF
    
    for j in range(len(query_vecs)):      #Compute and store the sparse query vectors
        cur_sparse = {}
        allzero_chk_r.append(0)
        allzero_chk_nr.append(0)

        for i in range(len(query_vecs[j])):
            if query_vecs[j][i] != 0:
                cur_sparse[i] = query_vecs[j][i]
            if centroids[1][j][i] != 0:
                allzero_chk_r[j] = 1     
            if centroids[2][j][i] != 0:
                allzero_chk_nr[j] = 1
                
        sparse_query_vecs.append(cur_sparse)
                
         
    for i in range(len(all_doc_words)):              #Iterate over all the documents
        
        if len(all_doc_words[i]) == 0:               #For empty documents, store a score of 0 for all queries 
            for param_num in range(len(params[0])):
                for j in range(len(query_vecs)):
                    all_scores_psrf[param_num][j].append(0)
                    all_scores_rf[param_num][j].append(0)
            continue
        
        info = get_sparse_doc_vector(all_doc_words[i],i,df,inverted_idx,term_idx)   #Get the vector and its L2-norm for current document
        
        cur_sparse = info[0]
        doc_vec = info[1]
        cur_doc_norm = info[2]
        
        for j in range(len(query_vecs)):                #Iterate over all queries
            dot_pdtq = 0                                #Dot product of current document with original query
            dot_pdtd_psrf = 0                           #Dot product of current document with centroid of relevant documents for PsRF
            dot_pdtd_rf_rel = 0                         #Dot product of current document with centroid of relevant documents for RF
            dot_pdtd_rf_nonrel = 0                      #Dot product of current document with centroid of non-relevant documents for RF
            
            #For dot product with original query, sparse query vector is used
            #For all other dot products, sparse document vector is used
            
            for key,value in sparse_query_vecs[j].items():
                dot_pdtq = dot_pdtq + (value*doc_vec[key])                 

            for key,value in cur_sparse.items():
                dot_pdtd_psrf = dot_pdtd_psrf + (value*centroids[0][j][key])

            if allzero_chk_r[j] != 0:
                for key,value in cur_sparse.items():
                    dot_pdtd_rf_rel = dot_pdtd_rf_rel + (value*centroids[1][j][key])

            if allzero_chk_nr[j] != 0:
                for key,value in cur_sparse.items():
                    dot_pdtd_rf_nonrel = dot_pdtd_rf_nonrel + (value*centroids[2][j][key])        
                            
            #Computation of cosine similarity score in accordance with the equation in Rocchio's algorithm
            for param in range(len(params[0])):
                dot_pdt = params[0][param]*dot_pdtq + params[1][param]*dot_pdtd_psrf
                all_scores_psrf[param][j].append(dot_pdt/(cur_doc_norm*all_query_norms_psrf[param][j]))
                
                dot_pdt = params[0][param]*dot_pdtq + params[1][param]*dot_pdtd_rf_rel - params[2][param]*dot_pdtd_rf_nonrel
                all_scores_rf[param][j].append(dot_pdt/(cur_doc_norm*all_query_norms_rf[param][j]))

    all_scores = [all_scores_psrf, all_scores_rf]            
    return all_scores


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

def create_file(queries, scores, doc_ids, scheme, param_pos):

    csv_output = 'PB_22_ranked_list_' + scheme + '_' + param_pos + '.csv'     #File name
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
    doc_names = [os.path.basename(file_path) for file_path in doc_paths]   #Store the corresponding file names from the paths

    print("\nNumber of documents = ",len(doc_paths))
    
    gold_std = sys.argv[3]             #Path to gold standard ranked list
    ranked_list = sys.argv[4]          #Path to ranked list obtained from Part A - Task 2 (lnc.ltc scheme)
    
    relevant_doc_names_psrf = get_docs_by_query_psrf(ranked_list)    #Obtaining relevant documents for each query for PsRF scheme
    all_doc_names_rf = get_docs_by_query_rf(ranked_list, gold_std)   #Obtaining relevant and non-relevant documents for RF scheme
    
    relevant_doc_names_rf = all_doc_names_rf[0]                #Relevant documents for RF
    non_relevant_doc_names_rf = all_doc_names_rf[1]            #Non-relevant documents for PsRF

    docs_idx = {}                               #Stores a mapping from a document name to its corresponding index in the document list
    for i in range(len(doc_names)):
        docs_idx[doc_names[i]] = i

    relevant_docs_idx_psrf = []                #Stores the indices in document list for relevant documents from their names in PsRF scheme 
    for i in range(len(relevant_doc_names_psrf)):
        cur_idx = set()
        for j in range(len(relevant_doc_names_psrf[i])):
            cur_idx.add(docs_idx[relevant_doc_names_psrf[i][j][1]])
        relevant_docs_idx_psrf.append(list(cur_idx))
        
    relevant_docs_idx_rf = []                #Stores the indices in document list for relevant documents from their names in RF scheme 
    for i in range(len(relevant_doc_names_rf)):
        cur_idx = set()
        for j in range(len(relevant_doc_names_rf[i])):
            cur_idx.add(docs_idx[relevant_doc_names_rf[i][j][1]])
        relevant_docs_idx_rf.append(list(cur_idx))

    non_relevant_docs_idx_rf = []          #Stores the indices in document list for non-relevant documents from their names in RF scheme
    for i in range(len(non_relevant_doc_names_rf)):
        cur_idx = set()
        for j in range(len(non_relevant_doc_names_rf[i])):
            cur_idx.add(docs_idx[non_relevant_doc_names_rf[i][j][1]])
        non_relevant_docs_idx_rf.append(list(cur_idx))
        

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


    term_list = sorted(inverted_idx.keys())                     #Store the vocabulary terms in a list
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
    
    query_file_path = sys.argv[5]                        #Path to text file containing the pre-processed queries
    
    with open(query_file_path,'r') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    queries = [list(line.split(" ")) for line in lines]   #Store the queries as list of terms in each query
    
    alpha = [1,0.5,1]
    beta = [1,0.5,0.5]
    gamma = [0.5,0.5,0]

    print("\nObtaining query vectors...")
    query_vectors = get_query_vectors(queries, df,len(doc_paths),term_idx)
    all_new_query_vecs = []
    
    print("Obtaining document set centroids for each query...")
    print("Centroids of relevant documents in PsRF...")
    centroid_list_psrf = get_all_centroids(query_vectors, relevant_docs_idx_psrf, doc_contents, inverted_idx, df, term_idx)
    
    print("Centroids of relevant documents in RF...")
    centroid_list_rf_rel = get_all_centroids(query_vectors, relevant_docs_idx_rf, doc_contents, inverted_idx, df, term_idx)
    
    print("Centroids of non-relevant documents in RF...")
    centroid_list_rf_nonrel = get_all_centroids(query_vectors, non_relevant_docs_idx_rf, doc_contents, inverted_idx, df, term_idx)

    centroid_list = [centroid_list_psrf, centroid_list_rf_rel, centroid_list_rf_nonrel]
    
    print("Obtaining document vectors and Scoring...")                                                                                  
    all_scores = get_scores_feedback(doc_contents, query_vectors, df, inverted_idx, term_idx, centroid_list, [alpha, beta, gamma])
    
    print("Writing results to file...")
    print("PsRF...")
    scheme = "PsRF"
    for i in range(len(all_scores[0])):
        print("Alpha = ",alpha[i]," Beta = ",beta[i])
        create_file(queries, all_scores[0][i], doc_names, scheme, str(i))

    print("RF...")
    scheme = "RF"
    for i in range(len(all_scores[1])):
        print("Alpha = ",alpha[i]," Beta = ",beta[i], "Gamma = ",gamma[i])
        create_file(queries, all_scores[1][i], doc_names, scheme, str(i))
        
    
if __name__ == '__main__':
    main()
