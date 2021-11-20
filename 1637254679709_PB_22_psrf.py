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
        
    #Precalculating certain parameters needed for the different weighting schemes
    
    curvec = np.zeros(len(term_idx.keys()))       #Vector for the present document
        
    for word in doc_words:
        tf = 1 + math.log10(counts[word])
        curvec[term_idx[word]] = tf                #As value of document frequency component is 'n' in each scheme, so df=1
        
    return curvec

def get_sparse_doc_vector(doc_words, doc_no, df, inverted_idx, term_idx):

    counts = {}           #Dictionary mapping each term to its frequency in the document 
    
    for word in doc_words:
        temp = inverted_idx[word]
        counts[word] = temp[doc_no]
        
    #Precalculating certain parameters needed for the different weighting schemes
    
    wholevec = np.zeros(len(term_idx.keys()))       #Vector for the present document
    curvec = {}
    result = []
    
    for word in doc_words:
        tf = 1 + math.log10(counts[word])
        curvec[term_idx[word]] = tf                #As value of document frequency component is 'n' in each scheme, so df=1
        wholevec[term_idx[word]] = tf                #As value of document frequency component is 'n' in each scheme, so df=1
        
    #curvec[len(term_idx.keys())+1] = la.norm(wholevec)
    result.append(curvec)
    result.append(wholevec)
    result.append(la.norm(wholevec))
    return result


def get_docs_by_query(ranked_list):

    file_names = []
    rows = []
    
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


def get_query_vectors(queries, df, scheme, num_docs, term_idx):

    query_vecs = []                   #List to store the query vectors for all the queries

    for i in range(len(queries)):
        counter = Counter(queries[i][1:])                               #Store a map between each term and its frequency in the query
        #tf_list = np.array([value for key,value in counter.items()])    #Store the list of term frequencies

        curvec = np.zeros(len(term_idx.keys()))           #Vector for present query
                          
        for key,value in term_idx.items():                #Iterate over the whole vocabulary
            
            if key in queries[i][1:]:                     #Check if current term is in the query
                tf = 1 + math.log10(counter[key])
                cur_df = math.log10(num_docs/df[key])     
                    
                curvec[value] = (tf*cur_df)

            else:
                curvec[value] = 0                   #Term not present in the query

        query_vecs.append(curvec/la.norm(curvec))

    return query_vecs


def psrf(queries, relevant_docs, alpha, beta, indexer, total_docs):

    #total_docs = len(doc_id)
    total_relevant_docs = len(relevant_docs)
    new_queries = []
    
    for i in range(len(queries)):
        print("Query ",i)
        relevant_terms = dict()
        #query_id = int(queries[i][0])
        query_terms = Counter(queries[i][1:])

        for term, term_freq in query_terms.items():
            if term not in indexer.keys():
                continue
            tf = 1 + math.log10(term_freq)
            idf = math.log10(total_docs / len(indexer[term].keys()))
            relevant_terms[term] = alpha * tf * idf
            
        for term, docs in indexer.items():
            for doc_id, term_freq in docs.items():
                if doc_id not in relevant_docs:
                    continue
                tf = 1 + math.log10(term_freq)
                idf = 1
                if term not in relevant_terms:
                    relevant_terms[term] = (beta * tf * idf) / total_relevant_docs
                else:
                    relevant_terms[term] += (beta * tf * idf) / total_relevant_docs
        #relevant_terms['query_id'] = query_id

        modified_vector = list()
        for term in sorted(indexer.keys()):
            if term not in relevant_terms:
                modified_vector.append(0)
            else:
                modified_vector.append(relevant_terms[term])
                
        new_queries.append(modified_vector)
    
    return new_queries


def get_centroid(doc_vecs):

    centroid = np.zeros(len(doc_vecs[0]))
    #sparse_vec = []
    
    for i in range(len(centroid)):
        sum = 0
        for j in range(len(doc_vecs)):
            sum += doc_vecs[j][i]
        avg = sum/(len(doc_vecs))
        centroid[i] = avg

    return centroid

def get_all_centroids(queries, relevant_doc_idx, all_doc_words, inverted_idx, df, term_idx):

    centroid_list = []
    for i in range(len(queries)):
        print("Query ",i)
        doc_vecs = []
        for j in range(len(relevant_doc_idx[i])):
            cur_doc = relevant_doc_idx[i][j]
            doc_vec = get_doc_vectors(all_doc_words[cur_doc], cur_doc, df, inverted_idx, term_idx)
            doc_vec_normalised = doc_vec/(la.norm(doc_vec))
            doc_vecs.append(doc_vec_normalised)

        centroid = get_centroid(doc_vecs)
        centroid_list.append(centroid)
        
    return centroid_list


def pseudo_relevance_feedback(queries, relevant_doc_idx, params, all_doc_words, inverted_idx, df, term_idx):

    total_relevant_docs = len(relevant_doc_idx)
    alpha = params[0]
    beta = params[1]
    new_queries = []

    for i in range(len(queries)):
        print("Query ",i)
        doc_vecs = []
        for j in range(len(relevant_doc_idx[i])):
            cur_doc = relevant_doc_idx[i][j]
            doc_vec = get_doc_vectors(all_doc_words[cur_doc], cur_doc, df, inverted_idx, term_idx)
            doc_vec_normalised = doc_vec/(la.norm(doc_vec))
            doc_vecs.append(doc_vec_normalised)

        centroid = get_centroid(doc_vecs)
        #queries[i] = queries[i]/(la.norm(queries[i]))
        cur_query = alpha*queries[i] + beta*centroid
        new_queries.append(cur_query)

    return new_queries    

#-------Function to get the cosine similarity metric corresponding to all documents for each query

def get_scores(all_doc_words, all_query_vecs, df, scheme, inverted_idx, term_idx):

    scores = {}              #Mapping between a query number/index and its list of cosine similarity scores for all documents             
    query_norms = []         #List of L2-norms for all the queries 
    sparse_query_vecs = []   #List of only the non zero entries of a query vector

    all_scores = []
    all_query_norms = []
    
    for i in range(len(all_query_vecs)):
        all_scores.append(dict())
       
        for j in range(len(all_query_vecs[i])):       #Compute and store the norms and the sparse query vectors
            scores[j] = []
            all_scores[i][j] = []
            query_norms.append(la.norm(all_query_vecs[i][j]))
        
            '''cur_sparse = {}
            for k in range(len(all_query_vecs[i][j])):
              if all_query_vecs[i][j][k] != 0:
                cur_sparse[k] = all_query_vecs[i][j][k]      
                
            sparse_query_vecs.append(cur_sparse)
            print("Query ",j," len = ",len(cur_sparse))'''
            
        all_query_norms.append(query_norms)
        
    print_idx = range(0,len(all_doc_words),1000)

    for i in range(len(all_doc_words)):              #Iterate over all the documents
        
        if i in print_idx:
            print("Doc ",i)
        if len(all_doc_words[i]) == 0:               #For empty documents, store a score of 0 for all queries 
            for param_num in range(len(all_query_vecs)):
                for j in range(len(all_query_vecs[param_num])):
                    all_scores[param_num][j].append(0)
            continue
        
        #doc_vec = get_doc_vectors(all_doc_words[i],i,df,inverted_idx, term_idx)
        cur_sparse = get_sparse_doc_vector(all_doc_words[i],i,df,inverted_idx, term_idx)
        
        cur_doc_norm = cur_sparse[len(term_idx.keys())+1]        #Get document vector and its L2-norm for current document         
        cur_sparse.pop(len(term_idx.keys())+1)
        
        for j in range(len(all_query_vecs[0])):                #Iterate over all queries
            dot_pdt1 = 0
            dot_pdt2 = 0
            dot_pdt3 = 0
            for key,value in cur_sparse.items():
                    dot_pdt1 = dot_pdt1 + (value*all_query_vecs[0][j][key])
                    dot_pdt2 = dot_pdt2 + (value*all_query_vecs[1][j][key])
                    dot_pdt3 = dot_pdt3 + (value*all_query_vecs[2][j][key])
            #Compute dot product between current query and current document
            all_scores[0][j].append(dot_pdt1/(cur_doc_norm*all_query_norms[0][j]))   #Normalize and append score
            all_scores[1][j].append(dot_pdt2/(cur_doc_norm*all_query_norms[1][j]))
            all_scores[2][j].append(dot_pdt3/(cur_doc_norm*all_query_norms[2][j]))
            
    return all_scores


def get_scores_new(all_doc_words, query_vecs, df, inverted_idx, term_idx, centroids, params):

    scores = {}              #Mapping between a query number/index and its list of cosine similarity scores for all documents             
    #query_norms = []         #List of L2-norms for all the queries 
    sparse_query_vecs = []   #List of only the non zero entries of a query vector

    all_scores = []
    all_query_norms = []
    all_query_vecs = []

    '''for i in range(len(params[0])):
        alpha = params[0][i]
        beta = params[1][i]
        temp_queries = []
        for j in range(len(query_vecs)):
            temp_queries.append(alpha*query_vecs[j] + beta*centroids[j])
        all_query_vecs.append(temp_queries)    
    '''
    for i in range(len(params[0])):
        all_scores.append(dict())
        alpha = params[0][i]
        beta = params[1][i]
        query_norms = []
        for j in range(len(query_vecs)):
            all_scores[i][j] = []
            #query_norms.append(la.norm(all_query_vecs[i][j]))
            query_norms.append(la.norm(alpha*query_vecs[j] + beta*centroids[j]))
        all_query_norms.append(query_norms)    
       
    for j in range(len(query_vecs)):       #Compute and store the norms and the sparse query vectors
        scores[j] = []
        cur_sparse = {}
        for i in range(len(query_vecs[j])):
            if query_vecs[j][i] != 0:
                cur_sparse[i] = query_vecs[j][i]      
                
        sparse_query_vecs.append(cur_sparse)
        #print("Query ",j," len = ",len(cur_sparse))
            
         
    print_idx = range(0,len(all_doc_words),2000)

    for i in range(len(all_doc_words)):              #Iterate over all the documents
        
        if i in print_idx:
            print("Doc ",i)
        if len(all_doc_words[i]) == 0:               #For empty documents, store a score of 0 for all queries 
            for param_num in range(len(params[0])):
                for j in range(len(query_vecs)):
                    all_scores[param_num][j].append(0)
            continue
        
        #doc_vec = get_doc_vectors(all_doc_words[i],i,df,inverted_idx, term_idx)
        info = get_sparse_doc_vector(all_doc_words[i],i,df,inverted_idx,term_idx)
        
        #cur_doc_norm = cur_sparse[len(term_idx.keys())+1]        #Get document vector and its L2-norm for current document         
        #cur_sparse.pop(len(term_idx.keys())+1)
        cur_sparse = info[0]
        doc_vec = info[1]
        cur_doc_norm = info[2]
        
        for j in range(len(query_vecs)):                #Iterate over all queries
            dot_pdtq = 0
            dot_pdtd = 0
            for key,value in sparse_query_vecs[j].items():
                dot_pdtq = dot_pdtq + (value*doc_vec[key])                 #Compute dot product between current query and current document
            for key,value in cur_sparse.items():
                dot_pdtd = dot_pdtd + (value*centroids[j][key])
                
            for param in range(len(params[0])):
                dot_pdt = params[0][param]*dot_pdtq + params[1][param]*dot_pdtd
                all_scores[param][j].append(dot_pdt/(cur_doc_norm*all_query_norms[param][j]))
                
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

    csv_output = 'PB_psrf_22_ranked_list_' + scheme + '_' + param_pos + '.csv'     #File name
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
    
    ranked_list = sys.argv[3]
    relevant_doc_names = get_docs_by_query(ranked_list)

    docs_idx = {}
    for i in range(len(doc_names)):
        docs_idx[doc_names[i]] = i

    relevant_docs_idx = []

    for i in range(len(relevant_doc_names)):
        cur_idx = set()
        for j in range(len(relevant_doc_names[i])):
            cur_idx.add(docs_idx[relevant_doc_names[i][j][1]])
        relevant_docs_idx.append(list(cur_idx))
        
    #print("Relevant docs indices: \n", relevant_docs_idx)
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

    
    #term_list = [key for key,value in inverted_idx.items()]     #Store the vocabulary terms in a list
    term_list = sorted(inverted_idx.keys())
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
    
    query_file_path = sys.argv[4]                        #Path to text file containing the pre-processed queries
    
    with open(query_file_path,'r') as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]

    queries = [list(line.split(" ")) for line in lines]   #Store the queries as list of terms in each query
    
    print("\nTF-IDF Vectorization...\n")
    #schemes = ['A','B','C']                #Symbols for the three schemes used for weighting-normalizing
    scheme = 'A'
    alpha = [1,0.5,1]
    beta = [1,0.5,0.5]
    gamma = [0.5,0.5,0]

    print("Obtaining query vectors...")
    query_vectors = get_query_vectors(queries, df,scheme,len(doc_paths),term_idx)
    all_new_query_vecs = []
    
    '''for i in range(3):
        
        print("\nAlpha = ",alpha[i]," Beta = ",beta[i])
        
        print("Modifying query vectors...")
        new_query_vectors = pseudo_relevance_feedback(query_vectors, relevant_docs_idx, [alpha[i], beta[i]], doc_contents, inverted_idx, df, term_idx)
        all_new_query_vecs.append(new_query_vectors)
    '''
    print("Obtaining all centroids...")
    centroid_list = get_all_centroids(query_vectors, relevant_docs_idx, doc_contents, inverted_idx, df, term_idx)
    
    print("Obtaining document vectors and Scoring...")                                                                                  
    #all_scores = get_scores(doc_contents, all_new_query_vecs, df,scheme,inverted_idx, term_idx)
    all_scores = get_scores_new(doc_contents, query_vectors, df, inverted_idx, term_idx, centroid_list, [alpha, beta])
    
    print("Writing results to file...")
    for i in range(len(all_scores)):
        print("Alpha = ",alpha[i]," Beta = ",beta[i])
        create_file(queries, all_scores[i], doc_names, scheme, str(i))

        
    
if __name__ == '__main__':
    main()
