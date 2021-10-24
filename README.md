# IR-TermProject
Group Number: 22
Group Members:
1. Arnab Kumar Mallick - 18CH10011
2. Arghyadeep Bandyopadhyay - 18EE10012
3. Subham Karmakar - 18EE10067
4. Sankalp Srivastava - 18EE10069

Task 1A:
Generates pickle file and indexed documents file. The pickle file stores
doc-ids instead of actual doc-path. The doc-ids are the indices of that
particular document in the file - indexed_docs.txt. The pickle file contains a
dictinary with terms as keys and a list of doc-id and term-frequency as its
values, i.e. [doc-id, term frequency].

Task 1B:
Generates queries file.

Task 1C:
Logic and Algorithm:
1. Considered the tokens from the queries as they are joined by 'AND' logic.
2. Used the trivial merge algorithm(for 'AND' logic) for the boolean retrieval dicussed in the class, for merging the postings lists for the specific tokens obtained from the queries above. 

Assumption:
indexed_docs.txt file must be present in the root directory of the project.

Task 2A  (TF-IDF Vectorization)

Performed by Arghyadeep Bandyopadhyay (Roll No: 18EE10012)

Steps used:

1. The paths and file names of all the documents are extracted fromt the ‘en_BDNews24’ folder
2. The inverted index file is read and the document frequencies (df) along with the vocabulary are stored
3. For each document, the document text is stored as a list of terms using the inverted index.
4. The text file containing the queries is read and each query is then stored as a list of the terms contained in that query
5. The |V|-dimensional TF-IDF vectors are obtained for each query with a given weighting scheme
6. For each document, at first the |V|-dimensional TF-IDF vector is obtained with a given weighting scheme. Then, for each query vector, the value of the cosine similarity metric with normalization between the query vector and the current document vector is computed. The process is repeated for all the documents and the values of the cosine similarity metric for each query-document pair is stored.
7. For each query, the cosine similarity scores are sorted in descending order. The top 50 documents are then stored in a 2-column csv file in the format <query ID> : <document ID>.
8. Steps 5 to 7 are performed for three ddd.qqq schemes, namely scheme ‘A’ (lnc.ltc), scheme ‘B’ (Lnc.Lpc) and scheme ‘C’ (anc.apc)


Assumptions / Changes:

The term frequencies are not computed separately and are assumed to be stored in the inverted index itself.

Extra input / parameters:

The path to the “queries_<GROUP_NO>.txt is to be given along with the path to the inverted index file and the “en_BDNews24” folder. Here, GROUP_NO is 22.

To run Task 2A:
  
$>>python3 PAT2_22_ranker.py  <path to the en_BDNews24 folder> <path_to_model_queries_22.pth>  <path to queries_22.txt>

Python version used: 3.6.8

Library Requirements:

1. os                                                  4. numpy
2. sys                                                 5. math
3. pickle                                              6. csv
7. collections


Task 2B:
Logic:
1. Parsed the data of generated Ranked list A, B, C; Gold standard ranked lists and Queries in an array
2. Maintained a document that maps the query id with the relevant documents and relevance score
3. We then compared which documents in our ranked list is present in the Gold Standard list and calculated the Precision@K and Average Precision.
4. The relevance of the documents is then stored in an array and compared to the sorted array of the relevance scores. This way, we calculate the NDCG.
5. We then average over all the queries to find out the average parameters.

Assumptions:
1. The Data folder contains rankedRelevantDocList.csv
2. queries_22.txt must be present in the root directory of the project.
3. PAT2_22_ranked_list<K>.csv must be present in the root directory of the project.

To run Task 2B:<br/>
$>> python PAT2_22_evaluator.py ./Data/rankedRelevantDocList.csv PAT2_22_ranked_list_A.csv<br/>
$>> python PAT2_22_evaluator.py ./Data/rankedRelevantDocList.csv PAT2_22_ranked_list_B.csv<br/>
$>> python PAT2_22_evaluator.py ./Data/rankedRelevantDocList.csv PAT2_22_ranked_list_C.csv

The output will be generated in the root directory<br/>
PAT2_22_metrics_A.csv<br/>
PAT2_22_metrics_B.csv<br/>
PAT2_22_metrics_C.csv

Python version used: 3.9.7

Library Requirements:
1. sys
2. csv
3. numpy
4. tabulate
5. pickle
5. pickle
6. bs4
7. re
8. os
9. pickle
