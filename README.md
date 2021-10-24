# IR-TermProject
Group Number: 22
Group Members:
1. Arnab Kumar Mallick - 18CH10011
2. Arghyadeep Bandyopadhyay - 18EE10012
3. Subham Karmakar - 18EE10067
4. Sankalp Srivastava - 18EE10069

Task 1C:
Logic and Algorithm:
1. Considered the tokens from the queries as they are joined by 'AND' logic.
2. Used the trivial merge algorithm(for 'AND' logic) for the boolean retrieval dicussed in the class, for merging the postings lists for the specific tokens obtained from the queries above. 

Task 2B:
Logic:
1. Parsed the data of generated Ranked list A, B, C; Gold standard ranked lists and Queries in an array
2. Maintained a document that maps the query id with the relevant documents and relevance score
3. We then compared which documents in our ranked list is present in the Gold Standard list and calculated the Precision@K and Average Precision.
4. The relevance of the documents is then stored in an array and compared to the sorted array of the relevance scores. This way, we calculate the NDGC.
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




