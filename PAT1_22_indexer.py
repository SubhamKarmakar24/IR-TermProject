from bs4 import BeautifulSoup
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import pickle
import re
import sys


def main():
    walk_dir = sys.argv[1]
    docs_to_be_indexed = []
    for root, subdirs, files in os.walk(walk_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            docs_to_be_indexed.append(file_path)
    docs_to_be_indexed.sort()
    doc_map = dict()
    open('./indexed_docs.txt', 'w').close()
    with open('./indexed_docs.txt', 'a') as docs_file:
        for i in range(len(docs_to_be_indexed)):
            docs_file.write(docs_to_be_indexed[i] + '\n')
            doc_map[docs_to_be_indexed[i]] = i

    indexer = dict()

    for i in range(len(docs_to_be_indexed)):
        doc_file = docs_to_be_indexed[i]
        doc_no = i
        with open(doc_file, 'r') as file:
            new_doc_file = file.read().replace('\n', '')
        soup1 = BeautifulSoup(new_doc_file, features="lxml")

        text_temp = [u.string for u in soup1.find_all('text')]

        if len(text_temp) == 0:
            print(file0)
            continue

        text_in_file = str(text_temp[0])

        tokens = word_tokenize(text_in_file)
        new_tokens = []
        for token in tokens:
            modified_token = re.sub('[^a-zA-Z]', '', token)
            if len(modified_token) > 1:
                new_tokens.append(modified_token)

        stop_words = set(stopwords.words('english'))
        removed_stopwords = [
            word for word in new_tokens if word.lower() not in stop_words
        ]

        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in removed_stopwords]

        terms_set = set()
        for term in stemmed_tokens:
            terms_set.update(term)

        for term in terms_set:
            if term in indexer:
                indexer[term].append(doc_no)
            else:
                indexer[term] = [doc_no]

    pickle_out = open('model_queries_22.pth', 'wb')
    pickle.dump(indexer, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    main()
