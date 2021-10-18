from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sys
import re


def pre_process_text(text):
    if len(text) == 0:
        return ""
    tokens = word_tokenize(str(text))
    new_tokens = []
    for token in tokens:
        modified_token = re.sub('[^a-zA-Z0-9]', '', token)
        if len(modified_token) > 1:
            new_tokens.append(modified_token)

    stop_words = set(stopwords.words('english'))
    removed_stopwords = [
        word for word in new_tokens if word.lower() not in stop_words
    ]

    wnl = WordNetLemmatizer()
    lemmatized_tokens = [
        wnl.lemmatize(word).lower() for word in removed_stopwords
    ]

    return ' '.join(lemmatized_tokens)


def main():
    queries_file_path = sys.argv[1]
    with open(queries_file_path, 'r') as file:
        query_data = file.read().replace('\n', '')
        soup = BeautifulSoup(query_data, features="lxml")

        nums = [u.string for u in soup.find_all('num')]
        titles = [u.string for u in soup.find_all('title')]

        open('./queries_22.txt', 'w').close()
        parsed_queries = open('./queries_22.txt', 'a')
        for i in range(len(nums)):
            parsed_queries.write(nums[i] + ' ' + pre_process_text(titles[i]) +
                                 '\n')
        parsed_queries.close()


if __name__ == '__main__':
    main()
