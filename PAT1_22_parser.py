from bs4 import BeautifulSoup
import sys


def main():
    queries_file_path = sys.argv[1]
    print(queries_file_path)
    with open(queries_file_path, 'r') as file:
        query_data = file.read().replace('\n', '')
        soup = BeautifulSoup(query_data, features="lxml")

        nums = [u.string for u in soup.find_all('num')]
        titles = [u.string for u in soup.find_all('title')]

        open('./queries_22.txt', 'w').close()
        parsed_queries = open('./queries_22.txt', 'a')
        for i in range(len(nums)):
            parsed_queries.write(nums[i] + ' ' + titles[i] + '\n')
        parsed_queries.close()


if __name__ == '__main__':
    main()
