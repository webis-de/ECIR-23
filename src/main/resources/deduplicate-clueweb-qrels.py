from trectools import TrecQrel
from deduplication_util import ClueWebQrelDeduplication

if __name__ == '__main__':
    qrel_files = ['qrels.inofficial.web.1-50-with-duplicates.txt', 'qrels.web.101-150-with-duplicates.txt',
                  'qrels.web.151-200-with-duplicates.txt', 'qrels.web.201-250-with-duplicates.txt',
                  'qrels.web.251-300-with-duplicates.txt', 'qrels.web.51-100-with-duplicates.txt'
                  ]

    for qrel_file in qrel_files:
        qrel_file = 'src/main/resources/unprocessed/topics-and-qrels/' + qrel_file
        qrel = TrecQrel(qrel_file)
        qrel = ClueWebQrelDeduplication().deduplicated(qrel)

        qrel.qrels_data.to_csv(qrel_file.replace('-with-duplicates', ''), index=False, header=False, sep=' ')
