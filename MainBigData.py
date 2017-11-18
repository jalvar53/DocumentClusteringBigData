import re
from pyspark import SparkContext, SparkConf

if __name__ == "__main__":
    sc = SparkContext(appName="DocumentClustering")
    files_RDD = sc.wholeTextFiles("/user/jalvar53/datasets/gutenberg")
    documents_names = files_RDD.keys()
    documents = documents_names.values().map(lambda document: re.split('\W+', document))
    print documents_names
    print documents
