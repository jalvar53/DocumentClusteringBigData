import re
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

if __name__ == "__main__":
    sc = SparkContext(appName="DocumentClustering")
    files_RDD = sc.wholeTextFiles("hdfs:///user/jalvar53/datasets/gutenberg/*")
    documents_names = files_RDD.keys()
    documents = files_RDD.values().map(lambda doc: re.split('[^a-zA-Z]+',doc))

    #From https://spark.apache.org/docs/2.1.1/mllib-feature-extraction.html#tf-idf
    hashingTF = HashingTF()
    tf = hashingTF.transform(documents)
    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)

    print(tfidf.take(2))