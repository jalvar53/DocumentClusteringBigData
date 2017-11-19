import re
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from math import sqrt

if __name__ == "__main__":
    sc = SparkContext(appName="DocumentClustering")

    files_RDD = sc.wholeTextFiles("hdfs:///user/jalvar53/datasets/gutenberg/*")
    documents_names = files_RDD.keys()
    documents = files_RDD.values().map(lambda doc: re.split('[^a-zA-Z]+', doc))

    # From https://spark.apache.org/docs/2.1.1/mllib-feature-extraction.html#tf-idf
    hashingTF = HashingTF()
    tf = hashingTF.transform(documents)
    tf.cache()
    idf = IDF(minDocFreq=10).fit(tf)
    tfidf = idf.transform(tf)

    clusters = KMeans.train(tfidf, 2, maxIterations=10, initializationMode="random")

    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x ** 2 for x in (point - center)]))

    WSSSE = tfidf.map(lambda point: error(point)).reduce(lambda x, y: x + y)

    print("Within Set Sum of Squared Error = " + str(WSSSE))