import re
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF

if __name__ == "__main__":
    sc = SparkContext(appName="DocumentClustering")

    files_RDD = sc.wholeTextFiles("hdfs:///datasets/gutenberg-txt-es/19*.txt")
    documents_names = files_RDD.keys()
    documents = files_RDD.values().map(lambda doc: re.split('[^a-zA-Z]+', doc))

    # From https://spark.apache.org/docs/2.1.1/mllib-feature-extraction.html#tf-idf
    hashingTF = HashingTF()
    tf = hashingTF.transform(documents)
    tf.cache()
    idf = IDF(minDocFreq=10).fit(tf)
    tfidf = idf.transform(tf)

    clusters = KMeans.train(tfidf, 2, maxIterations=10, initializationMode="random")

    centers = clusters.predict(tfidf).collect()
    names = documents_names.collect()
    result = zip(names, centers)

    output = sc.parallelize(result)
    output.saveAsTextFile("/users/lgalle17/output.txt")

    sc.stop();