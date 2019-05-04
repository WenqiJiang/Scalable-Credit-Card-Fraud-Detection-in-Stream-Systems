from __future__ import print_function

import sys

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
    #     sys.exit(-1)
    sc = SparkContext(appName="PythonStreamingNetworkWordCount")
    ssc = StreamingContext(sc, 1)

    # lines = ssc.socketTextStream('localhost', 8888)
    lines = ssc.socketTextStream("localhost", 9997)
    lines.pprint()
    counts = lines.flatMap(lambda line: line.split(" "))\
                  .map(lambda word: (word, 1))\
                  .reduceByKey(lambda a, b: a+b)
    lines.pprint()

    ssc.start()
    ssc.awaitTermination()