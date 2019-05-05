from __future__ import print_function

import sys
import numpy as np

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

def raw_return(x):
    return np.float64(x)

def np_return(x):
    arr = np.array(x)
    return arr

def str_return(x):
    return str(x[0])

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
    #     sys.exit(-1)
    sc = SparkContext(appName="PythonStreamingNetworkWordCount")
    ssc = StreamingContext(sc, 1)

    # lines = ssc.socketTextStream('localhost', 8888)
    lines = ssc.socketTextStream("localhost", 9994)
    raw = lines.map(raw_return)
    arr = lines.map(np_return)

    raw.pprint()
    arr.pprint()

    ssc.start()
    ssc.awaitTermination()