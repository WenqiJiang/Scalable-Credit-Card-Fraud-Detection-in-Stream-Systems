from __future__ import print_function

import sys
import numpy as np

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

def raw_return(x):
    return x[0]

def type_return(x):
    return int(x)

def np_return(x):
    # arr = np.frombuffer(x.encode())
    # return arr
    # return x[1:-1].split()
    return np.array(x[1:-1].split(), dtype=np.float64)

def str_return(x):
    return str(x[0])

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage: network_wordcount.py <hostname> <port>", file=sys.stderr)
    #     sys.exit(-1)
    sc = SparkContext(appName="PythonStreamingNetworkWordCount")
    ssc = StreamingContext(sc, 1)

    # lines = ssc.socketTextStream('localhost', 8888)
    lines = ssc.socketTextStream("localhost", 9988)
    raw = lines.map(raw_return)
    typ = lines.map(type_return)
    np_arr = lines.map(np_return)

    np_arr.pprint()
    # typ.pprint()

    ssc.start()
    ssc.awaitTermination()