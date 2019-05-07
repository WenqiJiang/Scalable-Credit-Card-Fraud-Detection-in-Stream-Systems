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
    '''
    Return feed_freq (int), batch_size (int), list of a batch of transactions [(index (int), transaction(numpy.array))]
    '''
    # arr = np.frombuffer(x.encode())
    # return arr
    # return x[1:-1].split()
    # return np.array(x[1:-1].split(), dtype=np.float64)
    x_split = x.split()
    # return int(x_split[0]), int(x_split[1]), np.array(x_split[2:], dtype=np.float64)
    feed_freq = int(x_split[0])
    batch_size = int(x_split[1])

    x_split = x_split[2:]
    feedin_data = []
    for i in range(batch_size):
        index = int(x_split[i * 29])
        feature = np.array(x_split[i*29 + 1 : (i+1)*29], dtype=np.float64)
        feedin_data.append((index,feature))
    return feed_freq, batch_size, feedin_data

def str_return(x):
    return str(x[0])

if __name__ == "__main__":
    # if len(sys.argv) != 3:
    #     print("Usage:network_wordcount.py <hostname> <port>", file=sys.stderr)
    #     sys.exit(-1)
    sc = SparkContext(appName="PythonStreamingNetworkWordCount")
    ssc = StreamingContext(sc, 1)

    sc.setLogLevel('OFF')

    # lines = ssc.socketTextStream('localhost', 8888)
    lines = ssc.socketTextStream("localhost", 7017)
    raw = lines.map(raw_return)
    typ = lines.map(type_return)
    np_arr = lines.map(np_return)

    np_arr.pprint()
    # typ.pprint()

    ssc.start()
    ssc.awaitTermination()