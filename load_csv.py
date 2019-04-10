import numpy as np
import pandas as pd


credit_card = np.genfromtxt('C:/Users/lming/OneDrive/Desktop/creditcard.csv', dtype='str', delimiter=',', skip_header=1)
credit_card = np.char.strip(credit_card, '"').astype(float)
#for i in range(20):
    #print(credit_card[i])

