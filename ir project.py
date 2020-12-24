# import matplotlib.pyplot as plt            
# import pandas as pd
# from nltk.stem import PorterStemmer        
from nltk.tokenize import TweetTokenizer 
from nltk.tokenize import word_tokenize

# return tokens from sting (query,files) with small letter

# test retokenizer function
me_text=input("enter text \n")
me_text_tokins=rtokenize(me_text)
print(me_text_tokins)