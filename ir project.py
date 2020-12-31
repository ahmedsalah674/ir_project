#----------- imports---------
import nltk #to import tokenize            
from nltk.corpus import stopwords #to remove stopwords
from nltk.tokenize import word_tokenize #tokenize text and lowercacse it
import os
from os import path # to check if the file is there or not 
from nltk.stem import PorterStemmer # to stem tokiens
import time
import re
import numpy as np
import pandas as pd
import math
from collections import Counter

def read_env(env_path):
    with open(env_path, 'r', encoding="utf-8") as env:
        env_settings = env.read()
        env_settings = re.sub(' ','',env_settings)
        env_settings = env_settings.split('\n')
        env_settings = [setting.split('=') for setting in env_settings]
    return dict(env_settings)

def settings(env_settings):
    language = env_settings['language']
    path = env_settings['path']
    type_files = env_settings['type_files']
    number_files= int(env_settings['number_files'])
    download= bool(int(env_settings['download']))
    return language,path,type_files,number_files,download

def fileNames(path,type_str):
    file_names = [name for name in os.listdir(path) if name.endswith(type_str)]
    file_dirs=[]
    for name in file_names:
        file_dirs.append(path + '/' +name)
    return file_dirs,file_names

def scan_type_model(file_names,number_files):
    type_query=input('please enter 1 for Positional index model or 2 for Vector space model or 0 for exit: ')
    if (type_query=='1'):
        Positional_index_model()
    elif(type_query=='2'):
        sim=vector_space(number_files)
        for i in sim: print('sim of',file_names[i],'->','%.2f'% sim[i])
        scan_type_model(file_names,number_files)
    elif(type_query=='0'):
        print('thanks :)')
        time.sleep(3)
    else:
        print('\t\tsorry put you entered wrong number try again')
        scan_type_model(file_names,number_files)

def get_ls_steming_tokins():
    inquery = input("Please enter a query:\n")
    query_to = tokenize(inquery)
    query =Stemming(query_to)
    return query,query_to

def Stemming(tokins):
    stemmer = PorterStemmer() 
    reviews_stem = [] 
    reviews_stem = [stemmer.stem(word) for word in tokins]
    return reviews_stem 

def tokenize(query): # get text and return list of tokens withot stopwords and lowercase 
    stopword = stopwords.words(language) #get list of stop words in language
    tokens = word_tokenize(query,language=language)  #get list of tokens
    tokens_without_stop_word = [word.lower() for word in tokens if not word in stopword] # remove stopwords
    return  tokens_without_stop_word
################# pharse query #####################
def positional (tokens): # get tokens and return []
    pos_index = dict() 
    for index, term in enumerate(tokens): # index is int value for index and term is str value for token
        if term in pos_index:
            pos_index[term].append(index + 1)
        else:
            pos_index[term] = []
            pos_index[term].append(index + 1)
    return pos_index

def docID(plist): # get list and return number of doc in index 0 plist =[docID,[indices]]
    return plist[0]

def position(plist): # get list and return number of doc in index 0 plist =[docID,[indices]]
    return plist[1]

def pos_intersect(p1, p2, k): 
    answer = []
    len1 = len(p1)
    len2 = len(p2)
    i = j = 0
    while i != len1 and j != len2:  # p1 != null and p2 != null
        if docID(p1[i]) == docID(p2[j]):
            l = []  # l <- ()
            pp1 = position(p1[i])  # pp1 <- positions(p1) , p1[i]=[docID,[indices]]
            pp2 = position(p2[j])  # pp2 <- positions(p2) , p1[i]=[docID,[indices]]
            plen1 = len(pp1)
            plen2 = len(pp2)
            ii = jj = 0
            while ii != plen1:  # while (pp1 != null)
                while jj != plen2:  # while (pp2 != null)
                    if abs(pp1[ii] - pp2[jj]) == k:  # if (|index(pp1) - index(pp2)| == k) , k=1
                        l.append(pp2[jj]) 
                    elif pp2[jj] > pp1[ii]:  # index(pp2) > index(pp1)
                        break
                    jj += 1  # next(pp2)->int other index for second term
                answer.append([docID(p1[i]),  l])  #answer=[docID, index(first_term), ps_secondTerm]
                ii += 1  #  next(pp1)->int other index for first term
            i += 1  #  next(p1) ,p1=[docID,[indices]]
            j += 1  #  next(p2) ,p2=[docID,[indices]]
        elif docID(p1[i]) < docID(p2[j]):  # increace counter of term if his docID is smaller than the other
            i += 1  #  next(p1)
        else:
            j += 1  #  next(p2)
    return answer

def Positional_index_model():
    positions = dict()
    files=list()
    finalresult = list()
    query,query_to=get_ls_steming_tokins()
    # print('\n',query,'\n')
    for term in query:
        positions[term]=list()
        positions[term].append(0)
        positions[term].append([])
    file_dirs,file_names =fileNames(path,type_files)
    for i in range(0,len(file_dirs)):
        with open(file_dirs[i], 'r', encoding="utf-8") as doc:
            read_string = doc.read()
            tokens=tokenize(read_string)
            tokens=Stemming(tokens)
            pos_index=positional(tokens)
            for word in query:
                for pos, term in enumerate(pos_index):
                    if term == word:
                        positions[word][1].append([i, pos_index[word]])
                        positions[word][0]=positions[word][0]+1
    # print(positions)
    ra=0
    for term in positions:
        print(query_to[ra],' :',positions[term]) #dict {term: [term_freq,[[doc1,[indices]],[doc2,[indices]]]]}
        ra +=1
    # print('len(q)->',len(query)) 
    if len(query)!=1:

        i=0
        j=0
        while i < len(query):
            j=i+1
            if j<len(query):
                # send positions without freq positions[query][1]->[[doc1,[inindices],[doc2,[inindices],...]]
                files.append(pos_intersect(positions[query[i]][1], positions[query[j]][1], 1)) #get intersect of two terms in query
            i=i+1
        b = 1
        lfiles =len(files)  #files=[[[doc1,index],[doc2,index]],[[doc1,index],[doc2,index]],....]
        #  -###################     call pos_intersect on the output of the last loop 
        while b<lfiles: #loop N-1 time on intersect of lists 
            i=0
            j=0
            while i < len(files):#files=[[[doc1,index],[doc2,index]],[[doc1,index],[doc2,index]],....]
                j=i+1
                if j<len(files):
                    finalresult.append(pos_intersect(files[i], files[j], 1))
                i=i+1
            files=finalresult.copy()
            finalresult=list()
            b +=1
        #######################
        if files!=[[]]:
            docs=set([i[0] for fi in files for i in fi if i[1]!=[]])
            for i in docs:print(file_names[i])
            scan_type_model(file_names,number_files)
        else :
            print('this query dosen\'t match ')
            scan_type_model(file_names,number_files)
    else:
        # for i in positions[query[0]][1]:
            # print(i)
        files=[docID(i) for i in positions[query[0]][1]]
        if (files) !=[]:
            for i in set(files):print(file_names[i])
            scan_type_model(file_names,number_files)
        else :
            print('this query dosen\'t match ')
            scan_type_model(file_names,number_files)



##############################vector space ############
def tf_dict():
    doc_vocab  = dict()
    for i in range(0,len(file_dirs)):
        doc_vocab[i] = dict()
        with open(file_dirs[i], 'r', encoding="utf-8") as doc:
            read_string = doc.read()  
            tokens = tokenize(read_string)
            tokens = Stemming(tokens)
            # get dict of {doc_num : {word1 : word1_count, word2 : word2_count, .... }}
            for words in tokens:
                if words in doc_vocab[i]:
                    doc_vocab[i][words] += 1
                else:
                    doc_vocab[i][words] = 1

    term_pd = pd.DataFrame.from_dict(doc_vocab, orient='index')
    term_pd.fillna(0,inplace=True)
    term_pd.sort_index(inplace=True)
    term_pd.to_csv('tf.csv')
    return term_pd

def df_and_idf(number_files):
    term_tf=tf_dict()
    df_ls=term_tf[term_tf > 0].count()
    idf=np.log(number_files/df_ls.values)
    idf_dict=pd.Series(data=idf,index=df_ls.index).to_dict()
    return term_tf,df_ls,idf_dict

def tf_idf_fun(number_files,df_ls,term_tf):
    tf_idf=term_tf.copy()
    for term in term_tf:
        for doc in range(0,len(term_tf)):
            if term_tf[term][doc] == 0:
                tf_idf[term][doc] = 0
            else:
                tf_idf[term][doc] = (1 + np.log(term_tf[term][doc])) * np.log10(number_files/df_ls[term])
    tf_idf.to_csv('tf_idf.csv')
    return tf_idf

def normalize_data_fun(tf_idf,lenth_docs):
    normalize_data=tf_idf.copy()
    for term in tf_idf:
        for doc in tf_idf.index:
            if lenth_docs[doc]==0:
                normalize_data[term][doc]=0
            else:
                normalize_data[term][doc]=tf_idf[term][doc] / lenth_docs[doc]
    normalize_data.to_csv('normalizedocs.csv') 
    return normalize_data

def tf_idf_query_fun(tf_query,idf_dict):
    tf_idf_query=dict()
    for term in tf_query:
        if term in idf_dict:
            tf_idf_query[term]=(1 + np.log(tf_query[term])) * idf_dict[term]
        else:
            tf_idf_query[term]= 0
    return tf_idf_query

def normalize_query_fun(tf_idf_query,lenth_query):
    normalize_query=tf_idf_query.copy()
    for term in tf_idf_query:
        if lenth_query !=0:
            normalize_query[term]=  tf_idf_query[term] / lenth_query
        else:
            normalize_query[term]=0
    return  normalize_query

def sim_fun(normalize_data,normalize_query):
    sim=normalize_data.copy()
    for term in normalize_data: # drop terms in docs and not in query  
        if term not in normalize_query:
            sim.drop(term,inplace=True,axis=1)
    for term in normalize_query: #to add terms in query and not in docs 
        if term not in sim:
            sim[term]=np.nan
    for term in normalize_query:
        for doc in normalize_data.index:
            if term in normalize_data:
                sim[term][doc]=normalize_data[term][doc] * normalize_query[term]
            else:
                sim[term][doc] = 0
    return sim

def sort_sim_dict(sim):
    sim_dict=dict()
    for i in range(0,len(sim.index)):
        sim_dict[i]=sum(sim.loc[i])
    sorted_sim = {}
    sorted_keys = sorted(sim_dict, key=sim_dict.get,reverse=True)  
    for w in sorted_keys:
        sorted_sim[w] = sim_dict[w]
    return sorted_sim

def vector_space(number_files):
    term_tf,df_ls,idf_dict=df_and_idf(number_files)
    tf_idf=tf_idf_fun(number_files,df_ls,term_tf)

    lenth_docs=[math.sqrt(sum((tf_idf**2).loc[i]) ) for i in tf_idf.index]
    normalize_data=normalize_data_fun(tf_idf,lenth_docs)

    query,query_to=get_ls_steming_tokins()
    tf_query=Counter(query)
    tf_idf_query=tf_idf_query_fun(tf_query,idf_dict)

    lenth_query=0
    for i in tf_idf_query.values():
        lenth_query=lenth_query+(i**2)
    lenth_query=math.sqrt(lenth_query)

    normalize_query=normalize_query_fun(tf_idf_query,lenth_query)

    sim=sim_fun(normalize_data,normalize_query)
    sim=sort_sim_dict(sim)
    return sim

language,path,type_files,number_files,download=settings(read_env('env.txt'))
#-----------downlad------------
if download:
    nltk.download(['stopwords','punkt'])
file_dirs,file_names =fileNames(path,type_files)
scan_type_model(file_names,number_files)
# stopword = stopwords.words(language)
# print(stopwords)
