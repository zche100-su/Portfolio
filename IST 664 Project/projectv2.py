from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import FreqDist
import nltk
from nltk.collocations import *
import re
import matplotlib.pyplot as plt
import pandas as pd
import string
from nltk.tokenize import sent_tokenize
## https://stanfordnlp.github.io/CoreNLP/history.html need download the VERSION 4.0.0
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from stanfordcorenlp import StanfordCoreNLP
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,recall_score

############## Parameter Configuration ##########
# BASE = "C:\\Users\\GuoxingYao\\Downloads\\"
BASE = "C:\\Users\\guoxingyao\\Desktop\\"
stopwordslist = stopwords.words('english')
stopwordslist = [w.lower() for w in stopwordslist]
nlp = StanfordCoreNLP(r'C:\Users\guoxingyao\Desktop\stanford-corenlp-full-2020-04-20')

sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'

# nlp.parse(sentence)
# pattern = re.compile(r'.*(\(VP.*).*')
# pattern.search(nlp.parse(sentence))
############## End of Parameter Configuration ##########


## readData
## read the raw data from Dr. Xiao
## split data into two parts, the personal experience and non personal experience by the value
## of personal experience column value of 1 or 0
def readData():
    file = BASE+"PersonalExperience_CMVSubmissions (3).xlsx"
    data = pd.read_excel(file, sheet_name='Sheet1')
    personal_experience = data[data['Personal Experience'] == 1]
    non_personal_experience = data[data['Personal Experience'] == 0]

    data_experience_filtered = pd.DataFrame()
    personal_experience_filtered = pd.DataFrame()
    non_personal_experience_filtered = pd.DataFrame()

    personal_experience_filtered_vector = []
    non_personal_experience_filtered_vector = []
    data_experience_filtered_vector = []

    numberOfSentences = 0

    ## remove http, www, https
    for row in data['Submission']:
        row = row.replace('.', '. ').replace('?', '? ').replace('!', '! ').replace('**', '').replace('&gt;', '').replace('*', '').replace('-', '')
        sentences = sent_tokenize(row)
        temp = ""
        for sentence in sentences:
            # sentence = sentence.replace('**','').replace('&gt;','').replace('*','').replace('-','')
            if 'www' in sentence or 'http' in sentence or 'https' in sentence:
                continue
            temp = temp + sentence +' '
            numberOfSentences += 1
        data_experience_filtered_vector.append(temp)
    data_experience_filtered['Submission'] = data_experience_filtered_vector
    data_experience_filtered['Personal Experience'] = data['Personal Experience']
    print("Totally, there are {} of sentences".format(numberOfSentences))

    ## remove http, www, https
    for row in personal_experience['Submission']:
        row = row.replace('.','. ').replace('?','? ').replace('!','! ').replace('**','').replace('&gt;','').replace('*','').replace('-','')
        sentences = sent_tokenize(row)
        temp = ""
        for sentence in sentences:
            # sentence = sentence.replace('**','').replace('&gt;','').replace('*','').replace('-','')

            if 'www' in sentence or 'http' in sentence or 'https' in sentence:
                continue
            temp = temp + sentence +' '
        personal_experience_filtered_vector.append(temp)
    personal_experience_filtered['Submission'] = personal_experience_filtered_vector
    personal_experience_filtered['Personal Experience'] = [1] * personal_experience.shape[0]


    for row in non_personal_experience['Submission']:
        row = row.replace('.','. ').replace('?','? ').replace('!','! ').replace('**','').replace('&gt;','').replace('*','').replace('-','')
        sentences = sent_tokenize(row)
        temp = ""
        for sentence in sentences:
            # sentence = sentence.replace('**','').replace('&gt;','').replace('*','').replace('-','')

            if 'www' in sentence or 'http' in sentence or 'https' in sentence:
                continue
            temp = temp + sentence +' '
        non_personal_experience_filtered_vector.append(temp)
    non_personal_experience_filtered['Submission'] = non_personal_experience_filtered_vector
    non_personal_experience_filtered['Personal Experience'] = [0] * non_personal_experience.shape[0]

    return personal_experience_filtered, non_personal_experience_filtered,data_experience_filtered


## compute word frequency to get the general idea of the data
def getWordsFreq(data,filename,n,upv):
    word_list = []
    for row in data['Submission']:
        row = (re.sub(r"\*[,.;:\(\)@#?!&$`\"\'-()~=]+\ *", " ", row)).strip()
        words = word_tokenize(row)
        for word in words:
            if word not in string.punctuation:
                word_list.append(word.lower())
    showTopWords(word_list,filename,n = 100,upperbound=upv)
    # return word_list

## compute top words
def showTopWords(wordslist, filename,n=50,upperbound=300):
    wordslist_no_stops = [w for w in wordslist if w not in stopwordslist]
    ndist = FreqDist(wordslist_no_stops)
    topnwords = ndist.most_common(n)
    tempdata = pd.DataFrame()
    tempdata['Most Frequent'] = topnwords
    tempdata.to_csv(BASE+filename)
    y_axis = []
    for item in topnwords:
        # print("word  {} exist times of {} ".format(item[0], item[1]))
        y_axis.append(item[1])
    x_axis = list(range(1,n+1))

    plt.clf()
    plt.title("Top {} words".format(n))
    plt.ylabel("Frequency")
    plt.xlabel("Words")
    plt.plot(x_axis,y_axis,'bo-')
    plt.ylim((0,upperbound))
    plt.xlim((0,n))
    cnt = 1
    for item in topnwords:
        plt.annotate(item[0], (cnt, item[1]))
        cnt += 1
    plt.show()
## check the past tense by POS
def checkPastTense(data):
    pos_tagged_array = []
    sentence_array = []
    for row in data['Submission']:
        sents = sent_tokenize(row)
        for sent in sents:
            sent = (re.sub(r"\*[,.;:\(\)@#?!&$`\"\'-()~=]+\ *", " ", sent)).strip()
            flag = False
            if sent is not None:
                words = word_tokenize(sent)
                for tag in nltk.pos_tag(words):
                    # eliminating unnecessary POS tags
                    if 'VBD' in tag[1]:
                        pos_tagged_array.append(tag)
                        flag = True
                if flag == True:
                    sentence_array.append(sent)
    print(sentence_array)
    return pos_tagged_array

# check past tense by stanford core nlp
def checkPastTenseStanford(data):
    negationWords = ['No','Not','None','No one','Nobody','Nothing','Neither','Nowhere','Never','Hardly','Scarcely','Barely','n\'t']
    pos_tagged_array = []
    sentence_array = []
    result = pd.DataFrame()
    targetValue = []
    pastTense = []
    # translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    pattern = re.compile(
        r'(\(PRP i\).*\(VP \(VBD .*)|(\(PRP we\).*\(VP \(VBD .*\))|(\(PRP\$ my\).*\(VP \(VBD .*)|(\(PRP\$ our\).*\(VP \(VBD .*)',
        re.IGNORECASE)
    index = 0
    for row,value in zip(data['Submission'],data['Personal Experience']):
        sents = sent_tokenize(row)
        length = len(sents)
        targetValue.extend([value]*length)
        index += 1
        for sent in sents:
            # flag = False
            # for word in negationWords:
            #     if word.lower() in sent:
            #         flag = True
            #         break
            # if flag == True:
            #     pastTense.append(0)
            #     # targetValue.pop()
            #     continue

            # sent = (re.sub(r"\*[,.;:\(\)@#?!&$`\"\'-()~=]+\ *", " ", sent)).strip()
            # sent = sent.translate(translator)
            if sent is not None and len(sent) > 1:
                flag1 = False
                try:
                    matched_vp = pattern.findall(nlp.parse(sent).replace('\r\n',''))
                    if len(matched_vp) >= 1:
                        print(sent)
                        sentence_array.append(sent)
                        pastTense.append(1)
                    else:
                        pastTense.append(0)

                    # for matched in matched_vp:
                    #     # print(matched)
                    #     if 'VBD' in matched:
                    #         print(sent)
                    #         sentence_array.append(sent)
                    #         pastTense.append(1)
                    #         flag1 = True
                    #         break
                    # if flag1 == False:
                    #     pastTense.append(0)
                except:
                    print("Got exception")
                    pastTense.append(0)
            else:
                pastTense.append(0)

        assert len(pastTense) == len(targetValue)
            # print('not equal at ', index)

    print(sentence_array)
    result['Past Tense'] = pastTense
    result['New Targert'] = targetValue
    result.to_csv(BASE + 'past_tense_stanford.csv')
    return pos_tagged_array,result

def dummy(doc):
    return doc

def getCountVectorDataFrame(data):
    # vectorizer = TfidfVectorizer()
    vectorizer = CountVectorizer(
        tokenizer=dummy,
        preprocessor=dummy,
    )
    temp = []
    tempTarget = []
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    for row,value in zip(data['Submission'],data['Personal Experience']):
        sents = sent_tokenize(row)
        length = len(sents)
        tempTarget.extend([value]*length)
        for sent in sents:
            # sent = (re.sub(r"\*[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]+\ *", " ", sent)).strip()
            # sent = (re.sub(r"\*[,.;:\(\)@#?!&$`\"\'-()~=]+\ *", " ", sent)).strip()
            sent = sent.translate(translator)
            words = word_tokenize(sent)
            words = [w for w in words if w.isalpha()]
            temp.append(words)

    vectors = vectorizer.fit_transform(temp)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()

    df = pd.DataFrame(denselist, columns=feature_names)
    df['Target'] = tempTarget
    df.to_csv(BASE+'vector.data.csv')
    return df

# def getBiTopBigram(n=100):
#     file = BASE+"PersonalExperience_CMVSubmissions (3).xlsx"
#     data = pd.read_excel(file, sheet_name='Sheet1')
#     rows = ""
#     for row in data['Submission']:
#         rows += row
#     tokens = nltk.wordpunct_tokenize(rows)
#     finder = BigramCollocationFinder.from_words(tokens)
#     bigram_measures = nltk.collocations.BigramAssocMeasures()
#     scored = finder.score_ngrams(bigram_measures.raw_freq)
#     result = sorted(bigram for bigram, score in scored)
#     y_axis = []
#     for item in result:
#         # print("word  {} exist times of {} ".format(item[0], item[1]))
#         y_axis.append(item[1])
#     x_axis = list(range(1,n+1))
#
#     plt.clf()
#     plt.title("Top {} words".format(n))
#     plt.ylabel("Frequency")
#     plt.xlabel("Words")
#     plt.plot(x_axis,y_axis,'bo-')
#     plt.ylim((0,100))
#     plt.xlim((0,n))
#     cnt = 1
#     for item in result:
#         plt.annotate(item[0], (cnt, item[1]))
#         cnt += 1
#     plt.show()


def getTFIDFDataFrame(data):
    vectorizer = TfidfVectorizer(
        analyzer='word',
        tokenizer=dummy,
        preprocessor=dummy,
    )
    # vectorizer = CountVectorizer(
    #     tokenizer=dummy,
    #     preprocessor=dummy,
    # )
    temp = []
    tempTarget = []
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    for row,value in zip(data['Submission'],data['Personal Experience']):
        sents = sent_tokenize(row)
        length = len(sents)
        tempTarget.extend([value]*length)
        for sent in sents:
            # sent = (re.sub(r"\*[!\"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]+\ *", " ", sent)).strip()
            # sent = (re.sub(r"\*[,.;:\(\)@#?!&$`\"\'-()~=]+\ *", " ", sent)).strip()
            sent = sent.translate(translator)
            words = word_tokenize(sent)
            words = [w for w in words if w.isalpha()]
            temp.append(words)

    vectors = vectorizer.fit_transform(temp)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()

    df = pd.DataFrame(denselist, columns=feature_names)
    df['Target'] = tempTarget
    df.to_csv(BASE+'vector.data.csv')
    return df

def classifier(data,title):
    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, ~data.columns.isin(['Target'])], data['Target'], test_size=0.1, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print(title)
    print('F1 value = ',f1_score(y_test, y_pred))
    print('Accuracy value = ',accuracy_score(y_test, y_pred))
    print('Recall value = ',recall_score(y_test, y_pred))
    return data

def classifierStanford(data,title):
    X_train, X_test, y_train, y_test = train_test_split(data['Past Tense'], data['New Targert'], test_size=0.1, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train.reshape(-1,1), y_train.reshape(-1,1)).predict(X_test.reshape(-1,1))
    print(title)
    print('F1 value = ',f1_score(y_test.reshape(-1,1), y_pred.reshape(-1,1)))
    print('Accuracy value = ',accuracy_score(y_test.reshape(-1,1), y_pred.reshape(-1,1)))
    print('Recall value = ',recall_score(y_test.reshape(-1,1), y_pred.reshape(-1,1)))
    return data


if __name__=='__main__':

    # Step -1: calculate bigram
    # getBiTopBigram()

    # step 0 : all_data saves all the submit field data values
    personal_experience_data,non_personal_experience_data,all_data = readData()

    # TEST purpose: already filter out the www, http, https
    # non_personal_experience_data.to_csv(BASE+"yes.csv")
    # End of TEST

    # step 1: analyze word frequency in this project
    getWordsFreq(all_data,"Top100_all_data.csv",n=100,upv=800)
    getWordsFreq(non_personal_experience_data,"Top100_personal_data.csv",n=100,upv=500)
    getWordsFreq(personal_experience_data,"Top100_non_personal_data.csv",n=100,upv=300)
    # checkPastTenseStanford(personal_experience_data,upv=600)

    # step 3: classifier for Bag of Word by counter vector
    data = getCountVectorDataFrame(all_data)
    classifier(data,"For Bag of words")

    # Step 4: classifier for tf idf vector
    data = getTFIDFDataFrame(all_data)
    classifier(data,"For TF-IDF ")

    # step 5 : use stanford parser to prepare past tense for clasification
    # checkPastTenseStanford(all_data)

    # step 6: use data generated by stanford parser to build navive bayers
    dataStanford = pd.DataFrame(pd.read_csv(BASE+'past_tense_stanford.csv'))
    classifierStanford(dataStanford,'For stanford parser ')
    print('stop here')