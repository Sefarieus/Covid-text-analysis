import pandas as pd
import nltk
#nltk.download()
from nltk import FreqDist
from nltk.corpus import opinion_lexicon
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

# Frequent words analysis, applied Part of Speech Tags
if __name__ == '__main__':
    df = pd.read_csv(r"C:/Users/肖瑜/Desktop/HW+lecture/web analytics/project/COVID-19-2110to2112.csv")
    i = 0
    n = int(str((df.shape[0])))
    rawlist = []
    raw = []
    while i in range(n):
        rawlist.append(df['content'][i])
        i = i + 1
    raw = ''.join(rawlist)
    raw = raw.replace(r'\n',' ')
    tokens = nltk.word_tokenize(raw)

    words1 = [w.lower() for w in tokens]
    words2 = [w for w in words1 if w.isalpha()]77

    freq = FreqDist(words2)
    sorted_freq = sorted(freq.items(),key = lambda k:k[1], reverse = True)
    stopwords = stopwords.words('english')  #use nltk stopwords
    words_nostopwords = [w for w in words2 if w not in stopwords]   # 去除全部默认的stopwords
    freq_nostw = FreqDist(words_nostopwords)
    sorted_freq_nostw = sorted(freq_nostw.items(),key = lambda k:k[1], reverse = True)
    freq_nostw.plot(30)

    POS_tags = nltk.pos_tag(tokens)
    POS_tag_list = [(word, tag) for (word, tag) in POS_tags if tag.startswith('J')]
    tag_freq = nltk.FreqDist(POS_tag_list)
    sorted_tag_freq = sorted(tag_freq.items(), key=lambda k: k[1], reverse=True)
    tag_freq.plot(30)
    pass


# Now for sentiment analysis, in this part I use 3 tools. LM, Lexicon and Textblob
if __name__ == '__main__':
    df = pd.read_table("COVID-19-2110to2112.csv", header= None, names = ['content'])
    data = df.content.str.lower()
    def count_pos_neg(data, positive_dict, negative_dict):
        poscnt = []
        negcnt = []
        netcnt = []

        for nrow in range(0,len(data)):
            text = data[nrow]
            qa = 0
            qb = 0
            for word in positive_dict :
                if (word in text) :
                    qa = qa + 1
            for word in negative_dict :
                if (word in text) :
                    qb = qb + 1

            qc = qa - qb
            poscnt.append(qa)
            negcnt.append(qb)
            netcnt.append(qc)
        return (poscnt, negcnt, netcnt)


    #LEXICON
    #net_list_BL counts how positive and negative this tweet is. If it is negative, the tweet is a negative one; if it is positive, the tweet is a positive one
    pos_list_BL=set(opinion_lexicon.positive())
    neg_list_BL=set(opinion_lexicon.negative())
    df['poscnt_BL'], df['negcnt_BL'], df['netcnt_BL'] = count_pos_neg(data, pos_list_BL, neg_list_BL)
    print(df[['content','poscnt_BL','negcnt_BL','netcnt_BL']].head(5))
    i = 1
    countpos_BL = 0
    countneg_BL = 0
    for i in range(n):
        if df['netcnt_BL'][i] > 0:
            countpos_BL = countpos_BL + 1
        if df['netcnt_BL'][i] < 0:
            countneg_BL = countneg_BL + 1
        i = i + 1
    percentage_pos = countpos_BL / n
    percentage_neg = countneg_BL / n
    print("The percentage of positive tweets are(according to BL): ", percentage_pos)
    print("The percentage of negative tweets are(according to BL): ", percentage_neg)


    #LM DICTIONARY NEEDS TO READ THE LOCAL FILE
    #netcnt_LM works like that of Lexicon dictionary
    def read_local_dictionary(file):
        words_dict = []
        with open(file, "r") as f:
            for line in f:
                t = line.strip().lower()
                words_dict.append(t)
        return words_dict
    pos_list_LM = read_local_dictionary('C:/Users/肖瑜/Desktop/HW+lecture/web analytics/project/positive-words-LM.txt')
    neg_list_LM = read_local_dictionary('C:/Users/肖瑜/Desktop/HW+lecture/web analytics/project/negative-words-LM.txt')
    df['poscnt_LM'], df['negcnt_LM'], df['netcnt_LM'] = count_pos_neg(data, pos_list_LM, neg_list_LM)
    print(df[['content','poscnt_LM','negcnt_LM','netcnt_LM']].head(5))
    i = 1
    countpos_LM = 0
    countneg_LM = 0
    for i in range(n):
        if df['netcnt_LM'][i] > 0:
            countpos_LM = countpos_LM + 1
        if df['netcnt_LM'][i] < 0:
            countneg_LM = countneg_LM + 1
        i = i + 1
    percentage_pos = countpos_LM / n
    percentage_neg = countneg_LM / n
    print("The percentage of positive tweets are(according to LM): ", percentage_pos)
    print("The percentage of negative tweets are(according to LM): ", percentage_neg)


    #TEXTBLOB
    # The polarity score is a float within the range [-1.0, 1.0].
    # The subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    df["score_TextBlob"] = df["content"].map(lambda x:TextBlob(x).sentiment)
    print (df[["content","score_TextBlob"]].head(5))
    i = 1
    countpos_TB = 0
    countneg_TB = 0
    for i in range(n):
        if df["score_TextBlob"][i][0] > 0:
            countpos_TB = countpos_TB + 1
        if df["score_TextBlob"][i][0] < 0:
            countneg_TB = countneg_TB + 1
        i = i + 1
    percentage_pos = countpos_TB / n
    percentage_neg = countneg_TB / n
    print("The percentage of positive tweets are(according to textblob): ", percentage_pos)
    print("The percentage of negative tweets are(according to textblob): ", percentage_neg)


    #VADERSENTIMENT(I failed to apply this, but I think this is a very great dictionary)
    # The compound score is the sum of positive, negative & neutral scores which is then normalized between -1(most extreme negative) and +1 (most extreme positive)
    #analyzer = SentimentIntensityAnalyzer()
    #scores = [analyzer.polarity_scores(sentence) for sentence in data]
    # print (scores)

    #neg_s = [j['neg'] for j in scores]
    #neu_s = [j['neu'] for j in scores]
    #pos_s = [j['pos'] for j in scores]
    #compound_s = [j['compound'] for j in scores]

    #df['negscore_Vader'], df['neuscore_Vader'], df['posscore_Vader'], df['compound_Vader'] = neg_s, neu_s, pos_s, compound_s
    #df[['content','negscore_Vader','neuscore_Vader','posscore_Vader','compound_Vader']].head(5)
    pass




