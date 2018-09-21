import gensim

from datamanager_All import datamanagerAll
import pickle
import pandas as pd
from konlpy.tag import Kkma, Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf_kor import tfidfScorer
from wordcloud import WordCloud


def getListStock(strDate,strDatebefore):

    dm = datamanagerAll()
    status = False

    try:
        file = open('KOREA_STOCK_pk', 'rb')
    except:
        KOREA_STOCK = pd.read_excel('korean_stock_close.xlsx', index_col=0)
        with open('KOREA_STOCK_pk', 'wb') as fin:
            pickle.dump(KOREA_STOCK, fin)
        file = open('KOREA_STOCK_pk', 'rb')

    KOREA_STOCK = pickle.load(file)
    print(KOREA_STOCK)

    KOREA_STOCK_UPDOWN = pd.DataFrame()
    KOREA_STOCK_UPDOWN['UPDOWN'] = (KOREA_STOCK.loc[strDate] - KOREA_STOCK.loc[strDatebefore]) / \
                                   KOREA_STOCK.loc[
                                       strDatebefore] * 100
    KOREA_STOCK_UPDOWN = KOREA_STOCK_UPDOWN.sort_values(by=["UPDOWN"], ascending=False)
    print(KOREA_STOCK_UPDOWN)

    UPSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] > 5]
    DownSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] < -5]

    print(len(UPSTOCK))
    print(len(DownSTOCK))

    if len(UPSTOCK) < 20 :
        UPSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] > 4]
        if len(UPSTOCK) < 20:
            UPSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] > 3]
            if len(UPSTOCK) < 20:
                UPSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] > 2]

    if len(DownSTOCK) < 20 :
        DownSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] < -4]
        if len(DownSTOCK) < 20:
            DownSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] < -3]
            if len(DownSTOCK) < 20:
                DownSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] < -2]

    # '''리포트 12만건 학습'''
    dfRS = dm.getDataReport(update=status)

    # '''비지니스 써머리 최신 종목별 학습'''
    dfRS1 = dm.getDataSummary(update=status)

    # '''공시 종목별 학습'''
    dfRS2 = dm.getDataDisclouse(update=status)

    # '''한경기사 최신 종목별 학습'''
    dfRS3 = dm.getDataHANKYUNG(update=status)

    # '''이투데이기사 최신 종목별 학습'''
    dfRS4 = dm.getDataETODAY(update=status)
    dfFull = pd.concat([dfRS, dfRS1])#, dfRS2, dfRS3, dfRS4])
    # corpus = dfFull['BIZ_STATUS_BRIEF'].to_string()
    # import re
    # corpus =re.sub('[^가-힣a-zA-Z0-9_]', ' ', corpus)
    # print(corpus)
    # from soynlp.noun import NewsNounExtractor,LRNounExtractor
    # noun_extractor = NewsNounExtractor()
    # noun_extractor.train([corpus])  # list of str or like
    # nouns = noun_extractor.extract()
    #
    # dfWords = pd.DataFrame(columns={'word','score','frequency'})
    # for word, score in nouns.items():
    #     dfWords=   dfWords.append({'word':word, 'score':score.score, 'frequency':score.frequency},ignore_index=True)
    # dfWords.to_csv('keywords2.csv')
    # dfWords = pd.read_csv('keywords2.csv')
    # # dfWords = dfWords.drop_duplicates(['word'], keep='first')
    # # dfWords = dfWords[dfWords['frequency']< 200]
    # # print(dfWords)

    dfWords = pd.read_csv('hit.csv',encoding='cp949')
    dfWords = dfWords.drop_duplicates(['word'], keep='first')
    dfWords = dfWords[dfWords['hit']< 100]
    dfWords = dfWords[dfWords['hit']> 2]
    print(dfWords)

    dfUPSTOCK = dfFull[dfFull['CODE'].isin(UPSTOCK.index.tolist())]
    dfDownSTOCK = dfFull[dfFull['CODE'].isin(DownSTOCK.index.tolist())]
    dfUPSTOCK['comment'] = dfUPSTOCK.groupby('CODE')['BIZ_STATUS_BRIEF'].apply('.'.join)
    dfDownSTOCK['comment'] = dfDownSTOCK.groupby('CODE')['BIZ_STATUS_BRIEF'].apply('.'.join)
    print(dfUPSTOCK)
    return dfUPSTOCK,dfDownSTOCK


def getKeyword(dfUpdown):
        data_text = dfUpdown[['BIZ_STATUS_BRIEF']]
        data_text['index'] = data_text.index
        documents = data_text

        dfWords = pd.read_csv('hit.csv', encoding='cp949')
        dfWords = dfWords.drop_duplicates(['word'], keep='first')
        dfWords = dfWords[dfWords['hit'] < 100]
        dfWords = dfWords[dfWords['hit'] > 2]
        print(dfWords)
        tw = Twitter()
        def preprocess(text):
            #result = tw.morphs(text)
            result = []
            for token in gensim.utils.simple_preprocess(text):
                print("token",token)
                #print(dfWords['word'].isin([token]))
                if len(dfWords[dfWords['word'].isin([token])])>0:
                    result.append(token)
                    # if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                    #     result.append(token)
            return result

        # 판다스 데이타프레임을 포문을 돌리지 않고 map 이라는 명령어주면 같은 효과내는것 같아요.
        processed_docs = documents['BIZ_STATUS_BRIEF'].map(preprocess)
        print(processed_docs[:10])

        dictionary = gensim.corpora.Dictionary(processed_docs)
        count = 0
        for k, v in dictionary.iteritems():
            print(k, v)
            count += 1
            if count > 10:
                break

        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=10000)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        if len(bow_corpus) < 50:
            dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=10000)
            bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
            if len(bow_corpus) < 30:
                dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000)
                bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
                if len(bow_corpus) < 20:
                    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=10000)
                    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        from gensim import corpora, models
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        from pprint import pprint
        for doc in corpus_tfidf:
            pprint(doc)
            break

        listWord_bow_corpus = []
        listWord_bow_corpus_tfidf = []

        lda_model = gensim.models.LdaModel(bow_corpus, num_topics=5, id2word=dictionary, passes=2)
        x=lda_model.show_topics(num_topics=5, num_words=5,formatted=False)
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
        topics_words = [([wd[0] for wd in tp[1]]) for tp in x]
        topics_words = sum(topics_words,[])
        topics_words = list(set(topics_words))

        lda_model_tfidf = gensim.models.LdaModel(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2)
        x=lda_model_tfidf.show_topics(num_topics=5, num_words=5,formatted=False)
        topics_words_tfidf = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
        topics_words_tfidf = [([wd[0] for wd in tp[1]]) for tp in x]
        topics_words_tfidf = sum(topics_words_tfidf,[])
        topics_words_tfidf = list(set(topics_words_tfidf))

        listTotal = topics_words + topics_words_tfidf
        listTotal = list(set(listTotal))

        return listTotal

        print('bow_corpus-----------------\n',topics_words)
        print('bow_corpus_tfidf------------------\n',topics_words_tfidf)
        print('listTotal-----------------\n',listTotal)

def makeWordCloud(listUpDown):
    import matplotlib.pyplot as plt
    import platform
    from matplotlib import font_manager, rc
    plt.rcParams['axes.unicode_minus'] = False
    path = ''
    if platform.system() == 'Darwin':
        rc('font', family='AppleGothic')
    elif platform.system() == 'Windows':
        path = 'c:/Windows/fonts/malgun.ttf'
        font_name = font_manager.FontProperties(fname=path).get_name()
        rc('font', family=font_name)
    else:
        print('UnKnown System')

    # convert list to string and generate
    unique_string = (" ").join(listUpDown)

    wordcloud = WordCloud(width=1000, height=500,font_path=path).generate(unique_string)
    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    #plt.savefig("listUpDown" + ".png", bbox_inches='tight')
    plt.show()


dfUPSTOCK,dfDownSTOCK = getListStock('2018-09-07','2018-09-06')

listDownKeyWord= getKeyword(dfDownSTOCK)
listUpKeyWord = getKeyword(dfUPSTOCK)

print(listUpKeyWord)
print(listDownKeyWord)

makeWordCloud(listUpKeyWord)
makeWordCloud(listDownKeyWord)






    # for topic,words in topics_words_tfidf:
    #     listWord_bow_corpus_tfidf.append(words)
    #
    # print(listWord_bow_corpus)

    #
    # for idx, topic in lda_model.print_topics(-1):
    #     print('lda_model_Topic: {} Words: {}'.format(idx, topic))
    #     listWord_bow_corpus.append(topic)
    #
    # lda_model_tfidf = gensim.models.LdaModel(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2)
    # for idx, topic in lda_model_tfidf.print_topics(-1):
    #     print('lda_model_tfidf_Topic: {} Word: {}'.format(idx, topic))
    #     listWord_bow_corpus_tfidf.append(topic)

    # listWord_bow_corpus = sum(list(listWord_bow_corpus),[])
    # listWord_bow_corpus_tfidf = sum(list(listWord_bow_corpus_tfidf),[])
    # listTotal = listWord_bow_corpus + listWord_bow_corpus_tfidf
    # listTotal = sum(list(listTotal),[])

    # x=lda_model.show_topics(num_topics=12, num_words=5,formatted=False)
    # topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
    #
    # #Below Code Prints Topics and Words
    # for topic,words in topics_words:
    #     print(str(topic)+ "::"+ str(words))
    # print()
    #
    # #Below Code Prints Only Words
    # for topic,words in topics_words:
    #     print(" ".join(words))

    # print('bow_corpus-----------------\n',listWord_bow_corpus)
    # print('bow_corpus_tfidf------------------\n',listWord_bow_corpus_tfidf)
    # print('listTotal-----------------\n',listTotal)



    # for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    #     print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
    #
    # for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    #     print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
    #
    #
    # unseen_document = 'How a Pentagon deal became an identity crisis for Google'
    # bow_vector = dictionary.doc2bow(preprocess(unseen_document))
    # for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    #     print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


