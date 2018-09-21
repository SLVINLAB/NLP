# import warnings
# warnings.filterwarnings('ignore')
# warnings.filterwarnings('default')

from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import _split
from wordcloud import WordCloud

import pandas as pd
import numpy as np
import pickle

#from fbprophet import Prophet
from datetime import datetime

import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager,rc
from _datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split
#
try:
    from AI.AIEngine.datamanager_All import datamanagerAll
    from AI.AIEngine.sendMail import SendEMail
except:
    from datamanager_All import datamanagerAll
    from sendMail import SendEMail

import sklearn.feature_extraction.text as text
#from tfidf_kor import tfidfScorer

class FnAIMarket():


    def __init__(self):
        print("FnAIMarket")
        self.dm = datamanagerAll()
        self.status = False

    def findsomething(self,market = "ALL"):
        dfUPSTOCK, dfDownSTOCK = self.getListStock('2018-06-22', '2018-06-21')

        listDownKeyWord = self.getKeyword(dfDownSTOCK)
        listUpKeyWord = self.getKeyword(dfUPSTOCK)

        print(listUpKeyWord)
        print(listDownKeyWord)

        self.makeWordCloud(listUpKeyWord)
        self.makeWordCloud(listDownKeyWord)

    def getListStock(self,strDate, strDatebefore):

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

        if len(UPSTOCK) < 20:
            UPSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] > 4]
            if len(UPSTOCK) < 20:
                UPSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] > 3]
                if len(UPSTOCK) < 20:
                    UPSTOCK = KOREA_STOCK_UPDOWN[KOREA_STOCK_UPDOWN['UPDOWN'] > 2]

        if len(DownSTOCK) < 20:
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
        # dfRS = dfRS.drop_duplicates(['CODE'], keep='first')
        # dfFull = pd.concat([dfRS1])
        dfFull = pd.concat([dfRS, dfRS1])
        # corpus = dfFull['BIZ_STATUS_BRIEF'].to_string()
        # import re
        # #corpus = re.sub('[^가-힣a-zA-Z0-9_]', ' ', corpus)
        # corpus = re.sub('[^가-힣a-zA-Z_]', ' ', corpus)
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
        # dsadjasdlja
        # dfWords = pd.read_csv('keywords2.csv')
        # dfWords = dfWords.drop_duplicates(['word'], keep='first')
        # dfWords = dfWords[dfWords['frequency']< 200]
        # print(dfWords)

        # dfWords = pd.read_csv('hit.csv',encoding='cp949')
        # dfWords = dfWords.drop_duplicates(['word'], keep='first')
        # dfWords = dfWords[dfWords['hit']< 100]
        # dfWords = dfWords[dfWords['hit']> 2]
        # print(dfWords)

        dfUPSTOCK_ = pd.DataFrame()
        dfDownSTOCK_ = pd.DataFrame()
        dfUPSTOCK = dfFull[dfFull['CODE'].isin(UPSTOCK.index.tolist())]
        dfDownSTOCK = dfFull[dfFull['CODE'].isin(DownSTOCK.index.tolist())]
        dfUPSTOCK_['comment'] = dfUPSTOCK.groupby('CODE')['BIZ_STATUS_BRIEF'].apply('.'.join)
        dfDownSTOCK_['comment'] = dfDownSTOCK.groupby('CODE')['BIZ_STATUS_BRIEF'].apply('.'.join)
        print(dfUPSTOCK_)
        return dfUPSTOCK_, dfDownSTOCK_

    def getKeyword(self,dfUpdown):
        stopwords = ['때문', '경우', '분위기', '수익성이', '중임', '실현', '있기', '선정', '어려움', '흑전', '포함', '대부분', '반면', '잔고',
                     '이익증가', '역성장',
                     'company', '모멘텀에', '영업가치', '수도권', '해외시장']

        dfWords = pd.read_csv('keywords4.csv', encoding='cp949')
        dfWords = dfWords.drop_duplicates(['word'], keep='first')
        dfWords = dfWords[dfWords['frequency'] < 100]
        dfWords = dfWords[dfWords['frequency'] > 2]
        dfWords = dfWords[dfWords['index'] == 1]
        print(dfWords)

        vectorizer = text.CountVectorizer(input='content', stop_words='english', min_df=3)
        # dfUPSTOCK, dfDOWNSTOCK = self.getListStock('2018-08-28', '2018-08-27')
        # dfUPSTOCK = dfUPSTOCK[0:50]
        # dfUPSTOCK['token_BIZ_STATUS_BRIEF'] = dfUPSTOCK['comment']#.apply(tokenizer_twitter_morphs) + dfThema['BIZ_STATUS_BRIEF'].apply(tokenizer_morphs)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)
        doc_ko = dfUpdown['comment'].tolist()
        dtm = vectorizer.fit_transform(doc_ko)
        df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names(), index=dfUpdown.index)

        print(df.to_string)
        dfNew = pd.DataFrame()
        for st in df.columns:
            if len(dfWords[dfWords['word'].isin([st])]) > 0:
                dfNew[st] = df[st]
        import numpy as np
        dfNew = np.sign(dfNew)
        sums = dfNew.select_dtypes(pd.np.number).sum().rename('total')
        dfNew= dfNew.append(sums)
        #dfNew.loc['total'] = dfNew.select_dtypes(pd.np.number).sum()


        dfNew.to_csv('DTM.csv')
        print(dfNew)

        listUpDown = []
        try:
            for str in dfNew.columns.get_values():
                if dfNew[str]['total'] > 1:
                    if str in stopwords:
                        pass
                    else:
                        listUpDown.append(str)
        except:
            pass
        return listUpDown

    def makeWordCloud(self,listUpDown):
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
        if len(unique_string) == 0:
            unique_string = 'NoData'
        wordcloud = WordCloud(width=1000, height=500, font_path=path).generate(unique_string)
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud)
        plt.axis("off")
        # plt.savefig("listUpDown" + ".png", bbox_inches='tight')
        plt.show()




if __name__ == '__main__':
    #python manage.py runserver 11.4.3.200:8000

    AI = FnAIMarket()
    print(AI.findsomething())






