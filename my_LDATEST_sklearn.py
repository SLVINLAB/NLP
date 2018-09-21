import gensim

from datamanager_All import datamanagerAll
import pickle
import pandas as pd
from konlpy.tag import Kkma, Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf_kor import tfidfScorer
from wordcloud import WordCloud
import sklearn.feature_extraction.text as text

stopwords = ['때문','경우','분위기','수익성이','중임','실현','있기','선정','어려움','흑전','포함','대부분','반면','잔고','이익증가','역성장','company','모멘텀에','영업가치','수도권','해외시장']

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
    # dfRS = dfRS.drop_duplicates(['CODE'], keep='first')
    #dfFull = pd.concat([dfRS1])
    dfFull = pd.concat([dfRS,dfRS1])
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
    dfDownSTOCK_= pd.DataFrame()
    dfUPSTOCK = dfFull[dfFull['CODE'].isin(UPSTOCK.index.tolist())]
    dfDownSTOCK = dfFull[dfFull['CODE'].isin(DownSTOCK.index.tolist())]
    dfUPSTOCK_['comment'] = dfUPSTOCK.groupby('CODE')['BIZ_STATUS_BRIEF'].apply('.'.join)
    dfDownSTOCK_['comment'] = dfDownSTOCK.groupby('CODE')['BIZ_STATUS_BRIEF'].apply('.'.join)
    print(dfUPSTOCK_)
    return dfUPSTOCK_,dfDownSTOCK_


dfWords = pd.read_csv('keywords4.csv',encoding='cp949')
dfWords = dfWords.drop_duplicates(['word'], keep='first')
dfWords = dfWords[dfWords['frequency']< 100]
dfWords = dfWords[dfWords['frequency']> 2]
dfWords = dfWords[dfWords['index']==1]
print(dfWords)


vectorizer = text.CountVectorizer(input='content', stop_words='english', min_df=2)
dfUPSTOCK,dfDOWNSTOCK = getListStock('2018-08-28','2018-08-27')
#dfUPSTOCK = dfUPSTOCK[0:50]
#dfUPSTOCK['token_BIZ_STATUS_BRIEF'] = dfUPSTOCK['comment']#.apply(tokenizer_twitter_morphs) + dfThema['BIZ_STATUS_BRIEF'].apply(tokenizer_morphs)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)
doc_ko = dfUPSTOCK['comment'].tolist()
dtm = vectorizer.fit_transform(doc_ko)
df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names(),index=dfUPSTOCK.index)

print(df.to_string)
dfNew = pd.DataFrame()
for st in df.columns:
    if len(dfWords[dfWords['word'].isin([st])])>0:
        dfNew[st]= df[st]
import numpy as np
dfNew = np.sign(dfNew)
dfNew.loc['total'] = dfNew.select_dtypes(pd.np.number).sum()
dfNew.to_csv('DTM.csv')
print(dfNew)

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
    plt.close()

listUpDown = []

for str in dfNew.columns.get_values():
    if dfNew[str]['total'] > 2:
        if str in stopwords:
            pass
        else:
            listUpDown.append(str)

makeWordCloud(listUpDown)


#
#
# import pandas as pd
#
# pd.options.mode.chained_assignment = None
#
# import numpy as np
#
# np.random.seed(0)
#
# from konlpy.tag import Twitter
#
# twitter = Twitter()
#
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
# from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
#
#
# # tokenizer : 문장에서 색인어 추출을 위해 명사,동사,알파벳,숫자 정도의 단어만 뽑아서 normalization, stemming 처리하도록 함
# def tokenizer(raw, pos=["Noun", "Alpha", "Verb", "Number"], stopword=[]):
#     return [
#         word for word, tag in twitter.pos(
#             raw,
#             norm=True,  # normalize 그랰ㅋㅋ -> 그래ㅋㅋ
#             stem=True  # stemming 바뀌나->바뀌다
#         )
#         if len(word) > 1 and tag in pos and word not in stopword
#     ]
#
#
# # 테스트 문장
# rawdata = [
#     '남북 고위급회담 대표단 확정..남북 해빙모드 급물살',
#     '[남북 고위급 회담]장차관만 6명..판 커지는 올림픽 회담',
#
#     '문재인 대통령과 대통령의 영부인 김정숙여사 내외의 동반 1987 관람 후 인터뷰',
#     '1987 본 문 대통령.."그런다고 바뀌나? 함께 하면 바뀐다"',
#
#     '이명박 전 대통령과 전 대통령의 부인 김윤옥 여사, 그리고 전 대통령의 아들 이시형씨의 동반 검찰출석이 기대됨'
# ]
#
# vectorize = CountVectorizer(
#     tokenizer=tokenizer,
#     min_df=2    # 예제로 보기 좋게 1번 정도만 노출되는 단어들은 무시하기로 했다
#                 # min_df = 0.01 : 문서의 1% 미만으로 나타나는 단어 무시
#                 # min_df = 10 : 문서에 10개 미만으로 나타나는 단어 무시
#                 # max_df = 0.80 : 문서의 80% 이상에 나타나는 단어 무시
#                 # max_df = 10 : 10개 이상의 문서에 나타나는 단어 무시
# )
#
# # 문장에서 노출되는 feature(특징이 될만한 단어) 수를 합한 Document Term Matrix(이하 DTM) 을 리턴한다
# X = vectorize.fit_transform(rawdata)
#
# print(
#     'fit_transform, (sentence {}, feature {})'.format(X.shape[0], X.shape[1])
# )
# # fit_transform, (sentence 5, feature 7)
#
# print(type(X))
# # <class 'scipy.sparse.csr.csr_matrix'>
#
# print(X.toarray())
#
# features = vectorize.get_feature_names()
#
# # 검색 문장에서 feature를 뽑아냄
# srch=[t for t in tokenizer('1987 관람한 대통령 인터뷰 기사') if t in features]
# print(srch)
# # ['1987', '대통령']
#
# # dtm 에서 검색하고자 하는 feature만 뽑아낸다.
# srch_dtm = np.asarray(X.toarray())[:, [
#     # vectorize.vocabulary_.get 는 특정 feature 가 dtm 에서 가지고 있는 index값을 리턴한다
#     vectorize.vocabulary_.get(i) for i in srch
# ]]
#
#
# # [[0, 1, 2, 0, 0, 0, 1],
# # [0, 1, 1, 0, 0, 0, 2],
# # [1, 0, 0, 2, 1, 1, 0],
# # [1, 0, 0, 1, 0, 0, 0],
# # [0, 0, 0, 3, 1, 1, 0]]
#
# score = srch_dtm.sum(axis=1)
# print(score)
# # array([0, 0, 3, 2, 3], dtype=int64) 문장별 feature 합계 점수
#
# for i in score.argsort()[::-1]:
#     if score[i] > 0:
#         print('{} / score : {}'.format(rawdata[i], score[i]))
#
# # 이명박 전 대통령과 전 대통령의 부인 김윤옥 여사, 그리고 전 대통령의 아들 이시형씨의 동반 검찰출석이 기대됨 / score : 3
# # 문재인 대통령과 대통령의 영부인 김정숙여사 내외의 동반 1987 관람 후 인터뷰 / score : 3
# # 1987 본 문 대통령.."그런다고 바뀌나? 함께 하면 바뀐다" / score : 2
#
# vectorize = TfidfVectorizer(
#     tokenizer=tokenizer,
#     min_df=2,
#
#     sublinear_tf=True  # tf값에 1+log(tf)를 적용하여 tf값이 무한정 커지는 것을 막음
# )
# X = vectorize.fit_transform(rawdata)
#
# print(
#     'fit_transform, (sentence {}, feature {})'.format(X.shape[0], X.shape[1])
# )
# # fit_transform, (sentence 5, feature 7)
#
# print(X.toarray())
#
# # ([[0.        , 0.40824829, 0.81649658, 0.        , 0.        , 0.        , 0.40824829],
# # [0.        , 0.40824829, 0.40824829, 0.        , 0.        , 0.        , 0.81649658],
# # [0.41680418, 0.        , 0.        , 0.69197025, 0.41680418, 0.41680418, 0.        ],
# # [0.76944707, 0.        , 0.        , 0.63871058, 0.        , 0.        , 0.        ],
# # [0.        , 0.        , 0.        , 0.8695635 , 0.34918428, 0.34918428, 0.        ]])
#
# # 문장에서 뽑아낸 feature 들의 배열
# features = vectorize.get_feature_names()
#
# # 검색 문장에서 feature를 뽑아냄
# srch=[t for t in tokenizer('1987 관람한 대통령 인터뷰 기사') if t in features]
# print(srch)
# # ['1987', '대통령']
#
# # dtm 에서 검색하고자 하는 feature만 뽑아낸다.
# srch_dtm = np.asarray(X.toarray())[:, [
#     # vectorize.vocabulary_.get 는 특정 feature 가 dtm 에서 가지고 있는 index값을 리턴한다
#     vectorize.vocabulary_.get(i) for i in srch
# ]]
#
# #   	1987 	대통령
# # 0 	0.000000 	0.000000
# # 1 	0.000000 	0.000000
# # 2 	0.416804 	0.691970
# # 3 	0.769447 	0.638711
# # 4 	0.000000 	0.869563
#
# score = srch_dtm.sum(axis=1)
# print(score)
# # array([0.         0.         1.10877443 1.40815765 0.8695635 ], dtype=int64) 문장별 feature 합계 점수
#
# for i in score.argsort()[::-1]:
#     if score[i] > 0:
#         print('{} / score : {}'.format(rawdata[i], score[i]))
#
# # 1987 본 문 대통령.."그런다고 바뀌나? 함께 하면 바뀐다" / score : 1.408157650537996
# # 문재인 대통령과 대통령의 영부인 김정숙여사 내외의 동반 1987 관람 후 인터뷰 / score : 1.1087744279177436
# # 이명박 전 대통령과 전 대통령의 부인 김윤옥 여사, 그리고 전 대통령의 아들 이시형씨의 동반 검찰출석이 기대됨 / score : 0.869563495264799
#
# ectorize = HashingVectorizer(
#     tokenizer=tokenizer,
#     n_features=7               # 기본 feature 수를 설정하며 기본값이 2의 20승이다. 아래 예시를 위해 feature 를 7로 한정했으나, 아래 유사문장을 찾을때는 다시 n_features 주석처리 했다.
# )
# X = vectorize.fit_transform(rawdata)
#
# print(X.shape)
# # (5, 7)
#
# print(X.toarray())
#
# # ([[ 0.33333333,  0.33333333, -0.33333333,  0.33333333,  0.33333333, 0.66666667,  0.        ],
# # [ 0.        ,  0.        , -0.57735027,  0.57735027,  0.57735027, 0.        ,  0.        ],
# # [ 0.        ,  0.        ,  0.        ,  0.        , -0.21821789, -0.43643578,  0.87287156],
# # [ 0.        ,  0.        ,  0.        ,  0.81649658,  0.        , -0.40824829,  0.40824829],
# # [ 0.28867513,  0.28867513, -0.28867513,  0.28867513, -0.57735027, 0.        ,  0.57735027]])
#
# # search 문장 벡터
# srch_vector = vectorize.transform([
#     '1987 관람한 대통령 인터뷰 기사'
# ])
#
# print(srch_vector.shape)
# # (1, 7)
#
#
# from sklearn.metrics.pairwise import linear_kernel
#
# # linear_kernel는 두 벡터의 dot product 이다.
# cosine_similar = linear_kernel(srch_vector, X).flatten()
# # cosine_similar = (srch_vector*X.T).toarray().flatten()
#
# print(cosine_similar)
# # [0.         0.         0.62017367 0.31622777 0.3]
#
# print(cosine_similar.shape)
# # (5,)
#
# # 유사한 rawdata index
# sim_rank_idx = cosine_similar.argsort()[::-1]
# print(sim_rank_idx)
# #[2 3 4 1 0]
#
# for i in sim_rank_idx:
#     if cosine_similar[i] > 0:
#         print('{} / score : {}'.format(rawdata[i], cosine_similar[i]))
#
# # 문재인 대통령과 대통령의 영부인 김정숙여사 내외의 동반 1987 관람 후 인터뷰 / score : 0.6201736729460423
# # 1987 본 문 대통령.."그런다고 바뀌나? 함께 하면 바뀐다" / score : 0.3162277660168379
# # 이명박 전 대통령과 전 대통령의 부인 김윤옥 여사, 그리고 전 대통령의 아들 이시형씨의 동반 검찰출석이 기대됨 / score : 0.3
#
# # 문장에서 뽑아낸 feature 들의 배열
# features = vectorize.get_feature_names()