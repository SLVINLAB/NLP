# --*-- coding:utf-8 --*--
import gensim

import jpype
import numpy as np
try:
    from  AI.AIEngine.database import db_session
except:
    from database  import db_session
import pandas as pd
# dump tf-idf into file
import _pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Twitter ,Komoran
import os
import re

twitter = Twitter()
komoran = Komoran()
stoplist = []

class Singleton(type):
    instance = None

    def __call__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls.instance

class datamanagerAll():
    def __init__(self):
        print('datamanagerAll')

        #리포트요약본
        self.tot_textreviewsRP = []
        self.tot_titlesRP = []
        self.tot_gicodesRP = []

        #비지니스 써머리
        self.tot_textreviewsBS = []
        self.tot_titlesBS = []
        self.tot_gicodesBS = []

        #공시의 사업목적
        self.tot_textreviewsDS = []
        self.tot_titlesDS = []
        self.tot_gicodesDS = []

        #한경 기사
        self.tot_textreviewsHK = []
        self.tot_titlesHK = []
        self.tot_gicodesHK = []

        #이투데이 기사
        self.tot_textreviewsET = []
        self.tot_titlesET = []
        self.tot_gicodesET = []

    def getDataReport(self,update = False):
        update = False
        modelFile1 = 'dfReport.csv'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile1)
        if update :
            #디비에서 가져와서 저장한다
            def read_query(query="""SELECT SUBSTR(GICODE_ARRAY,0,7) CODE ,title || SYNOPSIS || SYNOPSIS_DETAIL1 || SYNOPSIS_DETAIL2 || SYNOPSIS_DETAIL3 BIZ_STATUS_BRIEF ,B.ITEMABBRNM --,bullet_DT
            FROM FNR_BULLET_REP A , FNS_J_MAST B
            WHERE bullet_DT > '20150101' AND LENGTH(GICODE_ARRAY) = 8 AND GICODE_ARRAY != 'A000000&' AND A.GICODE_ARRAY = B.GICODE || '&'"""):
                cursor = db_session.cursor()
                try:
                    cursor.execute(query)
                    names = [x[0] for x in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=names)
                finally:
                    if cursor is not None:
                        cursor.close()

            dfBusiness = read_query()
            dfBusiness.to_csv(file_path1)
            return dfBusiness
        else:
            #dfBusiness = pd.read_csv(file_path1,encoding='cp949')
            dfBusiness = pd.read_csv(file_path1, encoding='utf-8')
            return dfBusiness

    def getDataSummary(self,update = False):
        update = False
        modelFile1 = 'dfSummary.csv'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile1)
        if update :
            #디비에서 가져와서 저장한다
            def read_query(query="""
SELECT A.GICODE CODE,BIZ_STATUS_BRIEF,B.ITEMABBRNM  FROM FNJ_CORP_SUMMARY A , FNS_J_MAST B WHERE gs_ym = (SELECT max(gs_ym) FROM FNJ_CORP_SUMMARY) AND BIZ_STATUS_BRIEF IS NOT NULL AND A.GICODE = B.GICODE"""):
                cursor = db_session.cursor()
                try:
                    cursor.execute(query)
                    names = [x[0] for x in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=names)
                finally:
                    if cursor is not None:
                        cursor.close()

            dfBusiness = read_query()
            dfBusiness.to_csv(file_path1)
            return dfBusiness
        else:
            #dfBusiness = pd.read_csv(file_path1,encoding='cp949')
            dfBusiness = pd.read_csv(file_path1, encoding='utf-8')
            return dfBusiness

    def getDataDisclouse(self, update=False):
            update = False
            modelFile1 = 'dfDisclosure.csv'
            module_dir = os.path.dirname(__file__)  # get current directory
            file_path1 = os.path.join(module_dir, modelFile1)
            if update:
                # 디비에서 가져와서 저장한다
                def read_query(query="""
    SELECT A.GICODE CODE,B.ITEMABBRNM,OBJ_BIZ BIZ_STATUS_BRIEF FROM fnj_ab A,FNS_J_MAST B WHERE gs_ym> 201711 AND A.GICODE = B.GICODE ORDER BY A.GICODE,BIZ_STATUS_BRIEF"""):
                    cursor = db_session.cursor()
                    try:
                        cursor.execute(query)
                        names = [x[0] for x in cursor.description]
                        rows = cursor.fetchall()
                        return pd.DataFrame(rows, columns=names)
                    finally:
                        if cursor is not None:
                            cursor.close()

                dfBusiness = read_query()
                dfDisclosureS_onerow = pd.DataFrame(columns={'CODE', 'ITEMABBRNM','BIZ_STATUS_BRIEF'})
                strPrevious = ''
                strBizLong =''
                for idx, row in dfBusiness.iterrows():
                    strGicode = row['CODE']
                    strGiName = row['ITEMABBRNM']
                    strBiz = row['BIZ_STATUS_BRIEF']
                    if strPrevious == strGicode:
                        strBizLong = strBizLong + ' ' + strBiz
                        dfDisclosureS_onerow = dfDisclosureS_onerow[dfDisclosureS_onerow.CODE != strGicode]
                        dfDisclosureS_onerow = dfDisclosureS_onerow.append({"CODE": strGicode,"ITEMABBRNM":strGiName, "BIZ_STATUS_BRIEF": strBizLong},
                                                                           ignore_index=True)
                    else:
                        strBizLong = strBiz
                        dfDisclosureS_onerow = dfDisclosureS_onerow.append({"CODE": strGicode,"ITEMABBRNM":strGiName, "BIZ_STATUS_BRIEF": strBiz},
                                                                           ignore_index=True)
                    strPrevious = strGicode

                print(dfDisclosureS_onerow)
                dfDisclosureS_onerow.to_csv(file_path1)
                #dfBusiness.to_csv(file_path1)
                return dfDisclosureS_onerow
            else:
                #dfBusiness = pd.read_csv(file_path1, encoding='cp949')
                dfBusiness = pd.read_csv(file_path1, encoding='utf-8')
                return dfBusiness

    def getDataHANKYUNG(self, update=False):
        update = False
        modelFile1 = 'dfHANKYUNG.csv'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile1)
        if update:
            # 디비에서 가져와서 저장한다
            def read_query(query="""
    SELECT A.GICODE CODE,(B.NEWS_TITLE || c.CONV_NEWS_BODY) BIZ_STATUS_BRIEF,D.ITEMABBRNM FROM FNC_NEWS_J_CD A,FNC_NEWS_MAST B ,FNC_NEWS_BODY C ,FNS_J_MAST D  WHERE a.NEWS_KEY = b.NEWS_KEY AND a.NEWS_KEY = C.NEWS_KEY AND A.GICODE = D.GICODE AND A.NEWS_KEY in (SELECT NEWS_KEY FROM UFNGDBA.FNC_NEWS_J_CD GROUP BY NEWS_KEY HAVING count(NEWS_KEY) = 1)
"""):
                cursor = db_session.cursor()
                try:
                    cursor.execute(query)
                    names = [x[0] for x in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=names)
                finally:
                    if cursor is not None:
                        cursor.close()

            dfBusiness = read_query()
            dfBusiness.to_csv(file_path1)
            return dfBusiness
        else:
            dfBusiness = pd.read_csv(file_path1, encoding='utf-8')
            return dfBusiness

    def getDataETODAY(self, update=False):
        update = False
        modelFile1 = 'dfETODAY.csv'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile1)

        modelFile2 = 'dfJMAST.csv'
        module_dir1 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir1, modelFile2)
        if update:
            # 디비에서 가져와서 저장한다
            def read_query(query="""
    SELECT CATEGORY_NM,ATC_TITLE,ATC_CTNS FROM UFNGDBA.FNC_NEWS_ETODAY"""):
                cursor = db_session.cursor()
                try:
                    cursor.execute(query)
                    names = [x[0] for x in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=names)
                finally:
                    if cursor is not None:
                        cursor.close()

            # 디비에서 가져와서 저장한다
            def read_query2(query="""
                SELECT GICODE,ITEMABBRNM FROM FNS_J_MAST"""):
                cursor = db_session.cursor()
                try:
                    cursor.execute(query)
                    names = [x[0] for x in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=names)
                finally:
                    if cursor is not None:
                        cursor.close()

            dfJMAST = read_query2()
            dfBusiness = read_query()

            dfResult = pd.merge(dfJMAST, dfBusiness, on='ITEMABBRNM', how='inner')
            dfResult.to_csv(file_path1)

            return dfResult
        else:
            # dfETODAY_NEW = pd.read_csv("new_ETODAY_NEW.csv", encoding='utf-8')
            #
            # def strSplitComa(strHead):
            #     strGicode = str(strHead).split(',')
            #     print(strGicode[0])
            #     return strGicode[0]
            #
            # def strSplitBlank(strHead):
            #     strGicode = str(strHead).split(' ')
            #     print(strGicode[0])
            #     return strGicode[0]
            #
            # def strRemove(strHead):
            #     strGicode = str(strHead).replace('[답변공시]', '')
            #     strGicode = str(strGicode).replace('[조회공시]', '')
            #     return strGicode
            #
            # def strRemoveTAG(strHead):
            #     strGicode = str(strHead).replace('<br /><br />', '')
            #     return strGicode
            #
            # # dfETODAY_NEW['ITEMABBRNM'] = dfETODAY['ATC_TITLE'].apply(strSplitComa)
            # # dfETODAY_NEW['BIZ_STATUS_BRIEF'] = dfETODAY['ATC_CTNS']
            #
            # dfETODAY_NEW['ITEMABBRNM'] = dfETODAY_NEW['ITEMABBRNM'].apply(strRemove)
            # dfETODAY_NEW['ITEMABBRNM'] = dfETODAY_NEW['ITEMABBRNM'].apply(strSplitBlank)
            # dfETODAY_NEW['BIZ_STATUS_BRIEF'] = dfETODAY_NEW['BIZ_STATUS_BRIEF'].apply(strSplitBlank)
            #
            # print(dfETODAY_NEW)
            # # dfETODAY_NEW.to_csv('new_ETODAY_NEW.csv')
            # # dfETODAY_NEW = pd.read_csv("new_ETODAY_NEW.csv", encoding='utf-8')
            # dfJMAST = pd.read_csv(file_path2, encoding='utf-8')
            # dfResult = pd.merge(dfJMAST, dfETODAY_NEW, on='ITEMABBRNM', how='inner')
            # dfResult.to_csv(file_path1)
            dfResult= pd.read_csv(file_path1, encoding='utf-8')
            return dfResult


    def PreprocessTfidf(self,texts, stoplist=[], stem=False):
        jpype.attachThreadToJVM()
        newtexts = []
        for text in texts:
            # tmp = text.split(' ')
            # newtexts.append(' '.join(tmp))
            try:
                tmp = twitter.morphs(text)
                newtexts.append(' '.join(tmp))
            except:
                newtexts.append(' '.join(' '))
                pass

        return newtexts

    def get_vectorizer(self,dfData , update = False ,sourceType = 'BS'):
        update = False
        '''5개타입이 있음 현재 1,비지니스써머리(BS),2.리포트요약(RP), 3.전자공시(DS),4.한경기사(HK), 5.ETODAY 기사(ET)'''
        modelFile1 = 'vectorizer_' + sourceType + '.pk'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile1)

        modelFile2 = 'vec_tfidf_' + sourceType + '.pk'
        module_dir2 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir2, modelFile2)

        for idx, item in dfData.iterrows():
            if sourceType == 'BS':
                self.tot_textreviewsBS.append(item["BIZ_STATUS_BRIEF"])
                self.tot_titlesBS.append(item["ITEMABBRNM"])
                self.tot_gicodesBS.append(item["CODE"])
            elif sourceType == 'RP':
                self.tot_textreviewsRP.append(item["BIZ_STATUS_BRIEF"])
                self.tot_titlesRP.append(item["ITEMABBRNM"])
                self.tot_gicodesRP.append(item["CODE"])
            elif sourceType == 'DS':
                self.tot_textreviewsDS.append(item["BIZ_STATUS_BRIEF"])
                self.tot_titlesDS.append(item["ITEMABBRNM"])
                self.tot_gicodesDS.append(item["CODE"])
            elif sourceType == 'HK':
                self.tot_textreviewsHK.append(item["BIZ_STATUS_BRIEF"])
                self.tot_titlesHK.append(item["ITEMABBRNM"])
                self.tot_gicodesHK.append(item["CODE"])
            elif sourceType == 'ET':
                self.tot_textreviewsET.append(item["BIZ_STATUS_BRIEF"])
                self.tot_titlesET.append(item["ITEMABBRNM"])
                self.tot_gicodesET.append(item["CODE"])

        vectorizer = TfidfVectorizer(min_df=1)
        if update:
            if sourceType == 'BS':
                processed_reviews = self.PreprocessTfidf(self.tot_textreviewsBS, stoplist, True)
            elif sourceType == 'RP':
                processed_reviews = self.PreprocessTfidf(self.tot_textreviewsRP, stoplist, True)
            elif sourceType == 'DS':
                processed_reviews = self.PreprocessTfidf(self.tot_textreviewsDS, stoplist, True)
            elif sourceType == 'HK':
                processed_reviews = self.PreprocessTfidf(self.tot_textreviewsHK, stoplist, True)
            elif sourceType == 'ET':
                processed_reviews = self.PreprocessTfidf(self.tot_textreviewsET, stoplist, True)

            mod_tfidf = vectorizer.fit(processed_reviews)
            vec_tfidf = mod_tfidf.transform(processed_reviews)
            # 데이타 만드는 작업
            with open(file_path1, 'wb') as fin:
                pickle.dump(mod_tfidf, fin)

            with open(file_path2, 'wb') as fin:
                pickle.dump(vec_tfidf, fin)

            return mod_tfidf,vec_tfidf
        else:
            # if sourceType == 'ET':
            #     processed_reviews = self.PreprocessTfidf(self.tot_textreviewsET, stoplist, True)
            #
            #     mod_tfidf = vectorizer.fit(processed_reviews)
            #     vec_tfidf = mod_tfidf.transform(processed_reviews)
            #     # 데이타 만드는 작업
            #     with open(file_path1, 'wb') as fin:
            #         pickle.dump(mod_tfidf, fin)
            #
            #     with open(file_path2, 'wb') as fin:
            #         pickle.dump(vec_tfidf, fin)
            #
            #     return mod_tfidf,vec_tfidf
            # else:
            # 데이타 로딩 작업
            file = open(file_path1, 'rb')
            mod_tfidf = pickle.load(file)
            file = open(file_path2, 'rb')
            vec_tfidf = pickle.load(file)
            return mod_tfidf ,vec_tfidf

    def getThema(self,mod_tfidf,vec_tfidf,query,sourceType = 'BS'):
        from sklearn.metrics.pairwise import cosine_similarity
        query_vec = mod_tfidf.transform(self.PreprocessTfidf([' '.join(query)], stoplist, True))
        sims = cosine_similarity(query_vec, vec_tfidf)[0]
        indxs_sims = sims.argsort()[::-1]
        dfResult = pd.DataFrame(columns={"종목명", "코드", "내용"})
        if sourceType == 'BS':
            for d in list(indxs_sims)[:200]:
                if sims[d] > 0:
                    dfResult = dfResult.append(
                        {'종목명': self.tot_titlesBS[d], '코드': self.tot_gicodesBS[d], "내용": self.tot_textreviewsBS[d]},
                        ignore_index=True)
            dfResult = dfResult.drop_duplicates(['종목명', '코드'], keep='first')
        elif sourceType == 'RP':
            for d in list(indxs_sims)[:200]:
                if sims[d] > 0:
                    dfResult = dfResult.append(
                        {'종목명': self.tot_titlesRP[d], '코드': self.tot_gicodesRP[d], "내용": self.tot_textreviewsRP[d]},
                        ignore_index=True)
            dfResult = dfResult.drop_duplicates(['종목명', '코드'], keep='first')
        elif sourceType == 'DS':
            for d in list(indxs_sims)[:200]:
                if sims[d] > 0:
                    dfResult = dfResult.append(
                        {'종목명': self.tot_titlesDS[d], '코드': self.tot_gicodesDS[d], "내용": self.tot_textreviewsDS[d]},
                        ignore_index=True)
            dfResult = dfResult.drop_duplicates(['종목명', '코드'], keep='first')

        elif sourceType == 'HK':
            for d in list(indxs_sims)[:200]:
                if sims[d] > 0:
                    dfResult = dfResult.append(
                        {'종목명': self.tot_titlesHK[d], '코드': self.tot_gicodesHK[d], "내용": self.tot_textreviewsHK[d]},
                        ignore_index=True)
            dfResult = dfResult.drop_duplicates(['종목명', '코드'], keep='first')
        elif sourceType == 'ET':
            for d in list(indxs_sims)[:200]:
                if sims[d] > 0:
                    dfResult = dfResult.append(
                        {'종목명': self.tot_titlesET[d], '코드': self.tot_gicodesET[d], "내용": self.tot_textreviewsET[d]},
                        ignore_index=True)
            dfResult = dfResult.drop_duplicates(['종목명', '코드'], keep='first')

        return dfResult

    def tokenizer_twitter_morphs(self, doc):
        return twitter.morphs(doc)

    def tokenizer_twitter_noun(self, doc):
        return twitter.nouns(doc)

    def tokenizer_twitter_pos(self, doc):
        return twitter.pos(doc, norm=True, stem=True)

    def tokenizer_noun(self, doc):
        return komoran.nouns(doc)

    def tokenizer_morphs(self, doc):
        return komoran.morphs(doc)

    def text_split(self, doc):
        return str(doc).split(' ')

    def text_remove(self, doc):
        return str(doc).replace("<br /><br />",'')

    def get_model(self, update=True):
        update = False
        modelFile = 'word2vecModel.pk'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile)

        modelFile2 = 'dfReport.csv'
        module_dir2 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir2, modelFile2)

        modelFile3 = 'dfSummary.csv'
        module_dir3 = os.path.dirname(__file__)  # get current directory
        file_path3 = os.path.join(module_dir3, modelFile3)

        if update == True:
            dfBusiness = pd.read_csv(file_path2, encoding='utf-8')
            dfSummary = pd.read_csv(file_path3, encoding='utf-8')
            dfThema = pd.concat([dfBusiness, dfSummary])
            dfThema['token_BIZ_STATUS_BRIEF'] = dfThema['BIZ_STATUS_BRIEF'].apply(self.tokenizer_twitter_morphs) + \
                                                dfThema['BIZ_STATUS_BRIEF'].apply(
                                                       self.tokenizer_morphs)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)

            from gensim.models.word2vec import Word2Vec
            model = Word2Vec(dfThema['token_BIZ_STATUS_BRIEF'],min_count= 10)
            model.init_sims(replace=True)
            # 데이타 만드는 작업
            with open(file_path1, 'wb') as fin:
                pickle.dump(model, fin)
            return model
        else:
            # 데이타 로딩 작업
            file = open(file_path1, 'rb')
            model = pickle.load(file)
            return model

    def get_model_disclosure(self, update=True):
        update = False
        modelFile = 'word2vecModel_disclosure.pk'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile)

        modelFile2 = 'dfDisclosure.csv'
        module_dir2 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir2, modelFile2)

        if update == True:
            dfBusiness = pd.read_csv(file_path2, encoding='utf-8')
            dfThema = dfBusiness
            dfThema['token_BIZ_STATUS_BRIEF'] = dfThema['BIZ_STATUS_BRIEF'].apply(self.tokenizer_twitter_morphs) + \
                                                dfThema['BIZ_STATUS_BRIEF'].apply(
                                                       self.tokenizer_morphs)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)

            from gensim.models.word2vec import Word2Vec
            model = Word2Vec(dfThema['token_BIZ_STATUS_BRIEF'],min_count= 10)
            model.init_sims(replace=True)
            # 데이타 만드는 작업
            with open(file_path1, 'wb') as fin:
                pickle.dump(model, fin)
            return model
        else:
            # 데이타 로딩 작업
            file = open(file_path1, 'rb')
            model = pickle.load(file)
            return model

    def get_modelWIKI(self, update=False, keyword=""):
        update = False
        modelFile = 'wiki.ko.vec'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile)

        modelFile2 = 'wiki.pk'
        module_dir2 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir2, modelFile2)

        strResult = ""
        if update == True:
            model = gensim.models.KeyedVectors.load_word2vec_format(file_path1)
            # 데이타 만드는 작업
            with open(file_path2, 'wb') as fin:
                pickle.dump(model, fin)
            #return model
        else:
            # 데이타 로딩 작업
            file = open(file_path2, 'rb')
            model = pickle.load(file)
            #return model
        try:
            result = model.most_similar(keyword, topn=300)
            strResult = "<br><br> 관련된 WIKI 키워드는 <br/>"
            arrayResult= []
            arrayResult2 = []
            jpype.attachThreadToJVM()
            strKey= ''

            for idx in result:
                strKey = strKey +' '+ idx[0]

            arrayResult = self.tokenizer_twitter_morphs(strKey)
            arrayResult2= self.tokenizer_morphs(strKey)
            arrayResultF = set(arrayResult).union(set(arrayResult2))
            arrayResultF = list(set(arrayResultF))
            str1 = ','.join(str(e) for e in arrayResultF)
            strResult = strResult + ',' + str1
            #arrayResult = list(set(arrayResult))
            print(model.wv.most_similar(keyword, topn=300))
        except:
            strResult = "관련된 키워드는 현재 없습니다."
        return strResult

    def get_modelWIKI_KO(self, update=False):
        update = False
        #update = False
        modelFile = 'wiki.ko.mine.vec'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile)

        modelFile2 = 'wiki.mine.pk'
        module_dir2 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir2, modelFile2)

        strResult = ""
        if update == True:
            path_dir = os.path.join(module_dir2, "text")
            listWIKI = []
            for root, dirs, files in os.walk(path_dir):
                for file in files:
                    filepath = root + '/' + file
                    print(root + '/' + file)
                    f = open(filepath, 'r', encoding='utf-8')
                    data = f.read()
                    arrayResult = data.split('</doc>')
                    for text in arrayResult:
                        arrayResult2 = text.split('">\n')
                        try:
                            stringSummary = arrayResult2[1]
                            parseString = re.sub('[^가-힣A-Za-z0-9\s]+', '', stringSummary)
                            print(parseString)
                            # listWIKI.append(arrayResult2[1].replace('\xa0','').replace('\n',''))
                            listWIKI.append(parseString)
                        except:
                            pass

                    # print(arrayResult)
                    f.close()
            pdWIKI = pd.DataFrame(listWIKI, columns=["wiki"])
            jpype.attachThreadToJVM()
            pdWIKI['token_wiki'] = pdWIKI['wiki'].apply(self.tokenizer_twitter_morphs) + \
                                   pdWIKI['wiki'].apply(
                                       self.tokenizer_morphs)

            from gensim.models.word2vec import Word2Vec
            model = Word2Vec(pdWIKI['token_wiki'],min_count= 10)
            model.init_sims(replace=True)
            # 데이타 만드는 작업
            with open(file_path2, 'wb') as fin:
                pickle.dump(model, fin)
            return model
        else:
            # 데이타 로딩 작업
            file = open(file_path2, 'rb')
            model = pickle.load(file)
            return model

        # try:
        #     result = model.most_similar(keyword, topn=300)
        #     strResult = "<br><br> 관련된 WIKI 키워드는 <br/>"
        #     arrayResult= []
        #     arrayResult2 = []
        #     jpype.attachThreadToJVM()
        #     strKey= ''
        #
        #     for idx in result:
        #         strKey = strKey +' '+ idx[0]
        #
        #     arrayResult = self.tokenizer_twitter_morphs(strKey)
        #     arrayResult2= self.tokenizer_morphs(strKey)
        #     arrayResultF = set(arrayResult).union(set(arrayResult2))
        #     arrayResultF = list(set(arrayResultF))
        #     str1 = ','.join(str(e) for e in arrayResultF)
        #     strResult = strResult + ',' + str1
        #     #arrayResult = list(set(arrayResult))
        #     print(model.wv.most_similar(keyword, topn=300))
        # except:
        #     strResult = "관련된 키워드는 현재 없습니다."
        # return strResult

    def get_model_HANKYUNG(self, update=False):
        update = False
        modelFile = 'word2vecModel_HANKYUNG.pk'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile)

        modelFile2 = 'dfHANKYUNG.csv'
        module_dir2 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir2, modelFile2)

        if update == True:
            jpype.attachThreadToJVM()
            dfBusiness = pd.read_csv(file_path2, encoding='utf-8')
            dfThema = dfBusiness
            dfThema['token_BIZ_STATUS_BRIEF'] = dfThema['BIZ_STATUS_BRIEF'].apply(self.tokenizer_twitter_morphs) + \
                                                dfThema['BIZ_STATUS_BRIEF'].apply(
                                                       self.tokenizer_morphs)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)

            from gensim.models.word2vec import Word2Vec
            model = Word2Vec(dfThema['token_BIZ_STATUS_BRIEF'],min_count= 10)
            model.init_sims(replace=True)
            # 데이타 만드는 작업
            with open(file_path1, 'wb') as fin:
                pickle.dump(model, fin)
            return model
        else:
            # 데이타 로딩 작업
            file = open(file_path1, 'rb')
            model = pickle.load(file)
            return model

    def get_model_ETODAY(self, update=False):
        update = False
        modelFile = 'word2vecModel_ETODAY.pk'
        module_dir = os.path.dirname(__file__)  # get current directory
        file_path1 = os.path.join(module_dir, modelFile)

        modelFile2 = 'dfETODAY.csv'
        module_dir2 = os.path.dirname(__file__)  # get current directory
        file_path2 = os.path.join(module_dir2, modelFile2)

        if update == True:
            jpype.attachThreadToJVM()
            dfBusiness = pd.read_csv(file_path2, encoding='utf-8')
            dfBusiness["BIZ_STATUS_BRIEF"] = dfBusiness["BIZ_STATUS_BRIEF"].apply(self.text_remove)
            #dfBusiness.to_csv(file_path2)
            dfThema = dfBusiness
            dfThema['token_BIZ_STATUS_BRIEF'] = dfThema['BIZ_STATUS_BRIEF'].apply(self.tokenizer_twitter_morphs) + \
                                                dfThema['BIZ_STATUS_BRIEF'].apply(
                                                       self.tokenizer_morphs)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)

            from gensim.models.word2vec import Word2Vec
            model = Word2Vec(dfThema['token_BIZ_STATUS_BRIEF'],min_count= 10)
            model.init_sims(replace=True)
            # 데이타 만드는 작업
            with open(file_path1, 'wb') as fin:
                pickle.dump(model, fin)
            return model
        else:
            # 데이타 로딩 작업
            file = open(file_path1, 'rb')
            model = pickle.load(file)
            return model
    # def get_model_Brief(self, dfResult):
    #     dfBusiness = dfResult
    #     # dfBusiness['token_BIZ_STATUS_BRIEF'] = dfBusiness['내용'].apply(self.tokenizer_twitter_noun) + \
    #     #                                        dfBusiness['내용'].apply(
    #     #                                            self.tokenizer_noun)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)
    #     dfBusiness['token_BIZ_STATUS_BRIEF'] = dfBusiness['내용'].apply(self.tokenizer_twitter_morphs) + \
    #                                            dfBusiness['내용'].apply(
    #                                                self.tokenizer_morphs)  # + dfBusiness['BIZ_STATUS_BRIEF'].apply(text_split)
    #
    #     from gensim.models.word2vec import Word2Vec
    #     model = Word2Vec(dfBusiness['token_BIZ_STATUS_BRIEF'])
    #     model.init_sims(replace=True)
    #     return model

if __name__ == '__main__':
    da = datamanagerAll()
    #print(da.getDataETODAY())
    # print(da.getDataHANKYUNG())
    #da.get_model_HANKYUNG(True)
    da.get_model_ETODAY(True)
    # da.get_model(True)
    # da.get_model_disclosure(True)
    # da.get_modelWIKI_KO(True)
    print('go')









