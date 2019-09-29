from gensim import corpora, matutils
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import numpy as np

#引数check
if len(sys.argv) < 5 :
    print("【Warn】引数が指定されていません。モデルの第一候補、第二候補、第三候補と処理対象のフォルダパスをを引数として指定してください。")
    sys.exit

#モデル読み込み

import pickle
from pathlib import Path

loaded_model1 = pickle.load(open(sys.argv[1], 'rb'))
loaded_model2 = pickle.load(open(sys.argv[2], 'rb'))
loaded_model3 = pickle.load(open(sys.argv[3], 'rb'))


#テキスト読み込み
p = Path(sys.argv[4])
filePathList = list(p.glob("**/*.txt"))

texts = [open(path, encoding='UTF-8').readlines() for path in filePathList]
names = [os.path.basename(path)[:4] for path in filePathList]


#textsを分かち書き
from MeCabShell import MeCabShell
texts = [[MeCabShell.Analysis(0,line.rstrip(os.linesep)) for line in text] for text in texts]

#ベクトル化
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below = 5, no_above = 0.5)

#2次元配列を作る。
np.set_printoptions(precision=2)

#TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=dictionary.doc2bow, use_idf=True, token_pattern=u'(?u)\\b\\w+\\b',min_df=0.05, max_df=0.8)

#テスト用データ作成
train_data = vectorizer.fit_transform(np.array(texts)).toarray()

#標準化
sc = StandardScaler()
sc.fit(train_data)
self.train_data_std = sc.transform(train_data)

#評価


print(texts)
print(names)
#csvにoutput




def readTxtfiles(filePath):
    """
    指定されたフォルダ配下の.txt拡張子のファイルの中身をList形式で返却する。
    Returns
    -------
    texts: str[][]
        指定されたフォルダ配下のファイルを行毎にList化したものをList化
    """
    # Pathオブジェクトを生成
    p = Path(filePath)
    filePathList = list(p.glob("**/*.txt"))

    result = [open(path, encoding='UTF-8') for path in filePathList]

    xmlDataList = []

    return result
