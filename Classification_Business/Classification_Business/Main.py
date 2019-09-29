from gensim import corpora, matutils
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
import pickle
from pathlib import Path

#引数check
if len(sys.argv) < 7 :
    print("【Warn】引数が指定されていません。モデルの第一候補、第二候補、第三候補とVectorizer、処理対象のフォルダパスと結果の出力先を引数として指定してください。")
    sys.exit

#モデル読み込み
loaded_model1 = pickle.load(open(sys.argv[1], 'rb'))
loaded_model2 = pickle.load(open(sys.argv[2], 'rb'))
loaded_model3 = pickle.load(open(sys.argv[3], 'rb'))

#テキスト読み込み
p = Path(sys.argv[5])
filePathList = list(p.glob("**/*.txt"))

texts = [open(path, encoding='UTF-8').readlines() for path in filePathList]
names = [os.path.basename(path)[:4] for path in filePathList]

#textsを分かち書き
from MeCabShell import MeCabShell
texts = [[MeCabShell.Analysis(0,line.rstrip(os.linesep)) for line in text] for text in texts]
texts = [[flatten for inner in text for flatten in inner] for text in texts]

#2次元配列を作る。
np.set_printoptions(precision=2)

#TfidfVectorizerをロード（fit済）
vectorizer = pickle.load(open(sys.argv[4], "rb"))

#テスト用データ作成
tfidf = vectorizer.transform(np.array(texts)).toarray()

#評価
result1 = loaded_model1.predict(tfidf)
result2 = loaded_model2.predict(tfidf)
result3 = loaded_model3.predict(tfidf)

#csvにoutput
with open(sys.argv[6], 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, lineterminator='\n')
    for i in range(len(names)):
        writer.writerow([names[i], result1[i],result2[i],result3[i]])