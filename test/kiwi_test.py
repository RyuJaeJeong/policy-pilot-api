from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

string = "수습 기간이 있나요?"
kiwi = Kiwi()
stopwords = Stopwords()
print(kiwi.tokenize(string, normalize_coda=True, stopwords=stopwords))