from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from nltk import word_tokenize
import io


class FeatureVector:

    def readData(self, path):
        trainX = []  # our training set features
        trainY = []  # our training set lables

        # first, we read our sentences from training data
        with io.open(path, 'r', encoding='utf8') as train:
            data = train.readlines()

        # now we loop over sentences and do preprocessing
        for row in data:
            text = row.split("\t")
            if len(text[0]) == 1:
                trainX.append(text[1])
                trainY.append(text[0])

        return trainX, trainY

    # we start building our feature vectors here
    def build(self, trainX, trainY):

        X_train = []

        # now we loop over sentences and do preprocessing
        for row in trainX:
            X_train.append(self.preprocess(row))

        # now that we have the preprocessed sentence we start building our vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(X_train)
        X_train = X.toarray()

        return X_train, trainY

    # text preprocessing method
    def preprocess(self, text):

        # first, we clean the text
        text = self.cleaning(text)

        # in this step, we do the following:
        # 1 - tokenizing the text
        # 2 - stemming the tokens
        # 3 - removing stopwords
        text = self.tokenize(text)

        return text

    # spanish tokenizer
    def tokenize(self, text):

        # punctuation to remove
        non_words = list(punctuation)

        # we add spanish punctuation
        non_words.extend(['¿', '¡'])
        non_words.extend(map(str, range(10)))

        # remove punctuation
        text = ''.join([c for c in text if c not in non_words])

        # tokenize
        tokens = word_tokenize(text)

        # spanish stemmer
        stemmer = SnowballStemmer('spanish')
        stems = self.stem_tokens(tokens, stemmer)

        # removing spanish stop words
        spanish_stopwords = stopwords.words('spanish')
        text = ' '.join([c for c in stems if c not in spanish_stopwords])

        return text

    # spanish stemmer
    def stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    # removing extra characters and cleaning the text
    def cleaning(self, text):
        text = text.replace("\n","")
        text = text.replace("]]>","")

        text = text.split(" ")
        # we should remove links from the sentence
        cleaned = ' '.join([c for c in text if "http://" not in c and "@" not in c])

        return cleaned