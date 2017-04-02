from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from nltk import word_tokenize


class FeatureVector:

    # we start building our feature vector here
    def build(self):
        trainX = [] # our training set features
        trainY = [] # our training set lables

        # first, we read our sentences from training data
        with open('data/train.txt', 'r') as train:
            data = train.readlines()

        # now we loop over sentences and do preprocessing
        for row in data:
            text = row.split("\t")
            if len(text[0]) == 1:
                trainX.append(self.preprocess(text[1]))
                trainY.append(text[0])

        # now that we have the preprocessed sentence we start building our vectors
        # from sklearn.feature_extraction.text import CountVectorizer
        # vectorizer = CountVectorizer(min_df=1)
        # vectorizer.fit_transform(trainX)

        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(min_df=1)
        X = vectorizer.fit_transform(trainX)
        trainX = X.toarray()

        return trainX, trainY

    # we do text preprocessing in this method
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

    # removing extra characters and cleaning the text
    def cleaning(self, text):
        text = text.replace("\n","")
        return text

    # stemming tokens
    def stem_tokens(self, tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed