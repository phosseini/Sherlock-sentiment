import nltk
from sklearn.feature_extraction.text import CountVectorizer

class FeatureVector:

    # we start building our feature vector here
    def build(self):

        trainX = [] # our training set features
        trainY = [] # our training set lables

        # first, we read our sentences from training data
        with open('data/train.txt', 'r') as train:
            data = train.readlines()

        # now we loop over our sentences and do preprocessing
        for row in data:
            text = row.split("\t")
            if len(text[0]) == 1:
                trainX.append(self.preprocess(text[1]))
                trainY.append(text[0])

        # now that we have the preprocessed sentence we start building our vectors
        vectorizer = CountVectorizer(min_df=1)
        X = vectorizer.fit_transform(trainX)

        return X

    # we do text preprocessing in this method
    def preprocess(self, text):
        text = self.cleaning(text)
        # spanish_tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")
        # print(spanish_tokenizer.tokenize(text))
        return text

    # removing extra characters and cleaning the text
    def cleaning(self, text):
        text = text.replace("\n","")
        return text