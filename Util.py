import io


class Util:

    def readData(self, path):

        X = []  # our training set features
        y = []  # our training set lables

        # first, we read our sentences from training data
        with io.open(path, 'r', encoding='utf8') as train:
             data = train.readlines()

        # now we loop over sentences and separate labels and sentences
        for row in data:
            text = row.split("\t")
            if len(text[0]) == 1:
                X.append(text[1])
                y.append(text[0])

        return X, y