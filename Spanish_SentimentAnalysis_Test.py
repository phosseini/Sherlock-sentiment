import json
import re
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 as \
    features

## This command needs the user to put in his IBM Bluemix generated account username and password.
## When placed in the repository, the username will have "YOUR SERVICE USERNAME/PASSWORD" to prevent
## other users from utilizing my Bluemix account. For actual production, use your own username/password.
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='YOUR SERVICE USERNAME',
    password='YOUR SERVICE PASSWORD')

errors = 0

for i, line in enumerate(open('E:\\Documents\\CSCI_6907\\sample_test_set.txt')):
    res = re.search('\d(?=\t)', line)
    num = res.group(0)
    actual_label = "negative"
    if (num == "0"):
        actual_label = "neutral"
    elif (num == "1"):
        actual_label = "positive"
    elif (num == "2"):
        actual_label = "negative"
    else:
        print("Error~~~")
    response = natural_language_understanding.analyze(
        text=line, features=[features.Sentiment()])
    print(json.dumps(response, indent=2))

    generated_label = response['sentiment']['document']['label']
    if (generated_label != actual_label):
        errors = errors + 1
        print("Actual was: ", actual_label, ". Watson returned: ", generated_label)

accuracy = (1 - (errors / (i + 1))) * 100
print("Watson's Accuracy for this test set is: ", accuracy, "%")
