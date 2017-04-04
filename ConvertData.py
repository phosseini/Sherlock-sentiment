import re

f = open('result.txt', "w")
flg = True

for i, line in enumerate(open('training.txt')):

    if (flg):
        result = re.search('(?<=\<content\>\<\!\[CDATA\[).*', line)
        if (result is not None):
            res = result.group(0)
            #print(res)
            text = res
            #f.write('\"'+ res + '\"'+ ',')
            flg = False
    else:
        result = re.search('(?<=\<value\>)\w+', line)
        if (result is not None):
            res = result.group(0)
            #print(res)
            if ((res == "NONE") or (res == "NEU")):
                tag = "0"
                #f.write("0\n")
            elif ((res == "P") or (res == "P+")):
                tag = "1"
                #f.write("1\n")
            elif ((res == "N") or (res == "N+")):
                tag = "2"
                #f.write("2\n")
            else:
                print("Unknown tag")
            flg = True
            f.write(tag + '\t' + text + '\n')
f.close()
