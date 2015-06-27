import re

with open ("DICTPAGE.txt", "r") as myfile:
    data=myfile.read().replace('\n', '')


pattern = "#(.*?)  "

matchResult = re.findall(pattern, data)
matchesLines = [i.split(', ') for i in matchResult]

matches = []
i = 0
for matchLine in matchesLines:
    for word in matchLine:
        i += 1
        print i
        if word not in matches:
            matches.append(word)


with open ("WORDS_DICTPAGE.txt", "w") as myfile:
    for word in matches:
        myfile.write(word + '\n')
