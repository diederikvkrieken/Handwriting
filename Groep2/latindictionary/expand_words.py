import re


def unique(seq, idfun=None):
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

with open ("DICTPAGE.txt", "r") as myfile:
    data=myfile.read().replace('\n', '')


pattern = "#(.*?)  "

matchResult = re.findall(pattern, data)
matchesLines = [i.split(', ') for i in matchResult]

matches = []
i = 0
import string
lower =  string.ascii_lowercase
upper = string.ascii_uppercase
#print len(matches)
for n in range(len(lower)):
    matches.append(lower[n])
    matches.append(upper[n])


for matchLine in matchesLines:
    for word in matchLine:
        i += 1

        if not word.isalpha() and (word != "-" and word != "(gen.)"):
            if "(gen.)" in word:
               word = re.sub("\(gen.\)", "", word)
               matches.append(word)
            elif "." in word:
                word = re.sub("\.", "", word)
                matches.append(word)
            elif "(i)" in word:
                word1 = re.sub("\(i\)", "", word)
                word2 = re.sub("\(i\)", "i", word)
                matches.append(word1)
                matches.append(word2)
            elif "(ii)" in word:
                word1 = re.sub("\(ii\)", "", word)
                word2 = re.sub("\(ii\)", "i", word)
                matches.append(word1)
                matches.append(word2)
            elif "(n)" in word:
                word1 = re.sub("\(n\)", "", word)
                word2 = re.sub("\(n\)", "n", word)
                matches.append(word1)
                matches.append(word2)
            elif "-" in word:
                if "-um" in word:
                    actual_word = word.split()[0]
                    if actual_word.endswith("a"):
                        word = actual_word.replace(actual_word[-1:], 'um')
                    elif actual_word.endswith("us"):
                        word = actual_word.replace(actual_word[-2:], 'um')
                if "-or" in word:
                   word = re.sub("-or", "", word)
                if "-us" in word:
                    actual_word = word.split()[0]
                    word = actual_word.replace(actual_word[-2:], 'us')
                if "-ae" in word:
                    actual_word = word.split()[0]
                    word = actual_word.replace(actual_word[-1:], 'ae')
                if word.endswith('-e'):
                    actual_word = word.split()[0]
                    word = actual_word.replace(actual_word[-3:], 'er')
                if '-a' in word:
                    actual_word = word.split()[0]
                    word = actual_word.replace(actual_word[-2:], 'a')
                if '-u' in word:
                    actual_word = word.split()[0]
                    word = actual_word.replace(actual_word[-2:], 'u')
                if '-ia' in word:
                    actual_word = word.split()[0]
                    word = actual_word.replace(actual_word[-2:], 'ia')
                if '-ius' in word:
                    actual_word = word.split()[0]
                    word = actual_word.replace(actual_word[-2:], 'ius')
                matches.append(word)
                matches.append(actual_word)
            elif '/' in word:
                actual_word = word.split('/')
                if len(actual_word[1])==2:
                    actual_word[1] = actual_word[0].replace(actual_word[0][-2:], 'is')
                else:
                    actual_word[1] = actual_word[1].split()[0]
                matches.append(actual_word[0])
                matches.append(actual_word[1])
            else:
                word = word.split()[0]
                matches.append(word)
        else:
            if word.isalpha():
                matches.append(word)


import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8') # vary depending on your lang/locale
matches = unique(matches)
matches2 = sorted(matches, cmp=locale.strcoll)

if 'Q' in matches2:
    print "hoi"


with open ("WORDS_DICTPAGE2.txt", "w") as myfile:
    for word in matches2:
        myfile.write(word + '\n')
