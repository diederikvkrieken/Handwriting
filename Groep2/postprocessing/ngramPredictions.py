__author__ = 'diederik'

class pathBuilder():

    def __init__(self):
        pass

    def build(self, segmentOptions, ngram, actualWord = 'not given!'):


        #Determines how many top result we return:
        topLimit = 5

        # Reference dictionary
        levelArray = []

        # Add the two start nodes
        levelArray.append([[u'start', [[[u'start'], 1.0]]]])
        levelArray.append([[u'start', [[[u'start', u'start'], 1.0]]]])

        for char in segmentOptions[0]:
            levelArray.append([])

        # Build the level array, This is where every node is created for every letter.
        for word in segmentOptions:

            charCounter = 0
            for char in word:
                levelArray[charCounter+2].append([char, []])
                charCounter += 1


        # Add the end node
        levelArray.append([[u'end', []]])

        """
        print "---------- ", actualWord, " ------------"
        """
        """
        levelCounter = 0
        for level in levelArray:
            print "LEVEL %d: " % (levelCounter), " ", level
            levelCounter += 1
        # LevelArray [[Node Name], TOP 5 LIST[[List With traveled nodes], [probability]]]
        # Node = [NodeName], TOP 5 LIST[[list with traveled nodes], probability]
        # print levelArray
        """
        """
        fcount = 0
        for word in segmentOptions:
            print "Feature %i: %s" % (fcount, word[:])
            fcount += 1
        """

        # Start creating the paths
        levelCount = 2
        for level in levelArray[2:]:

            for currentNode in level:

                #print "Current node Change to: ", currentNode
                optionList = [[], []]

                for prevNode in levelArray[levelCount-1]:

                    # print "LENGTH: ", len(prevNode[1]), "<--------------------------------------------------"
                    #print "Current Node: ", currentNode[0], " Prev Node: ", prevNode
                    for topPic in prevNode[1]:

                        # print topPic
                        # Build Tri gram
                        trigram = "" + topPic[0][-2] + " " + topPic[0][-1] + " " + currentNode[0][:]

                        P = ngram.getProbability(trigram)

                        newProbability = P * topPic[1]
                        optionList[1].append(list(topPic[0]))
                        optionList[1][-1].append(currentNode[0])
                        optionList[0].append(newProbability)


                # Select the best 5 options and continue.
                optionList = sorted(zip(optionList[0],optionList[1]), key=lambda pair: pair[0])[::-1]

                for option in optionList[0:topLimit]:
                    currentNode[1].append([option[1], option[0]])

            levelCount += 1


        """
        for result in levelArray[-1][0][1]:
            print result[0][:]
        """

        # Return the top five of the last node
        results = []

        for result in levelArray[-1][0][1]:
            results.append(result[0][2:-1])

        return results

    def run(self, segmentsOptions, ngram):

        results = []
        for options in segmentsOptions:
            results.append(self.build(options, ngram))

        return results



