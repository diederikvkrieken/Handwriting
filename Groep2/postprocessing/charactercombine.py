from dictionary_builder import TrieNode
from levenshtein_dist import Levenshtein_Distance

class charactercombine():

    # Combines a sequence of character predictions to a word
    def combineChar(self, segments):
        word = ""   # Empty list to append characters to
        # Consider all characters predicted
        for char in segments:
            if len(char)==1:
                word = word + char[0]
            elif char != '**GARBAGE**':
                match_added = False
                match = ''
                for current_char in char[0]:
                    if current_char==word[-1:] and match_added==False:
                        match = current_char
                    else:
                        if current_char=='_' and match!=word[-1:]:
                            word = word+match
                            match_added = True
                        elif current_char!= '_':
                            word = word+current_char
                        match = ''
        return word


    def run(self, ppPredictions, type):
        if type == 1:#then we are validating!
            charPredictions = []
            for n in range(0,len(ppPredictions),5):
                all_letters = ppPredictions[n:n+5]
                charPredictions.append(all_letters)
            ppPredictions = charPredictions[:]

        trie = TrieNode().run()
        winner = [None]*len(ppPredictions)
        allpredictions = [None]*len(ppPredictions)
        count = 0

        for pred in ppPredictions:

            predicted_winners = [None]*len(pred)
            for n in range(len(pred)):
                predicted_winners[n] = self.combineChar(pred[n])
            allpredictions[count] = predicted_winners
            winner[count] = Levenshtein_Distance().run(predicted_winners, trie)
            count += 1

        return winner