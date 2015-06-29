#!/usr/bin/python
#By Steve Hanov, 2011. Released to the public domain



class Levenshtein_Distance():
    def __init__(self):
        pass

    # This recursive helper is used by the search function above. It assumes that
    # the previousRow has been filled in already.
    def searchRecursive(self, node, letter, word, previousRow, results, maxCost ):

        columns = len( word ) + 1
        currentRow = [ previousRow[0] + 1 ]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in xrange( 1, columns ):

            insertCost = currentRow[column - 1] + 1
            deleteCost = previousRow[column] + 1

            if word[column - 1] != letter and (word[column - 1] != "s" or word[column - 1] != "f"): #s and f are interchangable, so you dont get a +1
                replaceCost = previousRow[ column - 1 ] + 1
            else:
                replaceCost = previousRow[ column - 1 ]

            currentRow.append( min( insertCost, deleteCost, replaceCost ) )

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= maxCost and node.word != None:
            results.append( (node.word, currentRow[-1] ) )

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min( currentRow ) <= maxCost:
            for letter in node.children:
                self.searchRecursive( node.children[letter], letter, word, currentRow,
                    results, maxCost )

    # The search function returns a list of all words that are less than the given
    # maximum distance from the target word
    def search(self, word, maxCost, trie):
        # build first row
        currentRow = range( len(word) + 1 )

        results = []

        # recursively search each branch of the trie
        for letter in trie.children:
            self.searchRecursive( trie.children[letter], letter, word, currentRow,
                results, maxCost )
        return results

    def decide_winner(self, results):
        sorted_results = sorted(results, key=lambda x: x[1])
        if sorted_results[0][1] != sorted_results[1][1]:
            return sorted_results[0]
        else:
            print("We have multiple winners!")
            return sorted_results[0]

    def run(self, words, trie):
        print("Running the distance!")
        MAX_COST = 2
        potential_winner = []
        for n in words:
            n = n[:]
            if len(n)>1:
                results = self.search(n, MAX_COST ,trie)
                if results and len(results)>1:
                    potential_winner.append(self.decide_winner(results))
                elif results:
                    potential_winner.append(results[0])
                else:
                    potential_winner.append((n, MAX_COST))
            else:
                 potential_winner.append((n, 1))
        winner = self.decide_winner(potential_winner)
        return winner[0]