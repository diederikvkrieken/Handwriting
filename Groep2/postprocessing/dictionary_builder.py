DICTIONARY = "/home/bassie/PycharmProjects/handwriting-recognition2/Groep2/latindictionary/WORDS_DICTPAGE.txt"

# The Trie data structure keeps a set of words, organized with one node for
# each letter. Each node has a branch for each letter that may follow it in the
# set of words.
class TrieNode:
    def __init__(self):
        self.word = None
        self.children = {}

    def insert( self, word ):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()

            node = node.children[letter]

        node.word = word

    def run(self):
        # Keep some interesting statistics
        # read dictionary file into a trie
        trie = TrieNode()
        for word in open(DICTIONARY, "rt").read().split():
            trie.insert( word )

        return trie

