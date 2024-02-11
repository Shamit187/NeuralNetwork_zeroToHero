_DEBUG = True

def levenshtein_distance(word1, word2):
    len1 = len(word1) + 1
    len2 = len(word2) + 1

    # Create the distance matrix
    dp = [[0 for _ in range(len2)] for _ in range(len1)]

    # Initialize base cases
    for i in range(len1):
        dp[i][0] = i
    for j in range(len2):
        dp[0][j] = j

    # Calculate minimum edit operations
    for i in range(1, len1):
        for j in range(1, len2):
            if word1[i - 1] == word2[j - 1]:
                cost = 0  # Substitution cost if characters match
            else:
                cost = 1  # Substitution cost

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    return dp[len1 - 1][len2 - 1]  # Result at the bottom right of the matrix


class TrieNode:
    def __init__(self):
        self.children = {}
        self.path = None


class AppTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, path):
        current_node = self.root
        if current_node.path is None:
            current_node.path = path.split(" > ")[0]

        increasing_path = ""
        for word in path.split(" > "):
            increasing_path = increasing_path + " > " + word if increasing_path else word
            if word not in current_node.children:
                current_node.children[word] = TrieNode()
            current_node.children[word].path = increasing_path
            current_node = current_node.children[word]

    def search(self, query, threshold=10):
        results = []

        def _dfs_query(node: TrieNode, query: str):
            if node.children == {}:
                return
            
            if _DEBUG:
                print("Searching", node.children)

            for child_word, child_node in node.children.items():
                distance = levenshtein_distance(query, child_word)
                if _DEBUG:
                    print("Distance", distance, child_word, query)
                if distance <= threshold:
                    results.append(child_node.path)
                else :
                    _dfs_query(child_node, query)
        
        _dfs_query(self.root, query)
        return results
    
# Construct the trie for testing
trie = AppTrie()
trie.insert("select_post > new_comment > add_image")
trie.insert("settings > battery > toggle_battery_saver")
trie.insert("select_post > new_comment")
trie.insert("select_post")


print("output:", trie.search("comment_on_image", 10))
