class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_terminal = False
        self.path = None
        self.max_distance =  float('inf')   # Initial Levenshtein threshold

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
                dp[i - 1][j] + 1,     # Deletion 
                dp[i][j - 1] + 1,     # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[len1 - 1][len2 - 1]  # Result at the bottom right of the matrix

class AppLayoutTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, path):
        current_node = self.root
        for word in path.split(" > "):
            current_node = current_node.children.setdefault(word, TrieNode())
        current_node.is_terminal = True
        current_node.path = path
 
    def search(self, query, threshold=10):  # Default tolerance
        results = []

        def _dfs(node):

            if node.is_terminal:
                return

            for child_word, child_node in node.children.items():
                distance = levenshtein_distance(query, child_word)
                if distance <= threshold:
                    # path found
                    results.append(child_node.path)
                else:
                    _dfs(child_node)

                    
        _dfs(self.root)
        return results

# Construct the trie for testing
trie = AppLayoutTrie()
trie.insert("select_post > new comment > add_image")
trie.insert("settings > battery > toggle_battery_saver")
trie.insert("select_post > new_comment")
trie.insert("select_post")

# Test Cases from Your Description
# test_cases = [
#     {"input":"toggle_battery_saver","output" :"settings > battery > toggle_battery_saver"},
#     {"input":"turn_on_battery_saver","output":"settings > battery > toggle_battery_saver"},
#     {"input":"turn_of_battery_saver","output":"settings > battery > toggle_battery_saver"},
#     {"input":"comment_on_image","output": "select_post > new_comment > add_image"},
#     {"input":"comment_pn_post","output": "select_post > new_comment"},
#     {"input":"see_post","output": "select_post"}
# ]

# # Run tests
# for case in test_cases:
#     result = trie.search(case["input"])
#     assert result == [case["output"]], f"Failed for input: {case['input']}"

# print("All tests passed!")

print(trie.root.children)