### Top 300 Interview-Style Questions

**Topic – Data Structures & Algorithms (in Python)**
*(Questions only, numbered for easy reference. Grouped by sub-topics so you can practise methodically.)*

---

#### 1 – 20 | Time & Space Complexity / Big-O Analysis

1. Define Big-O, Big-Θ, and Big-Ω—how do they differ?
2. Show two different algorithms that are both *O(n log n)* yet one is faster in practice—why?
3. Explain amortised analysis; illustrate with Python’s list `append`.
4. Derive the time complexity of binary search and prove its lower bound.
5. Why is quadratic probing in hash tables considered *O(1)* on average?
6. Compare worst-case vs average-case for quicksort partitions.
7. Analyse the memory footprint of a Python list storing one million integers.
8. What is cache locality and how does it affect linked-list traversal?
9. Explain why Python’s slicing `a[:k]` is *O(k)*, not *O(1)*.
10. Prove that any comparison-based sort has Ω(n log n) lower bound.
11. Show how exponentiation by squaring reduces complexity from *O(n)* to *O(log n)*.
12. Describe Master Theorem cases; give an example for each.
13. When does constant-factor optimisation matter more than asymptotic gains?
14. Outline a real-world scenario where *O(1)* algorithm still performs poorly.
15. Define “pseudo-polynomial” time; give an algorithm that runs in it.
16. Explain big-O for recursive Fibonacci vs memoised Fibonacci.
17. Why is computing median of unsorted array in worst-case *O(n)*?
18. Discuss time–space trade-off in Bloom filters.
19. Show complexity of union-find with path compression and union-by-rank.
20. Explain why a perfectly balanced BST guarantees *O(log n)* search.

---

#### 21 – 40 | Python Built-ins & Idioms

21. Internally, how does `dict` achieve *O(1)* average lookup?
22. Compare `collections.deque` vs list for queue operations.
23. Explain why `heapq` implements a min-heap and how to create a max-heap.
24. Show two ways to perform binary search with `bisect`.
25. Describe the time complexity of `set.add` and circumstances when it degrades.
26. How does `itertools.islice` enable lazy windowed iteration?
27. Discuss `array.array` vs NumPy array for numeric storage.
28. Use `sortedcontainers` to maintain a sliding median—why simpler than manual BST?
29. Demonstrate constant-time stack min query using two Python lists.
30. Explain `functools.lru_cache`—what data structure backs the cache?
31. Show a Pythonic way to flatten nested lists with recursion limits considered.
32. How does `typing.NamedTuple` compare with `dataclass` for immutability?
33. Implement a circular buffer using `collections.deque` in O(1).
34. Why is `random.choice` *O(1)* while `random.sample` is *O(k)*?
35. Illustrate stable vs unstable sort behaviour with Python’s `sorted`.
36. Explain `hashlib` use for building a consistent hashing ring.
37. Show two ways to deduplicate a list while preserving order.
38. Describe how `weakref.WeakValueDictionary` aids cache eviction.
39. Use `graphlib.TopologicalSorter` to topologically order tasks.
40. Explain how `numpy.argsort` utilises tim-sort differences on small vs large arrays.

---

#### 41 – 60 | Arrays & Strings

41. Reverse an array in place without slicing.
42. Find the first non-repeating character in a string in O(n).
43. Explain two-pointer technique for removing duplicates from sorted array.
44. Rotate an array right by *k* steps using reversal algorithm.
45. Given an array, find the maximum sub-array sum (Kadane) and prove its correctness.
46. Detect if array contains a 132 pattern subsequence in O(n).
47. Explain prefix-sum array and compute range sum queries quickly.
48. Design an algorithm to find majority element (> n/2 occurrences).
49. Determine if two strings are anagrams using counting array vs hashing.
50. Implement substring search using KMP and discuss complexity.
51. Explain sliding-window technique for longest substring without repeating chars.
52. How do you merge two sorted arrays in place?
53. Discover the duplicate and missing number in array 1…n in O(n).
54. Explain array “Dutch national flag” partitioning.
55. Find kth largest element using quickselect; compare heap approach.
56. Describe monotonic stack to compute daily temperatures warmer values.
57. Given a binary string, count substrings with equal 0s and 1s.
58. Explain difference between stable partition vs standard partition.
59. Find median of two sorted arrays in O(log n).
60. Evaluate expression with +, –, \*, / using two stacks (Dijkstra’s shunting yard).

---

#### 61 – 80 | Linked Lists

61. Reverse a singly linked list iteratively and recursively.
62. Detect and locate start of cycle using Floyd’s algorithm.
63. Merge two sorted linked lists into one sorted list.
64. Remove kth node from end in single pass.
65. Explain why random access is O(n) in linked lists.
66. Convert binary number represented as linked list to integer.
67. Add two numbers stored in forward-order linked lists.
68. Split circular linked list into two halves.
69. Explain skip list structure and its average complexities.
70. Implement LRU cache with `OrderedDict` vs custom doubly linked list + hash.
71. Copy linked list with random pointers in O(n) time, O(1) extra space.
72. Delete node without head pointer in O(1).
73. Detect intersection node of two singly linked lists.
74. Explain sentinel (dummy) node advantage.
75. Reverse nodes in k-group chunks.
76. Flatten a multilevel doubly linked list.
77. Explain XOR linked list concept; can you implement in CPython?
78. Why does Python lack a built-in linked list type?
79. Partition linked list around a pivot value preserving order.
80. Sort a linked list in O(n log n) using merge sort.

---

#### 81 – 100 | Stacks & Queues

81. Evaluate postfix expression using a stack.
82. Implement queue using two stacks and analyse amortised cost.
83. Design stack supporting `min()` in O(1) per op.
84. Describe bracket matching algorithm and its applications.
85. Convert infix to postfix notation with operator precedence.
86. Implement sliding-window maximum using deque.
87. Explain why recursion uses an implicit call stack.
88. Decode nested string k\[encoded] using stack.
89. Solve Tower of Hanoi iteratively with explicit stack.
90. Design browser forward/back navigation using two stacks.
91. Implement fixed-size queue with circular array.
92. Explain “stack overflow” error and how recursion depth relates.
93. Use monotonic stack to calculate largest rectangle in histogram.
94. Implement median queue (supporting insert & median in O(log n)).
95. Describe producer–consumer queue with thread locks in Python.
96. Explain Josephus problem solution with queue.
97. Design data structure to support `push`, `pop`, and `get_max` in O(1).
98. How does Python’s `asyncio.Queue` differ from `queue.Queue`?
99. Implement k-queues in a single array.
100. Show how to avoid TLE when simulating queue‐based hot-potato game.

---

#### 101 – 120 | Hashing & Sets

101. Explain load factor and resizing strategy for Python dict.
102. Discuss hash collision resolution by separate chaining.
103. Implement two-sum using hash map in O(n).
104. Describe rolling hash for Rabin–Karp substring matching.
105. Detect duplicates within k-distance in array using hashing.
106. Explain unordered vs ordered hashing; why order matters in Python 3.7+.
107. Describe perfect hashing and feasibility.
108. Build phone directory with O(1) lookup for names and numbers.
109. Explain consistent hashing for distributed caches.
110. How do Bloom filters trade memory for false positives?
111. Implement LFU cache using hash + heap.
112. Explain open addressing vs chaining; pros/cons in CPU cache context.
113. Detect isomorphic strings via hashmap mapping.
114. Design data structure that supports insert, delete, and getRandom in O(1).
115. Implement polynomial rolling hash mod large prime; handle overflow.
116. Why are mutable objects unhashable?
117. Show custom `__hash__` and `__eq__` contract pitfalls.
118. Explain hash map implementation of adjacency list for sparse graph.
119. Use set operations to solve array intersection in one line.
120. Discuss cuckoo hashing idea; why rare in high-level languages.

---

#### 121 – 140 | Trees (Basic & Binary)

121. Perform inorder traversal iteratively using stack.
122. Level-order traversal using queue (BFS).
123. Check if two binary trees are identical.
124. Compute height of tree recursively vs iteratively.
125. Find lowest common ancestor (LCA) in binary tree.
126. Convert sorted array to height-balanced BST.
127. Serialize/deserialize binary tree with BFS encoding.
128. Count number of paths summing to target value.
129. Find diameter of binary tree (longest path).
130. Explain DFS preorder, inorder, postorder differences.
131. Mirror (invert) a binary tree.
132. Determine if tree is symmetric recursively.
133. Print right view of binary tree.
134. Explain Morris inorder traversal (O(1) space).
135. Calculate maximum width of binary tree.
136. Reconstruct tree from preorder and inorder sequences.
137. Explain threaded binary trees.
138. Compute vertical order traversal using ordered dict.
139. Flatten binary tree to linked list in preorder.
140. Describe binary indexed tree (Fenwick) and its range update.

---

#### 141 – 160 | BST & Balanced Trees

141. Validate if binary tree is a BST.
142. Search in BST iteratively and recursively.
143. Find kth smallest element in BST.
144. Delete node in BST and maintain validity.
145. Convert BST to doubly linked list in place.
146. What is AVL tree rotation; illustrate LL and LR cases.
147. Compare Red-Black vs AVL balancing strategies.
148. Explain why B-trees are suitable for databases.
149. Implement interval tree and query overlapping intervals.
150. Design order-statistics tree to support select and rank.
151. Describe Treap (tree + heap) randomised balancing.
152. Explain splay tree amortised complexity.
153. How does Python’s `bisect` mimic behavior of balanced BST?
154. Build segment tree for range minimum query.
155. Implement persistent segment tree conceptually.
156. Explain rope data structure for large string edits.
157. Compare skip list vs balanced BST performance.
158. Implement multiset with BST to support duplicate counts.
159. Describe k-d tree for nearest neighbour search.
160. Explain Scapegoat tree rebuilding heuristic.

---

#### 161 – 180 | Heaps & Priority Queues

161. Insert and delete in binary heap; prove complexity.
162. Explain heapify algorithm; why O(n) not O(n log n).
163. Merge k sorted lists using min-heap.
164. Find k closest points to origin in O(n log k).
165. Maintain running median with two heaps.
166. Implement d-ary heap benefits vs binary heap.
167. Explain decrease-key operation in Fibonacci heap.
168. Why aren’t Fibonacci heaps used in practice often?
169. Simulate elevator scheduling using priority queue.
170. Explain heap sort; in-place property.
171. Difference between heapq.nsmallest and partial sort.
172. Design data stream to return kth largest element.
173. Build sliding-window median with heap + lazy deletion.
174. Explain monotone priority queue in Dijkstra optimisation.
175. Use heap to efficiently extract intervals with earliest end time.
176. Implement min stack with heap and index map.
177. Compare pairing heap and binomial heap.
178. Analyse complexity of delete-arbitrary element in heapq.
179. Explain soft heap concept (Chazelle).
180. Build calendar booking system detecting overlaps with min-heap.

---

#### 181 – 200 | Graphs

181. Represent graph with adjacency list in Python; memory analysis.
182. DFS vs BFS—compare use cases.
183. Detect cycle in directed graph using DFS recursion stack.
184. Find connected components in undirected graph.
185. Implement topological sort; explain Kahn’s algorithm.
186. Shortest path in unweighted graph with BFS.
187. Dijkstra’s algorithm using heap; why fails on negative weights.
188. Bellman–Ford algorithm complexity and negative cycle detection.
189. Explain A\* search heuristic requirements.
190. Compute minimum spanning tree via Kruskal vs Prim.
191. Implement Union-Find with path compression for Kruskal.
192. Detect bridges (critical edges) using Tarjan.
193. Find articulation points in graph.
194. Explain strongly connected components with Kosaraju.
195. Count number of islands in 2-D grid.
196. Course schedule feasibility problem (cycle detection).
197. Clone directed graph via DFS/stack.
198. Solve word ladder transform length with BFS.
199. Explain bidirectional BFS advantage.
200. Evaluate alien dictionary order via topological sort.

---

#### 201 – 220 | Sorting & Searching

201. Stability in sorting—give example where required.
202. Implement quicksort with randomised pivot.
203. Explain insertion sort best-case utility.
204. Merge sort vs tim-sort; highlight Python’s hybrid.
205. Sort nearly sorted array efficiently.
206. Counting sort—when linear time possible?
207. Radix sort for strings; discuss memory cost.
208. Bucket sort for floating-point numbers in \[0, 1).
209. External merge sort for huge files.
210. Binary search variations: first ≥ key, last ≤ key.
211. Ternary search vs binary search for unimodal functions.
212. Order statistics via median of medians.
213. Search rotated sorted array in O(log n).
214. Interpolation search complexity expectation.
215. Explain exponential search for unbounded lists.
216. Pancake sort algorithm; applications?
217. Sleep sort gimmick—why unreliable.
218. Top-k frequent elements via bucket vs heap.
219. Stooge sort complexity analysis (fun).
220. Describe stable in-place merge algorithm challenges.

---

#### 221 – 240 | Recursion & Divide-and-Conquer

221. Solve merge k sorted linked lists with divide-and-conquer.
222. Derive recurrence for Karatsuba integer multiplication.
223. Explain tail recursion optimisation absence in CPython.
224. Count total nodes in full k-ary tree using recursion.
225. Determine number of ways to climb stairs with 1 or 2 steps.
226. Generate permutations of array recursively.
227. Implement power(x, n) recursively in O(log n).
228. Compute nth Catalan number via recursion + memo.
229. Solve subset sum using backtracking with pruning.
230. Describe recursion tree method for T(n)=2T(n/2)+n.
231. Explain divide-and-conquer closest pair of points.
232. Count inversions in array using modified merge sort.
233. Perform quickselect recursion depth analysis.
234. Implement recursive binary search iterative fallback.
235. Recurse to generate Gray code sequence.
236. Solve N-Queens with bitmask recursion.
237. Explain master theorem proof using recursion tree.
238. Compute convolution via FFT divide-and-conquer.
239. Construct segment tree recursively.
240. Discuss disadvantages of deep recursion in Python and mitigation.

---

#### 241 – 260 | Dynamic Programming & Memoisation

241. Define overlapping subproblems and optimal substructure.
242. Fibonacci DP bottom-up vs top-down memory.
243. 0/1 knapSack tabulation and complexity.
244. Longest common subsequence DP formulation.
245. Edit distance (Levenshtein) with O(mn) time, O(min) space.
246. Coin change ways vs minimum coins—two different DPs.
247. Rod-cutting problem and revenue memoisation.
248. Boolean parenthesisation count.
249. Matrix chain multiplication order optimisation.
250. Longest increasing subsequence O(n log n) algorithm explanation.
251. Wildcard string matching DP.
252. Palindromic substring table vs expand-around-center.
253. Minimum jumps to reach end array DP.
254. Egg dropping puzzle DP with optimized moves.
255. Maximum profit with k stock transactions DP.
256. Word break problem using trie + DP.
257. DP on trees for maximum independent set.
258. Bitmask DP for travelling salesman on ≤ 20 nodes.
259. Explain memory-efficient DP for knapSack using 1-D array.
260. Discuss pipelining DP into GPU kernels (CUDA) conceptually.

---

#### 261 – 280 | Greedy Algorithms

261. Activity selection proof of greedy optimality.
262. Huffman coding tree construction steps.
263. Fractional knapSack vs 0/1 knapSack difference.
264. Explain interval partitioning greedy approach.
265. Minimum platforms required at railway station problem.
266. Dijkstra’s algorithm as greedy choice.
267. Prim’s MST vs Kruskal greedy rationale.
268. Gas station circle completion problem.
269. Majority meeting rooms free schedule merging.
270. Jump game greedy solution vs DP.
271. Polish notation shortest superstring approximate greedy.
272. Arrange numbers to form largest value greedy trick.
273. Kruskal with DSU cycle rejection reasoning.
274. Greedy coin change fails for {1,3,4}; explain.
275. Minimise sum of array elements plus operation greedy.
276. Scheduling lectures to minimise late penalty.
277. Min number of arrows to burst balloons.
278. Partition labels greedy algorithm proof.
279. Find minimum number of conference rooms required.
280. Greedy algorithm for page-replacement (optimal) vs LRU.

---

#### 281 – 300 | Advanced Structures & Algorithmic Design

281. Explain union-find with rollback (persistence).
282. Describe suffix array construction vs suffix tree.
283. Build LCP array using Kasai algorithm.
284. Explain KMP failure function derivation.
285. Describe prefix function for Z-algorithm.
286. Implement rolling hash in string matching.
287. Explain Mo’s algorithm for offline range queries.
288. Sparse table for RMQ; preprocessing vs query time.
289. Bitset trick to speed subset convolution.
290. Describe binary lifting for LCA in trees.
291. Heavy-light decomposition for path queries.
292. Centroid decomposition in divide-and-conquer on trees.
293. Explain segment tree with lazy propagation.
294. Build implicit treap for dynamic sequence.
295. Describe Euler tour technique for subtree queries.
296. Explain bloomier filters vs bloom filters.
297. Implement double-ended priority queue (min-max heap).
298. Discuss complexity of FFT and its applications in convolution.
299. Explain van Emde Boas tree operations.
300. Describe I/O-efficient (cache-oblivious) B-tree variant.

---

**Tip for Practice:** Tackle one sub-topic each study session. For coding-heavy questions, implement in a Python notebook or REPL—time yourself and note edge-cases. Good luck mastering data structures & algorithms!
