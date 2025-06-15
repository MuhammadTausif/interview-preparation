### Core Python Programming — 300 Interview-Style Questions

*(Numbered for easy reference. Only questions are provided so you can practise recalling the answers yourself.)*

---

#### 1 – 40 | Language Basics & Syntax

1. What are the two phases of Python’s execution model?
2. How does Python’s interactive mode differ from script mode?
3. Explain the difference between `==` and `is`.
4. Why are integers in Python said to be “immutable”?
5. What is iterable unpacking and where can it appear besides assignment?
6. Show three ways to swap two variables without a temporary variable.
7. What is the purpose of the trailing comma in tuple literals?
8. Describe short-circuit evaluation in Boolean expressions.
9. How does `__future__` import affect syntax?
10. What does the walrus operator (`:=`) return inside a comprehension?
11. Explain how chained comparison operators are evaluated (`a < b < c`).
12. Why does `0.1 + 0.2 == 0.3` evaluate to `False`?
13. Contrast implicit string concatenation and `+` concatenation.
14. What is a “truthy” vs “falsy” value; list five falsy built-ins.
15. How does integer division differ between Python 2 and Python 3?
16. Why can’t you have a variable named `True`?
17. Show how to create a multi-line string without newline characters.
18. What happens if you modify a list while iterating over it?
19. Explain name mangling for variables beginning with `__`.
20. Describe the LEGB rule for variable resolution.
21. How do keyword-only parameters work and why were they added?
22. Differentiate positional-only, positional-or-keyword, and kw-only params.
23. What is the default mutability issue with function parameters?
24. Show how to force a function to accept no more than one positional arg.
25. Why is `zip(*list_of_tuples)` called a Python “trick”?
26. How does `range` differ from `xrange` (historically)?
27. Explain generator exhaustion and how to reuse a generator’s results.
28. Describe how slicing works with negative indices and steps.
29. What is the result of `[::-1]` on a sequence?
30. How does `bytes` differ from `bytearray`?
31. Explain how Python encodes source files (the `coding:` header).
32. When can a `finally` block be skipped?
33. Why is `pass` sometimes required inside an `except`?
34. What are f-strings and how do they compare to `format`?
35. Explain the difference between `locals()` and `globals()`.
36. How are backslashes interpreted inside raw strings?
37. What does `__slots__` do and why might you use it?
38. Explain reference cycles and how Python’s GC handles them.
39. Why are docstrings not comments?
40. What happens when you execute `del obj`?

#### 41 – 80 | Built-in Data Structures & Algorithms

41. Compare list vs tuple memory footprints.
42. How is a Python list implemented internally?
43. Explain hash table open addressing in Python dicts.
44. Why is `dict` insertion ordered since Python 3.7?
45. How does `set` handle hash collisions?
46. Show an O(1) algorithm to find duplicates using sets.
47. Why is slicing a list an O(k) operation?
48. Explain the time complexity of checking membership in a set.
49. Describe when you would prefer `collections.deque` over list.
50. How do you implement a priority queue using `heapq`?
51. Why shouldn’t you mutate a key used in a dictionary?
52. Compare shallow copy vs deep copy for nested dicts.
53. Show two idioms to flatten a nested list.
54. What data structure backs `NamedTuple` and how is it memory-efficient?
55. Explain ordered set behaviour using `dict` keys.
56. Demonstrate converting two lists into a mapping with comprehension.
57. How do `Counter` and `defaultdict` differ in typical use?
58. When does list concatenation become quadratic?
59. Implement quickselect in pure Python.
60. What is the difference between `in` for strings vs for lists?
61. Show how to rotate a list in O(1) with `deque`.
62. Explain the GIL’s impact on list append in multi-threading.
63. Why can’t sets contain other mutable sets?
64. Describe the algorithmic complexity of `dict.popitem()`.
65. What is array.array and when is it preferable to NumPy?
66. Implement Breadth-First Search using a queue from `collections`.
67. Compare bisect vs linear search for sorted lists.
68. Explain the Schwartzian Transform pattern in sorting.
69. How does `tim-sort` optimise runs in Python’s sort?
70. Write a stable sort key that ignores accents in Unicode.
71. What is a memoryview and when would you use one?
72. Show two ways to merge two sorted iterables lazily.
73. Explain the difference between `reversed()` and `list[::-1]`.
74. Why is tuple hash cached?
75. Describe the limitations of deepcopy with recursive structures.
76. When does modifying a list comprehension variable leak into scope?
77. Explain EAFP vs LBYL with dict access.
78. Demonstrate constant-time removal from the end of a list.
79. Why is a generator expression more memory-efficient than list comp?
80. Show how to implement a bloom filter using bitarray or int masks.

#### 81 – 120 | Object-Oriented & Functional Programming

81. Describe the descriptor protocol (`__get__`, `__set__`, `__delete__`).
82. Compare classmethod vs staticmethod use cases.
83. How does multiple inheritance resolve attribute lookup (MRO)?
84. Explain `super()` and its zero-arg form.
85. Why might you implement `__new__` instead of `__init__`?
86. Show a mixin pattern for JSON serialisation.
87. Explain polymorphism with duck typing vs ABCs.
88. What is metaclass programming and one practical use?
89. Demonstrate how to make an enum that behaves like an int.
90. Why is composition often preferred over inheritance?
91. Explain `@functools.total_ordering`.
92. Describe how `functools.cache` differs from `lru_cache`.
93. Give an example of currying with `functools.partial`.
94. How do you implement a decorator that preserves signature?
95. Explain first-class functions in Python with an example.
96. Compare closures vs classes for maintaining state.
97. Why can default args in closures be used to remember loop variables?
98. Implement a context manager using `@contextmanager`.
99. Describe weak references and their typical use case.
100. What is a proxy object in `weakref`?
101. Show how to overload arithmetic operators in a vector class.
102. How does `property` assist encapsulation?
103. Explain the flyweight pattern with string interning.
104. When would you override `__repr__` vs `__str__`?
105. Describe slots-based immutable dataclass.
106. How do you enforce singleton with metaclass?
107. Explain method resolution when both instance and class attribute overlap.
108. What are abstract base classes and how do they help type-checking?
109. Show implementing iterable protocol (`__iter__`, `__next__`).
110. Why shouldn’t you rely on `__del__` for resource cleanup?
111. What is covariance vs contravariance in typing?
112. Explain monkey-patching and its risks.
113. Demonstrate functional pipeline using generator comprehensions.
114. Describe memoisation and pitfalls with mutable args.
115. How can `@singledispatch` provide function overloading?
116. Explain structural pattern matching (`match … case`).
117. Compare dataclass `order=True` vs defining rich-compare manually.
118. What is dynamic attribute access via `__getattr__`?
119. Why can bound method objects hold strong refs to instance?
120. Show implementing an ABC enforcing async interface.

#### 121 – 160 | Modules, Packages & Environments

121. Explain how Python finds modules (import machinery).
122. What does `__init__.py` do in modern namespace packages?
123. Compare absolute vs relative imports.
124. Describe import-time side-effects and mitigation strategies.
125. How do you lazily import a heavy dependency?
126. Explain import hooks and one debugging use.
127. Describe editable installs (`pip install -e .`).
128. Contrast venv, conda env, and poetry virtual environments.
129. Why is `sys.meta_path` important?
130. What are implicit namespace packages (PEP 420)?
131. Explain how to make a module executable with `python -m`.
132. Show how to create a console-script entry point in `pyproject.toml`.
133. How does `importlib.reload` work?
134. Describe wheels vs source distributions.
135. What problem does PEP 517 solve?
136. Explain packaging version schemes (PEP 440).
137. What is dependency resolution backtracking in pip 23+?
138. How do you freeze exact dependencies reproducibly?
139. Explain environment variable `PYTHONPATH`.
140. What is `sitecustomize.py`?
141. Show how to write a plugin system using entry points.
142. Explain `zipapp` and when you might ship one.
143. Describe how to build a standalone Python binary with PyInstaller.
144. How can namespace collisions occur in site-packages?
145. What is hatchling’s role in building packages?
146. Explain three ways to run unit tests automatically on install.
147. Describe stub packages (`types-requests`) and their purpose.
148. How do you vendor a dependency safely?
149. Explain hash-based `.pyc` files (PEP 552).
150. Why does compiling C extensions require a build backend?
151. What is `importlib.resources` and how does it handle data files?
152. Show how to package CLI + library in one project.
153. Explain wheel “abi” tags.
154. What is `pkg_resources` and why is it considered slow?
155. Describe `sys.modules` caching behaviour.
156. How do you inspect an import lock?
157. Explain how to create a namespace collision intentionally (for tests).
158. Contrast poetry lock file vs pip tools requirements-in.txt.
159. Describe the security implications of `pip install`.
160. What are “vectorised wheels” on PyPI and why do they matter?

#### 161 – 200 | Standard Library & Tools

161. Compare `argparse`, `click`, and `typer` for CLI creation.
162. How does `logging` hierarchy propagate handlers?
163. Why use `pathlib` over `os.path`?
164. Explain `contextlib.ExitStack`.
165. How does `Enum.auto()` pick values?
166. Describe parsing configs with `configparser` vs `pydantic`.
167. What is `dataclasses.asdict` caveat with mutable fields?
168. How does `itertools.groupby` differ from SQL GROUP BY?
169. Demonstrate `itertools.cycle` memory characteristics.
170. Explain `functools.reduce` and why it’s less Pythonic than a loop.
171. Show a recipe with `collections.ChainMap`.
172. Describe `fractions.Fraction` exact arithmetic vs float.
173. How does `decimal` handle rounding modes?
174. Explain the usage of `secrets` module over `random`.
175. Why use `uuid.uuid4()` for unique IDs?
176. Show scheduling tasks with `sched` vs `time.sleep`.
177. Describe `subprocess.run` security pitfalls with user input.
178. Explain `shlex.split`.
179. Compare `multiprocessing.shared_memory` to `Queue`.
180. How does `signal` module handle `SIGINT` on Windows?
181. Explain `os.fork()` vs spawn in `multiprocessing`.
182. Demonstrate reading zip contents with `zipfile.Path`.
183. What is `tempfile.TemporaryDirectory` context manager?
184. How do you safely delete a directory tree?
185. Explain reading gzip files lazily with `gzip.open`.
186. Describe `pickle` protocol versions and backward compatibility.
187. Why avoid pickling untrusted data?
188. Compare `json.dumps` default behaviour to `orjson`.
189. Explain `importlib.metadata.version`.
190. What is `concurrent.futures` and its thread vs process pools?
191. Demonstrate a simple profiler with `cProfile` and `pstats`.
192. How does `traceback.format_exc` help logging?
193. Show using `warnings` to deprecate a function.
194. Explain `atexit` handlers.
195. Describe timezone handling with `datetime` and `zoneinfo`.
196. What is `graphlib.TopologicalSorter` used for?
197. Explain `types.MappingProxyType`.
198. How to simulate network delays in `unittest` using `time.sleep` patch.
199. Describe `contextvars` for async context propagation.
200. Explain specialised dict (`collections.UserDict`) subclass use.

#### 201 – 240 | Concurrency, Async & Parallelism

201. Explain the Global Interpreter Lock (GIL) in CPython.
202. Contrast IO-bound vs CPU-bound workloads in threading.
203. How does `asyncio` event loop schedule tasks?
204. Demonstrate creating a coroutine and running it.
205. Explain `await` vs `yield from`.
206. Describe cancellation in `asyncio` (`asyncio.CancelledError`).
207. What are `asyncio.gather` vs `asyncio.wait`.
208. How do you impose a timeout on an async call?
209. Explain backpressure handling with `asyncio.Queue`.
210. Show using `aiohttp` for concurrent HTTP requests.
211. Compare `multiprocessing` vs `concurrent.futures.ProcessPool`.
212. How does `ThreadPoolExecutor` leverage threads under GIL?
213. Explain `futures.as_completed`.
214. Describe “fan-in/fan-out” concurrency pattern in Python.
215. Why are daemon threads terminated abruptly?
216. Demonstrate using locks to protect shared state in threads.
217. How do `multiprocessing.Value` and `Array` share memory?
218. Explain deadlock and how to detect with `threading` debug.
219. Show barrier synchronisation example.
220. Describe `multiprocessing.Pool.imap`.
221. What is `uvloop` and when does it improve performance?
222. Compare trio vs asyncio.
223. Explain cooperative multitasking in gevent.
224. Show using `queue.SimpleQueue` for multi-process communication.
225. How does `asyncio.run_in_executor` bridge blocking code?
226. Explain event-driven architecture vs classic multi-threading.
227. Describe signal handling in child processes.
228. What is the overhead of process spawn on Windows vs fork on Unix?
229. Show using shared-memory `numpy` arrays across processes.
230. Explain parallelism trade-offs of joblib vs Dask vs Ray.
231. How does `asyncio.Semaphore` limit concurrency?
232. Describe pipelining tasks with `asyncio.create_task`.
233. Explain back-pressure in producer–consumer model.
234. Compare cooperative vs pre-emptive multitasking in Python context.
235. Show how to profile async code for latency bottlenecks.
236. Why is thread-safety important for Django ORM connections?
237. Describe GIL contention and how to detect with perf counters.
238. Explain C-extension workaround to bypass GIL for heavy numeric code.
239. Show integrating async code with sync via `nest_asyncio`.
240. What is structured concurrency and does Python support it?

#### 241 – 270 | Performance, Memory & Typing

241. How does CPython reference counting work?
242. Explain the difference between `sys.getsizeof` and actual memory used.
243. Describe memory fragmentation in Python and pymalloc arenas.
244. Profile speed using `timeit` vs `perf_counter`.
245. Show how to benchmark list vs deque append.
246. Explain PEP 659 “specialising adaptive interpreter”.
247. How does function call overhead impact micro-benchmarks?
248. Why use `slots` to save memory?
249. Demonstrate lazy evaluation to speed data parsing.
250. Describe vectorisation in NumPy vs pure Python loops.
251. Explain Just-in-Time compilation in PyPy.
252. Compare numba `nopython` mode vs Cython.
253. Describe `memory_profiler` line-by-line analysis.
254. How do arenas cause lingering memory after list deletions?
255. Explain interning of small integers.
256. What are type hints and how do they improve code quality?
257. Describe `typing.Protocol` for structural typing.
258. Explain `mypy --strict` benefits.
259. How does `typing.Any` harm static analysis?
260. Compare union types before and after PEP 604 (`int | str`).
261. Describe `typing.TypedDict`.
262. What is gradual typing?
263. Show using `pydantic` for data validation.
264. Explain `typing.final`.
265. How does Cython static typing speed loops?
266. Demonstrate profiling C extensions with `gprof`.
267. Describe caching strategies with `functools.cache`.
268. Explain copy-on-write in CPython 3.13 preview.
269. What is Python’s “zero-cost exception” mechanism?
270. Show eliminating temporary arrays with NumPy out parameter.

#### 271 – 300 | Testing, Packaging, Patterns & Best Practices

271. Explain AAA (Arrange-Act-Assert) structure in pytest.
272. How does fixture scope work in pytest?
273. Describe mocking external APIs with `responses` library.
274. What is property-based testing with Hypothesis?
275. Explain mutation testing and a Python tool for it.
276. Why follow PEP 8 naming conventions?
277. Describe black vs autopep8 vs isort differences.
278. What is cyclomatic complexity and threshold suggestions?
279. Show writing a simple tox matrix for multiple Python versions.
280. Explain semantic versioning for a package release.
281. Describe wheels’ manylinux compatibility.
282. What’s the builder pattern in Python context?
283. How would you implement observer pattern with callbacks?
284. Explain dependency injection using callable factories.
285. Describe the singleton anti-pattern caveats.
286. How do you secure sensitive configs with dotenv?
287. Explain 12-factor app principles in Python service.
288. Show a reproducible notebook environment using jupyter-kernel spec.
289. What is the `if __name__ == "__main__"` idiom and why vital?
290. Explain graceful shutdown for a Flask API.
291. Describe continuous deployment pipeline with GitHub Actions.
292. How do you automate version bump and changelog generation?
293. Discuss pyproject-toml only builds vs legacy setup.py.
294. Explain how to embed C code with cffi.
295. Describe sqlite WAL mode advantages in embedded apps.
296. What is a stub file `.pyi` and how is it generated?
297. Explain Sphinx autodoc for documentation generation.
298. Describe code coverage metrics and thresholds.
299. Show pattern to freeze random seeds for reproducible ML experiments.
300. Summarise the Zen of Python and provide two examples of “practicality beats purity”.

---

**Next step:** Try answering 5–10 questions at a time, then verify with official docs or quick Python REPL experiments. Happy practising!
