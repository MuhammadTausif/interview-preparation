### NumPy Fundamentals & Advanced Learning — 300 Interview-Style Questions

*(Questions only, numbered for easy reference. Grouped by sub-topics so you can practise methodically.)*

---

#### 1 – 20 | Array Creation & Basics

1. How do `np.array`, `np.asarray`, and `np.asarray_chkfinite` differ?
2. What is the default dtype of `np.zeros` on Windows vs Linux, and why?
3. Create a 3 × 4 array filled with the value 7 **without** using `np.full`.
4. Explain `np.arange` vs `np.linspace`; when is one preferred?
5. How can integer overflow occur in `np.arange`, and how to prevent it?
6. Demonstrate building a checkerboard pattern with one call to `np.indices`.
7. Use `np.fromfunction` to generate a multiplication table.
8. How does `np.empty` differ from `np.zeros` in initialisation cost?
9. Drawbacks of using Python list comprehensions for large numeric data.
10. Show three ways to create an identity matrix, and benchmark them.
11. Explain the purpose of `dtype=object` arrays.
12. Describe how `as_strided` can craft a view without copying.
13. Why is `np.eye(3, k=-1)` useful in linear-system algorithms?
14. Construct a 100-element bool array alternating True/False efficiently.
15. Use `np.repeat` and `np.tile` to build a 1-D wave pattern `[0 1 2 0 1 2 …]`.
16. Explain why `np.matrix` is discouraged in modern code.
17. Generate 1 GB of random bytes in NumPy—what function and dtype?
18. Explain row-major vs column-major order flags during creation.
19. Use `np.tri` to build a lower-triangular mask; show an application.
20. Build a 5-D array of ones where the physical memory footprint is only 8 bytes.

---

#### 21 – 40 | Indexing, Slicing & Views

21. Contrast basic slicing vs fancy indexing in view vs copy semantics.
22. Why can chained slicing `a[::2][:,1]` surprise you?
23. Use boolean masking to zero-out all negative values in place.
24. Explain how ellipsis (`...`) helps with high-dimensional arrays.
25. Demonstrate selecting every third column of a 2-D array.
26. What is the difference between `arr.flat` and `arr.ravel()`?
27. Show two ways to swap axes 0 and 2 of a 3-D array.
28. Use multi-index arrays to gather elements at positions `[(0,2),(3,1)]`.
29. Explain `np.take_along_axis` and when to prefer it over fancy indexing.
30. Demonstrate `np.put` to scatter values into a flat view.
31. How does `np.unravel_index` aid in translating flat indices?
32. Compute sliding 3-element windows using only stride tricks.
33. Describe the memory effect of `arr.T`—copy or view?
34. Show why `arr[:,None]` is equivalent to `arr.reshape(-1,1)`.
35. Explain negative step in slicing (`a[::-1]`) and its stride sign.
36. What happens if slice start > stop with positive step?
37. How to select an arbitrary set of rows and columns simultaneously.
38. Demonstrate delimiter splitting on a structured text file via `np.loadtxt` and advanced dtype indexing.
39. Use `np.expand_dims` vs `None` insertion; any performance difference?
40. Explain the role of `order='K'` when raveling.

---

#### 41 – 60 | Dtypes & Casting

41. Why does `np.int32(2) / np.int32(3)` yield a float64?
42. Show three ways to safely cast float32 array to float16.
43. Describe alignment requirements for structured dtypes.
44. Explain sub-dtype fields (`[('r', '<f4'), ('g', '<f4')]`).
45. When is `np.dtype('U10')` preferable to `np.dtype('S10')`?
46. Demonstrate creating a dtype alias for a C structure with padding.
47. Explain byte-order marks in NumPy dtypes (`'<', '>', '='`).
48. Why can `arr.astype(np.int64, copy=False)` still copy memory?
49. Use `np.can_cast` to check safe vs same-kind casting.
50. Explain `np.result_type` for mixed-dtype ufunc operands.
51. How do NaNs propagate in `float16` vs `bfloat16`?
52. Describe limitations of datetime64 with ± \~ 500 million years range.
53. What is a “void” dtype and a real-world use case?
54. Convert RGB uint8 image to float32 \[0,1] range in-place.
55. Use `np.view()` to reinterpret a float64 buffer as two float32 numbers.
56. Explain why `np.array([1,2,3], dtype=bool)` is valid.
57. Describe the memory representation of `complex128`.
58. Create a structured dtype representing a polynomial’s coefficients of varying length—why tricky?
59. What happens when you assign a Python float to a `np.uint8` array?
60. Compare `itemsize` vs `nbytes`.

---

#### 61 – 80 | Broadcasting & Vectorisation

61. State the two formal rules of NumPy broadcasting.
62. Broadcast a (5,1,4) array with a (1,3,1) array—show resulting shape.
63. Explain why broadcasting is memory-cheap yet compute-dense.
64. Use broadcasting to compute pairwise Euclidean distances without loops.
65. Demonstrate outer addition of two vectors via `[:,None]` idiom.
66. Why can broadcasting lead to unintuitive automatic dtype upcasting?
67. Compare `np.add.outer` vs manual broadcasting.
68. Show a gotcha with in-place broadcasted assignment.
69. Use `np.broadcast_to` and explain when it raises `ValueError`.
70. Compute softmax along axis 1 using stable exponent trick.
71. Summation trick to compute column means without explicit loops.
72. Broadcast a scalar into a masked array—any caveats?
73. Explain memory-layout pitfalls of unnecessarily broadcasting to full size.
74. Demonstrate computing a grayscale image from RGB by weighted sum via broadcasting.
75. Use Einstein summation (`np.einsum`) to compute matrix trace.
76. Explain why `np.outer` on 1-D arrays yields a 2-D result.
77. Fuse element-wise operations via `np.sqrt(a**2 + b**2)`—why compute twice?
78. Use `np.ufunc.reduce` along multiple axes after broadcasting.
79. Implement a batched dot-product using `np.matmul` broadcasting.
80. Describe the concept of “universal functions are broadcast-aware.”

---

#### 81 – 100 | Universal Functions & Aggregations

81. Explain type-signature dispatching in ufuncs (`float32,float32->float32`).
82. How does `out=` keyword avoid temporary allocations?
83. Implement the Huber loss using only ufuncs and broadcasting.
84. Describe reduce vs accumulate vs reduceat.
85. When is `np.add.reduce(a, dtype=np.int64)` required?
86. Create a custom ufunc via `np.frompyfunc`; what’s the limitation?
87. Why are `np.maximum`/`np.minimum` NaN-propagating?
88. Use `np.clip` to saturate image data efficiently.
89. Explain `where=` masked reduction argument added in NumPy 1.17.
90. Compare `np.sum` vs `arr.sum()` on multi-axis performance.
91. Demonstrate computing rolling cumulative sum via `np.add.accumulate`.
92. Write a ufunc for modular exponentiation using `np.vectorize`—why slow?
93. Use `np.fmod` vs Python `%` for negative values.
94. Explain nansum vs sum; when nansum might still produce NaN.
95. Show how `np.trapz` integrates sample points—ufunc involvement.
96. Compute L2 normalisation along axis -1 with broadcasting and `np.linalg.norm`.
97. Implement soft-thresholding for wavelet shrinkage vectorised.
98. Why does `np.divide` support `casting='unsafe'` but not default?
99. Use `np.logical_and.reduce` to test all elements of boolean mask.
100. Explain `np.tensordot` vs `np.einsum` for higher-order contraction.

---

#### 101 – 120 | Custom ufuncs & Numba / Cython

101. Steps to build a C ufunc with `numpy.ufunc.fromfunc`.
102. Explain vectorised looping in Numba’s `@vectorize` decorator.
103. Compare `@njit(parallel=True)` vs `@vectorize` in Numba.
104. Use Numba to accelerate element-wise sigmoid on float32.
105. Describe gufuncs and their core-signature `(m,n),(n,p)->(m,p)`.
106. Build a generalized ufunc for pairwise cosine similarity.
107. Explain `pyfunc` fallback path effect on performance.
108. Why does Cython need `boundscheck(False)` for speed?
109. Outline memoryviews in Cython to avoid GIL.
110. Use `numexpr` to compute `a*b + c` faster than NumPy—why?
111. Describe SIMD auto-vectorisation prospects in Numba.
112. What is the role of `nopython` mode?
113. Compare `weave` (deprecated) to modern solutions.
114. Implement rolling mean via `@guvectorize` on contiguous 1-D inputs.
115. Explain ufunc signature bracketing of core dimensions.
116. Build a fused multiply-add custom ufunc.
117. Discuss pitfalls of Python callbacks inside Numba-vectorized loops.
118. How to expose a C++ template function as NumPy ufunc via pybind11.
119. Use `cython.view.array` for interop with NumPy.
120. Explain typed memoryviews vs raw buffers.

---

#### 121 – 140 | Memory Layout, Strides & Internals

121. Compute strides of a C-order (4,3) float64 array.
122. How does `arr.T` change strides but not shape?
123. Define “contiguous” and its two flavours in NumPy.
124. Why does `arr.copy(order='F')` sometimes improve BLAS perf?
125. Show an example where transposed view is 0-stride along one axis.
126. Explain `arr.strides` effect on iteration order.
127. Use `np.lib.stride_tricks.sliding_window_view`—discuss memory risk.
128. Demonstrate broadcasting leading to stride 0.
129. How does alignment padding manifest in `arr.ctypes.data` addresses?
130. Why can `np.concatenate` incur a copy while `np.stack` often cannot avoid it?
131. Discuss reference counting vs buffer protocol for memory lifetime.
132. Explain effect of `arr.flags.writeable=False`.
133. Why is `as_strided` labelled “dangerous”?
134. Use `__array_interface__` dict to build custom ndarray wrapper.
135. Describe how memory-mapped arrays (`np.memmap`) share strides with disk file.
136. How does the small-array optimisation in NumPy 2 aim to improve cache?
137. Show effect of `COPY` vs `VIEW` semantics in `reshape`.
138. Compare `.base` attribute for shallow vs deep copies.
139. Explain why column-major contiguous array’s last stride equals element size.
140. Debugging segmentation fault from invalid stride—common cause?

---

#### 141 – 160 | Performance & Profiling

141. Benchmark `np.dot` vs Python double for-loop on 1k × 1k matrix.
142. Use `%timeit` to detect temporary allocations via `np.add`.
143. Explain BLAS/LAPACK linkage influence on linalg performance.
144. Describe `np.errstate` overhead vs global seterr.
145. Show how `np.set_printoptions(threshold=…)` can mislead perception of speed.
146. Use `np.broadcast_to` to avoid `np.repeat` copying; measure memory.
147. Explain why `np.sum(axis=0, dtype=np.float32)` can be slower than float64.
148. Profile cache misses with `perf` for strided array access.
149. Use OpenBLAS thread pool env vars to pin CPU cores.
150. Discuss GIL relevance for `np.dot` multi-threading.
151. When does `np.matmul` dispatch to GPU (CuPy) through NEP-full bridging?
152. Show using `np.ufunc.reduce` vs Python loop for large integer GCDs.
153. Compare broadcasting vs `np.outer` for large vector combos.
154. Investigate false-sharing in multi-threaded Numba loops.
155. Explain memory-aligned allocation via `np.empty(1000, 'float32', 'C', 64)`.
156. Measure advantage of `dtype=float32` in matrix mult on AVX512.
157. Discuss cost of casting for mixed dtype operations.
158. Use `np.bincount` vs explicit loops for histogramming large ints.
159. Explain page-fault slowdown in out-of-core algorithms.
160. Show effect of `np.setbufsize` (removed) historical tuning.

---

#### 161 – 180 | Structured & Record Arrays

161. Define record array vs structured array.
162. Create structured array of stocks `[(symbol,U4),(price,f4),(vol,u4)]`.
163. Explain field alignment padding in nested dtypes.
164. Access subarray field in structured dtype (`('vec', float32, 3)`).
165. Convert pandas DataFrame to structured array without copy.
166. Why does sorting by multiple fields require `order=['f1','f2']`?
167. Show merging two structured arrays by common key.
168. Use `np.rec.fromrecords` with heterogenous data sources.
169. Explain view casting from structured to unstructured float array.
170. Describe limitations in fancy indexing structured arrays.
171. Save structured array to CSV with correct column names.
172. Compute mean of nested vector field.
173. Append new field to existing structured array—why hard?
174. Compare hierarchical dtype vs object arrays for ragged records.
175. Use `np.lib.recfunctions.append_fields` safely.
176. Why is `np.recarray` deprecated for new code?
177. Demonstrate join-by operation on two recarrays.
178. Explain `np.packbits` on bit-field structured array.
179. Align structured array to 8-byte boundary for C interop.
180. Discuss impact of endianness on structured file I/O.

---

#### 181 – 200 | Masked Arrays & Fancy Tricks

181. When should you use `np.ma` instead of NaNs?
182. Compute mean over masked array ignoring masked values.
183. Explain mask-propagation in arithmetic between `MaskedArray`s.
184. Fill masked values with column means in-place.
185. Use `MaskError` context to catch invalid operations.
186. Compare speed of masked array vs boolean-indexed normal array.
187. Demonstrate plotting masked data gap in Matplotlib automatically.
188. Mask values greater than dynamic percentile threshold.
189. Explain record-masked arrays and their caveats.
190. Implement lazy masking with `np.ma.MaskedView`.
191. Use `np.where` to build a tri-state mask (0,1,NaN).
192. Fancy-index a 100k×100k sparse adjacency matrix into dense submatrix—memory plan.
193. Solve Sudoku using fancy indexing boolean constraints.
194. Use `np.putmask` vs direct indexing performance.
195. Explain broadcasting of masks in four-dimensional arrays.
196. Discuss limitations of masked arrays with Numba.
197. Convert masked array to pandas with nullable dtypes.
198. Mask invalid datetime64 values and compute gaps.
199. How to compress masked array to non-masked 1-D.
200. Contrast `np.ma.masked_invalid` vs `np.isfinite`.

---

#### 201 – 220 | Random Sampling & Statistics

201. Difference between `RandomState` legacy and `default_rng`.
202. Generate reproducible random ints on multiple threads.
203. Draw 10 million standard normals; memory considerations.
204. Explain Ziggurat algorithm vs Box-Muller.
205. Use `np.random.Generator.choice` with probability weights.
206. Why is `shuffle` in place but `permutation` returns new array?
207. Implement bootstrapping confidence interval with broadcasting.
208. Compare random bit generators PCG64 vs MT19937 quality.
209. Seed context manager for deterministic test.
210. Explain `bit_generator.jumped()` for parallel streams.
211. Draw random complex numbers inside unit circle.
212. Use `Generator.multivariate_normal` to sample correlated features.
213. Evaluate chi-square goodness-of-fit using NumPy histograms.
214. Generate Poisson-distributed sparse matrix efficiently.
215. Implement rejection sampling of truncated normal.
216. Discuss correlation between successive `randn` draws—should be none.
217. Why isn’t `beta` distribution implemented for large `alpha`?
218. Stratified shuffle split using pure NumPy.
219. Compute empirical CDF and inverse transform sampling.
220. Compare `np.random.rand` vs `np.random.random_sample`.

---

#### 221 – 240 | Linear Algebra Routines

221. Explain BLAS Level 1/2/3 and which NumPy calls each.
222. Compute eigenvalues of symmetric matrix; why use `eigvalsh`?
223. Explain difference between `np.linalg.solve` and `lstsq`.
224. Show when `np.inv` is numerically unsafe.
225. Use `np.linalg.svd` to compress an image.
226. Compute Moore-Penrose pseudoinverse with cutoff tolerance.
227. Explain QR decomposition and its uses.
228. Demonstrate Cholesky factor for positive definite matrix.
229. Compare performance of `@` vs `np.dot`.
230. Batched matrix multiplication via `np.matmul` on 3-D arrays.
231. Explain condition number and its relation to invertibility.
232. Compute determinant with `slogdet` to avoid overflow.
233. Solve Toeplitz system faster using SciPy—why NumPy lacks direct routine.
234. Illustrate least squares polynomial fit with Vandermonde matrix.
235. Use `einsum` for tensor contraction equivalent to `matmul`.
236. Discuss memory ordering’s effect on BLAS GEMM.
237. Implement iterative power method for dominant eigenvector.
238. Explain Householder reflections for QR.
239. Show how to verify orthogonality of eigenvectors numerically.
240. Compute pairwise cosine similarity with `np.linalg.norm` broadcasting.

---

#### 241 – 260 | FFT, Polynomials & Signal Processing

241. Compute 1-D FFT and inverse FFT; verify reconstruction.
242. Explain real FFT (`rfft`) speed advantage.
243. Zero-padding effect on frequency resolution.
244. Use `np.fft.fftfreq` to build frequency axis.
245. Compute convolution via FFT; compare to `np.convolve`.
246. Explain windowing functions and implement Hamming window.
247. Use `np.polyfit` vs `np.linalg.lstsq` for regression.
248. Evaluate polynomial with `np.polyval`—time complexity?
249. Convert polynomial coefficients to roots and back.
250. Differentiate polynomial analytically with `np.polyder`.
251. Implement fast cross-correlation with FFT on 2-D images.
252. Compare `fftshift` vs `ifftshift`.
253. Discuss numerical errors in repeated FFT-IFFT cycles.
254. Generate power spectrum density and interpret units.
255. Use `np.polynomial.Chebyshev` for stable approximation.
256. Explain aliasing in discrete Fourier transform.
257. Compute phase correlation to estimate image translation.
258. Perform polynomial interpolation using Lagrange form.
259. Discuss use of **dtype=complex64** for GPU FFT copy reduction.
260. Implement overlap-add method for long signal filtering.

---

#### 261 – 280 | Datetime, Timedelta & Finance Utilities

261. Construct datetime64 array spanning monthly periods.
262. Explain differences between `datetime64[D]` and `datetime64[s]`.
263. Compute business-day offsets using `numpy.busday_offset`.
264. Count US federal holidays between two dates.
265. Convert POSIX timestamps to datetime64.
266. Discuss leap-second handling in NumPy.
267. Calculate daily returns from price vector using `np.diff`.
268. Compute compounded annual growth rate (CAGR) with log returns.
269. Use vectorised boolean masks for trading strategy backtest.
270. Explain `np.pmt` financial function equivalent (manual).
271. Generate irregular time series and resample with pandas vs NumPy.
272. Determine week-of-year for datetime64 array.
273. Compute rolling volatility using stride tricks.
274. Handle timezone conversion limitations in NumPy.
275. Explain why timedelta64 supports up to attoseconds.
276. Find longest stretch of consecutive business days meeting condition.
277. Perform cash-flow discounted net present value calculation.
278. Interleave multiple datetime64 arrays into one sorted array.
279. Use `busdaycalendar` for custom regional holidays.
280. Compute Sharpe ratio and discuss numerical stability.

---

#### 281 – 300 | Interoperability & Expert Patterns

281. Convert PyTorch tensor to NumPy without copy—gotchas.
282. Describe buffer protocol and how NumPy exposes it.
283. Use memory-mapped file to share array across processes.
284. Explain `np.copyto(dst, src, where=mask)` semantics.
285. Zero-copy slice from `bytes` object into uint8 array.
286. Compare CuPy vs NumPy API compatibility layers.
287. Describe DLPack and data-exchange use cases.
288. Use `np.savez_compressed` vs HDF5—trade-offs.
289. Broadcast vs tile in TensorFlow interop converting to NumPy.
290. Explain NEP-49 array-API standardisation.
291. Build blocked algorithm for large matrix multiply with memory constraint.
292. Demonstrate SIMD vectorised loops via Cython `numpy.parallel`.
293. Use `np.nditer` with external loop for C F contiguous iteration.
294. Create ragged arrays via `dtype=object`—performance caveats.
295. Explain `np.add.at` atomic updates; image histogram example.
296. Port `numexpr` expression to GPU using CuPy AST.
297. Show constructing sparse COO arrays then converting to CSR (SciPy interop).
298. Debug nan chase using `np.isfinite` and masked logging.
299. Discuss upcoming NumPy-2 improvements (like dtype metaclass).
300. Summarise three key design philosophies of NumPy that enable scientific Python stack.

---

**How to Practise:**
Pick 5–10 questions a day, implement demos in a Jupyter notebook, and cross-check with official docs or quick experiments (`%timeit`, `np.show_config()`, etc.). Happy mastering NumPy!
