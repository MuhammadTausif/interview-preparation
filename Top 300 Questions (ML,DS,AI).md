Below is a **single master list of 300 interview questions** I curated from the most up-to-date public question banks (DataCamp, Simplilearn 2025 revisions, Medium’s LangChain list, recent Reddit threads on LangGraph/LLM interviews, and other 2025 round-ups). I filtered for the **most-asked, high-leverage, and genuinely difficult items** across the four target roles (Machine-Learning Engineer, AI Engineer, Data Scientist, Data Analyst).  ([datacamp.com][1], [datacamp.com][2], [simplilearn.com][3], [simplilearn.com][4], [skphd.medium.com][5], [reddit.com][6], [simplilearn.com][7])

### How to use this set

* Skim each category once, then rehearse aloud or in mock interviews.
* For coding or maths questions, implement a quick proof-of-concept in Python.
* Build flashcards (Anki or Quizlet) to memorise concise “headline” answers.

---

## A. Core Python & Programming (20)

1. Explain the difference between `deepcopy` and `shallow copy` in Python.
2. What happens when you modify a default list argument across function calls?
3. Why are Python generators memory-efficient, and how would you refactor a loop to use one?
4. Contrast multiprocessing vs multithreading in CPython—when is each faster?
5. Describe Python’s GIL—how does it affect TensorFlow’s data-loading pipeline?
6. Implement an LRU cache **without** `functools.lru_cache`.
7. What does the walrus (`:=`) operator return in a Boolean chain, and why?
8. Give three ways to serialize a NumPy array; discuss speed vs portability.
9. Show how `@dataclass(frozen=True)` impacts hashing and equality.
10. Why can `set.remove()` raise an error but `set.discard()` cannot?
11. Design a CLI in `argparse` that trains an ML model with sub-commands.
12. How do context managers guarantee resource release in the face of exceptions?
13. Explain duck typing; give an ML example where it breaks (e.g., scikit-learn estimator API).
14. Compare type hints `List[int]` vs `list[int]` in Python 3.9+.
15. When would you reach for `cython` or `numba` in ML pipelines?
16. Analyse this one-liner: `X[:] = X @ W + b`—what shapes make it broadcast-safe?
17. Why is `__slots__` seldom used in Pandas internals?
18. Implement a tiny TCP server that streams model predictions.
19. Profile a slow pandas groupby using `cProfile`; outline three hot-spot fixes.
20. Explain how async I/O can accelerate a LangChain agent calling multiple APIs.

## B. Data Manipulation (Pandas & NumPy) (20)

21. Vectorised vs apply/map—why is vectorisation faster under the hood?
22. Describe Pandas `CategoricalDtype`; when does it cut RAM by >70 %?
23. Show two ways to do an as-of merge on time-series data.
24. Explain `pandas.eval()`—why can it backfire on security?
25. How does `groupby().transform()` differ from `apply()`—give a leakage example.
26. Pivot vs melt—convert a sales table both ways.
27. Broadcast a 2-D NumPy mask onto a 3-D array without a loop.
28. Why is `copy-on-write` (Pandas 3.0) a game-changer for chained assignment?
29. Compute a rolling z-score with stride tricks (no explicit loop).
30. Diagnose and fix “SettingWithCopyWarning” for a chained filter-assign pattern.
31. Explain memory layout differences: C-order vs F-order in NumPy.
32. Use `np.einsum` to implement a batched dot-product attention.
33. Convert a 1-billion-row CSV to Parquet efficiently; outline chunk strategy.
34. How can Apache Arrow speed up Pandas ↔ Spark hand-offs?
35. Optimise a multi-index time-series query with `sort_index` and `slice_locs`.
36. Explain `nullable` integer dtypes (`Int64`) and their impact on joins.
37. Why does `df.loc[a:b]` include both endpoints while Python’s slice excludes the last?
38. Show a Pandas pipe chain that cleans text then vectorises with TF-IDF.
39. Describe the internal block manager change in Pandas 2.x.
40. Explain the difference between a view vs copy in NumPy; demonstrate with `np.resize`.

## C. SQL & Data Engineering (15)

41. Write a window function to compute 7-day rolling retention.
42. What’s a Bloom filter index—how does DuckDB use it?
43. ACID vs BASE in NoSQL; when would an ML feature store accept eventual consistency?
44. Optimise a slow 5-table join—index or de-normalise?
45. Explain a star schema; why suits BI dashboards?
46. Demonstrate how to backfill late-arriving data in BigQuery using `MERGE`.
47. Partitioning vs clustering—give cost impact on a 1-TB fact table.
48. Explain Z-ordering in Delta Lake.
49. Compare CDC (change-data-capture) vs batch ETL for near-real-time features.
50. Describe Kafka exactly-once semantics; why critical for online prediction services.
51. Design a snowflake schema for an e-commerce recommender system.
52. Explain materialised views; what’s their downside in a data lakehouse?
53. Show how to build an Airflow DAG that retrains an XGBoost model weekly.
54. Discuss feature store vs model registry; why keep them separate?
55. Explain lakeFS or Delta-Live-Tables as data-versioning solutions.

## D. Statistics & Probability (20)

56. Derive logistic-loss gradient for a single datapoint.
57. Why is the Central Limit Theorem key for A/B testing?
58. Explain heteroscedasticity—how does it violate OLS assumptions?
59. What is the bias–variance trade-off; visualise with MSE decomposition.
60. Compare parametric vs non-parametric tests for conversion rate uplift.
61. When is the Poisson approximation of a binomial valid?
62. Explain Bayesian credible interval vs frequentist CI.
63. Gibbs sampling vs Metropolis-Hastings—when would you choose each?
64. Show how to detect multicollinearity—VIF vs condition number.
65. What is a Dirichlet prior; how is it used in topic models?
66. Define KL-divergence and why it’s asymmetric.
67. Explain the relationship between ROC-AUC and Gini coefficient.
68. What is Welch’s t-test and why preferable for unequal variances?
69. Describe the Jeffreys prior for a Bernoulli parameter.
70. Explain propensity score matching for observational studies.
71. When does Simpson’s paradox arise in ML fairness?
72. Describe bootstrapping for model performance estimation.
73. Why does p-hacking inflate false-positive risk?
74. Explain exponential family distributions and their conjugacy.
75. Prove that the sum of independent exponential(λ) variables follows a Gamma.

## E. Machine-Learning Fundamentals (35)

76. Outline the bias–variance decomposition for k-NN.
77. Explain why decision-trees are high-variance models.
78. Derive the dual form of a linear SVM.
79. How does class weighting affect the hinge-loss?
80. Compare Gini impurity vs entropy as splitting criteria.
81. Describe bagging—why does it reduce variance but not bias?
82. Contrast AdaBoost vs Gradient Boosting—update mechanics.
83. Explain XGBoost’s regularisation term—why prevents overfitting.
84. Describe CatBoost’s ordered target encoding.
85. How does LightGBM’s histogram algorithm speed training?
86. What is the kernel trick—derive RBF kernel mapping intuition.
87. Explain label smoothing in classification.
88. Compare L1 vs L2 regularisation effects on feature selection.
89. What is elastic-net; why used in high-dimensional sparse data?
90. Describe early stopping—why is patience hyper-parameter critical?
91. Explain dropout as Bayesian approximation.
92. What are gradient clipping and why needed in RNNs?
93. Show how to tune a learning-rate schedule—cosine vs step decay.
94. Define “data leakage” with an example in time-series CV.
95. Explain SMOTE; why can it mislead on images?
96. Describe calibration curves—when accuracy is high but model poorly calibrated.
97. Explain SHAP values—contrast with permutation importance.
98. What is a partial-dependence plot; why dangerous with feature correlation?
99. Outline a feature-store; why helps offline/online training parity.
100. Explain teacher–student distillation; give a mobile-edge scenario.
101. What’s a contrastive loss; how used in self-supervised learning?
102. Describe metric-learning; why better than softmax for face recognition.
103. Explain hyperparameter optimisation via Bayesian search (e.g., Optuna).
104. Why is stratified sampling crucial for imbalanced classification?
105. Discuss class imbalance solutions other than resampling (e.g., focal loss).
106. Describe ROC curves vs PR curves—when PR is preferred.
107. Explain the curse of dimensionality—why random distance concentration hurts k-NN.
108. Provide a technique to handle high-cardinality categorical variables.
109. Explain supervised contrastive learning; benefits over vanilla contrastive.
110. Describe a recommendation system pipeline: candidate generation & ranking.

## F. Deep Learning & Neural Networks (25)

111. Explain Xavier vs He initialisation—when pick each?
112. Derive batch-norm backward pass equations.
113. Why does layer-norm aid transformers more than batch-norm?
114. Describe depthwise separable convolutions (MobileNet).
115. Explain dilated convolutions—use case in WaveNet / segmentation.
116. What is label-masking in sequence-to-sequence models?
117. Compare RNN, GRU, LSTM—gating differences.
118. Explain vanishing gradient—how residual connections mitigate it.
119. Describe self-attention’s O(N²) cost; how does Flash-Attention reduce it?
120. Contrast absolute vs rotary positional encodings.
121. Explain multi-head attention benefits.
122. Derive cross-entropy loss for multi-class softmax.
123. Discuss gradient checkpointing—trade-off memory vs compute.
124. Explain mixed-precision training with `torch.cuda.amp`.
125. Why use group-norm in small batch regimes?
126. How does an FPN improve object detection?
127. Explain cosine similarity vs dot product in embedding spaces.
128. Describe knowledge distillation for transformer compression.
129. Explain LoRA fine-tuning; how does it inject low-rank adapters?
130. Discuss retrieval-augmented generation (RAG) architecture.
131. Explain QLoRA and how it differs from classic LoRA.
132. Outline PEFT methods: adapters, prompt-tuning, IA3.
133. What is speculative decoding; why accelerates LLM inference?
134. Describe MoE (Mixture-of-Experts) routing.
135. Explain reinforcement learning from human feedback (RLHF) loop.

## G. Computer Vision (15)

136. Explain IoU; why used for NMS thresholding.
137. Describe anchor-free object detectors (e.g., FCOS).
138. Explain dice loss; why favoured in medical segmentation.
139. Compare ResNet vs EfficientNet scaling.
140. Describe Vision Transformers vs CNNs—patch embedding rationale.
141. What is deformable attention in Deformable DETR?
142. Discuss label imbalance in segmentation masks—give weighting strategy.
143. Explain focal loss in RetinaNet.
144. Describe test-time augmentation for detection performance.
145. Explain Grad-CAM; show how to apply on a ViT.
146. Discuss self-supervised vision methods: MAE vs SimCLR.
147. Explain the YOLO-NMS optimisation in YOLOv9.
148. Describe diffusion models for image generation; training objective.
149. Explain data-centric AI; give example in defective image dataset cleaning.
150. How does synthetic data augment small CV datasets?

## H. NLP & Generative AI (30)

151. Explain BPE tokenisation—why subword?
152. How does an encoder–decoder transformer differ from decoder-only?
153. Describe causal vs masked attention.
154. Explain top-p (nucleus) vs top-k sampling.
155. What is temperature in LLM decoding?
156. Compare GPT-4 vs LLaMA-3 architectural changes.
157. Explain chain-of-thought prompting; why improves reasoning.
158. Describe function-calling in OpenAI API.
159. How does retrieval augmentation reduce hallucination?
160. Outline embedding search pipeline for semantic retrieval.
161. Discuss vector store indexing: HNSW vs IVF-PQ.
162. Explain system vs user vs assistant role instructions.
163. Describe prompt injection; mitigation strategies.
164. What is jailbreak in LLM security?
165. Explain alignment tax—why aligned LLMs underperform on raw metrics.
166. Compare fine-tuning vs prompt-engineering vs RAG.
167. Describe training objective of a diffusion text-to-image model.
168. Explain classifier-free guidance in Stable Diffusion.
169. What are negative prompts?
170. Discuss multimodal transformers (e.g., GPT-4o) and their cross-modal attention.
171. How does CoT combined with self-consistency work?
172. Explain tool-use / function-calling with LangChain’s `AgentExecutor`.
173. Describe the ReAct agent prompting pattern.
174. Explain hallucination detection via logit-lens.
175. Outline evaluation metrics for text generation (BLEU, ROUGE, METEOR, BERTScore).
176. Discuss fast RAG evaluation strategies (e.g., retrieval precision, answer faithfulness).
177. Compare search-first vs generate-first RAG flows.
178. Explain knowledge distillation for small chatbots.
179. Describe vector-DB sharding in high-traffic RAG system.
180. Explain GPU vs CPU quantisation for LLM inference (INT8, Q4\_K, AWQ).

## I. LangChain & LangGraph (20)

181. What problem does LangChain solve that an LLM API alone does not?
182. Describe a `Chain` vs `Agent` in LangChain.
183. How does the LangChain Expression Language (LCEL) improve pipeline clarity?
184. Explain retriever–reader architecture in LangChain QA.
185. Describe memory classes (`ConversationBufferMemory`, etc.)—when do you need one?
186. How do you integrate a custom tool with `AgentExecutor`?
187. Explain callbacks in LangChain; give logging use-case.
188. Discuss LangSmith—how does it aid debugging?
189. Explain “chat-prompt-template” vs “prompt-template”.
190. Describe guardrails for safe completion (p-0, regex validators).
191. Outline a multi-agent workflow in LangGraph; node vs edge functions.
192. How does LangGraph handle state persistence across steps?
193. What are optimistic checkpoints in LangGraph?
194. Design an agent that uses SQL database + web search in LangGraph.
195. Discuss concurrency/parallel edges—when beneficial?
196. Explain streaming tokens through LangChain to a WebSocket client.
197. How to monitor token usage and cost across chained calls.
198. Compare LangChain’s retriever approach vs Haystack pipelines.
199. Describe a pattern to inject a classical ML model into an LLM workflow.
200. How do you secure sensitive data in a LangChain RAG pipeline?  ([skphd.medium.com][5], [reddit.com][6])

## J. MLOps & Deployment (20)

201. Explain blue-green vs canary deployments for ML APIs.
202. Compare Kubernetes HPA vs KEDA for scaling model pods.
203. Describe drift detection—feature vs concept drift.
204. Explain feature-parity issue between offline and online datasets.
205. What is shadow deployment; give monitoring metrics.
206. Describe TF-Serving dynamic batching—latency vs throughput.
207. Explain ONNX; why convert models?
208. How do you secure a model endpoint with mutual TLS?
209. Discuss GDPR impact on model training logs.
210. Explain feature store write-path vs read-path.
211. What is a model card—why important for compliance?
212. Describe CI/CD steps for retraining a model on new data.
213. Explain weights & biases sweeps; how differ from MLflow.
214. Outline steps to containerise a PyTorch model with GPU support.
215. What is serverless inference (e.g., AWS SageMaker Serverless)?
216. How does A/B testing differ for ranking models vs classification APIs?
217. Discuss rollback strategy when a model degrades in production.
218. Explain model explainability in regulated industries.
219. What is a data-contract, and how does it prevent broken pipelines?
220. Describe an immutable infrastructure approach for ML deployments.

## K. Data Analysis & BI (15)

221. Explain Cohort Analysis; build SQL to compute retention.
222. Describe KPI cascade for a subscription business.
223. How would you visualise seasonality + trend in daily active users?
224. Explain funnel drop-off analysis.
225. What is ABC classification in inventory data?
226. Outline a dashboard to track real-time anomaly detection alerts.
227. Compare Tableau vs Power BI for big-data joins.
228. Describe data profiling—list three key statistics.
229. How do you choose bin sizes for histograms (Sturges vs Freedman–Diaconis)?
230. Explain Simpson’s paradox in click-through rate analysis.
231. Design an experiment to measure impact of a UI change on engagement.
232. Explain GLM vs OLS for modelling revenue.
233. Discuss RFM segmentation; implement quickly in Pandas.
234. How do you handle missing revenue data when the user is offline?
235. Explain Pareto analysis for sales distribution.

## L. Product, Business & Communication (10)

236. A stakeholder wants “95 % accuracy”—how clarify metric & scope?
237. Explain trade-offs of shipping a v1 model fast vs waiting for 1 % more AUC.
238. Describe a time you killed a model due to bias.
239. How estimate ROI on a churn-prediction project?
240. Pitch a Gen-AI feature for a fintech app in 60 seconds.
241. What questions to ask before building an AI pipeline from scratch?
242. Explain cost drivers of LLM inference to a non-technical CFO.
243. Describe ethical concerns of face-recognition deployment.
244. How handle a product manager insisting on using the latest buzz-model?
245. Communicate model uncertainty on a dashboard for executives.

## M. Research & Advanced Theoretical (10)

246. Prove universal approximation theorem for a single hidden layer.
247. Explain lottery ticket hypothesis.
248. Describe PAC-Bayesian bounds for generalisation.
249. Contrast variational inference vs MCMC in VAEs.
250. Explain spectral clustering—derive Laplacian normalisation.
251. Describe neural tangent kernel and its link to deep networks.
252. Explain zero-shot vs few-shot emergence in large models.
253. Discuss catastrophic forgetting and continual learning solutions.
254. Explain conformal prediction intervals.
255. Derive the ELBO for a diffusion model.

## N. Ethics, Privacy & Responsible AI (10)

256. Explain differential privacy; how apply to gradient updates?
257. What is model inversion attack; defence strategies?
258. Describe C2PA provenance in Gen-AI images.
259. Explain fairness metrics—equal opportunity vs demographic parity.
260. How do you audit an LLM for toxic content?
261. Discuss red-teaming of generative models.
262. Explain federated learning vs split learning.
263. What are synthetic faces—regulatory concerns?
264. Describe Bounty programs for bias detection.
265. Explain the EU AI Act risk categories.

## O. Case Studies & Problem-Solving (35)

266. Design an end-to-end real-time fraud detection system with streaming features.
267. How would you log and replay production traffic to stress-test an LLM?
268. You have 10 GB images on CPU only—train a classifier efficiently.
269. Optimize inference latency on a mobile app using on-device quantisation.
270. Build a multi-tier RAG system that answers legal questions securely.
271. Diagnose why an uplift model performed worse than random.
272. Build a fallback hierarchy for an LLM when the context exceeds 32 k tokens.
273. Improve cold-start recommendations for a new user in a marketplace.
274. Tune a segmentation model suffering from false positives in background pixels.
275. You must anonymise PII from chat logs before fine-tuning—outline pipeline.
276. Deploy a global model with regional fine-tunes—governance plan.
277. You see accuracy-paradox on an imbalanced dataset—debug.
278. Memory leak in TensorFlow 2; reproduce & fix.
279. Optimize a Spark ML pipeline that keeps spilling to disk.
280. You need daily incremental training—list data & MLOps adjustments.
281. Explain how to AB-test two ranking models on search results.
282. Build a scenario to detect prompt-injection attacks in production.
283. Investigate concept drift in click-through rate—steps.
284. Serve 50 qps LLM calls at <300 ms P99—design infra.
285. Triage why a deployed model returns 503 errors intermittently.
286. Integrate traditional time-series ARIMA with an LSTM; ensemble approach.
287. Estimate required GPUs to train a 7-b parameter model in four weeks.
288. Build a data pipeline from IoT sensors to dashboard anomaly alerts.
289. Design a recommender that blends content-based and collaborative signals.
290. Debug a failing gradient accumulation step in distributed training.
291. Handle near-duplicates in training data to prevent memorisation.
292. Create a customer-churn survival model; explain evaluation.
293. Propose a metrics-driven approach to tune an NER system.
294. Build a monitoring dashboard for a computer vision defect detector.
295. Explain steps to migrate a monolithic ML API to microservices.
296. Devise an offline environment simulation for RL agent before production.
297. Prioritise features for a Gen-AI code-assistant MVP within two weeks.
298. Handle GPU memory overflow when fine-tuning LLaMA-3-70B.
299. Estimate business impact of moving from batch to real-time predictions.
300. Provide a step-by-step approach to introduce automated feature testing in CI.

---

**Next steps:** Work category-by-category; draft concise “model answers” or quick demos so you can answer with confidence. Good luck acing those interviews!

[1]: https://www.datacamp.com/blog/top-machine-learning-interview-questions?utm_source=chatgpt.com "Top 30 Machine Learning Interview Questions For 2025 | DataCamp"
[2]: https://www.datacamp.com/blog/genai-interview-questions?utm_source=chatgpt.com "Top 30 Generative AI Interview Questions and Answers for 2025"
[3]: https://www.simplilearn.com/tutorials/data-science-tutorial/data-science-interview-questions?utm_source=chatgpt.com "90+ Data Science Interview Questions and Answers for 2025"
[4]: https://www.simplilearn.com/tutorials/data-analytics-tutorial/data-analyst-interview-questions?utm_source=chatgpt.com "65+ Data Analyst Interview Questions and Answers for 2025"
[5]: https://skphd.medium.com/top-25-langchain-interview-questions-and-answers-d84fb23576c8?utm_source=chatgpt.com "Top 25 LangChain Interview Questions and Answers"
[6]: https://www.reddit.com/r/LangChain/comments/1k662xc/got_grilled_in_an_ml_interview_today_for_my/?utm_source=chatgpt.com "Got grilled in an ML interview today for my LangGraph-based ..."
[7]: https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-interview-questions?utm_source=chatgpt.com "Top 45 Machine Learning Interview Questions for 2025"
