### Advanced SQL & Relational-Database Interview Question Bank

**300 questions – organised in 15 thematic blocks (20 Q × 15 = 300).**
Use the numbering for quick reference during prep or mock interviews.

---

## 1 – 20 Core SQL Semantics & Advanced Syntax

1. Explain the three-valued logic of SQL (`TRUE`, `FALSE`, `UNKNOWN`) and show a case where a predicate involving `NULL` yields `UNKNOWN`.
2. What is the difference between `DISTINCT`, `DISTINCT ON (…)` (PostgreSQL) and a grouped `SELECT` with aggregates?
3. How do ordered sets (`PERCENTILE_CONT`, `PERCENTILE_DISC`) differ from normal aggregate functions?
4. Demonstrate how lateral joins (`CROSS/OUTER APPLY`, `LATERAL`) expand correlation scopes compared with a correlated sub-query.
5. Why does `UNION` incur an implicit `DISTINCT`, while `UNION ALL` does not? Describe performance implications.
6. Show two syntactically different ways to return the top-N rows per group. What are the optimiser trade-offs?
7. What is a common table expression (CTE) materialisation barrier? How does it differ across PostgreSQL 15, SQL Server, and Oracle?
8. Describe the semantics of `MERGE` (ANSI 2003) and list two DBMS-specific implementation pitfalls.
9. Compare `INTERSECT ALL` versus `INTERSECT`. Give an example where they return different row counts.
10. How do recursive CTEs avoid infinite loops? Explain using the “WITH RECURSIVE … SEARCH DEPTH FIRST” clause (Oracle 21c).
11. Explain why `IS DISTINCT FROM` is **NULL-safe** and provide a business case where it prevents subtle bugs.
12. Give an example of an anti-join written with `NOT EXISTS` and one with a left join + `IS NULL`. Which is typically faster and why?
13. Why might you prefer a windowed `COUNT(*) OVER ()` to a second query computing total rows?
14. Demonstrate handling gaps in sequences with `GENERATE_SERIES` (Postgres) or a recursive numbers CTE in SQL Server.
15. What are inline table-valued functions (ITVF) and why can they outperform scalar UDFs in SQL Server 2019+?
16. Write a query that transforms JSON array data into rows using standard SQL 2016 JSON table functions.
17. Explain `GROUPING SETS`, `CUBE`, and `ROLLUP` in one statement; illustrate the difference in the result sets.
18. Show how to create and query a temporal table (system-versioned) in SQL Server or PostgreSQL 14.
19. Describe `MATCH_RECOGNIZE` pattern matching (Oracle/BigQuery, SQL 2016). Provide a fraud-detection example.
20. Why do correlated sub-queries sometimes get rewritten as `JOIN + GROUP BY` by the optimiser? Illustrate with a plan.

---

## 21 – 40 Normalization & Relational Theory

21. Define 1NF through 5NF and give a concrete violation example for each.
22. Why is Boyce-Codd normal form stricter than 3NF? Illustrate a schema that is 3NF but violates BCNF.
23. Explain a lossless join decomposition. Provide the formal rule involving functional dependencies.
24. Describe multivalued dependencies and the need for 4NF.
25. How do partial and transitive dependencies differ, and why do they matter for 2NF/3NF?
26. Provide a real schema that is denormalised for performance; explain the specific trade-offs.
27. What is domain/key normal form (DKNF) and why is it rarely enforced in practice?
28. Discuss surrogate vs natural keys. When might a natural key be essential (e.g., dimensional models)?
29. Explain how foreign-key cascade actions (`ON UPDATE`, `ON DELETE`) can create concurrency hot-spots.
30. What are check constraints with sub-queries (Postgres 16) and why are they risky?
31. Describe an anomaly that can occur if a database is only in 1NF.
32. Why can a fully normalised design hinder analytical workloads?
33. Explain the difference between a star schema and a snowflake schema.
34. When designing for OLTP, why might you avoid wide composite primary keys?
35. Show how deferred foreign-key constraint checking works (Postgres).
36. How does the relational model treat duplicate rows conceptually versus SQL implementations?
37. Discuss inclusion dependencies and how they differ from traditional foreign keys.
38. Illustrate a case where a unique filtered index can enforce business-rule uniqueness more cleanly than a general unique key.
39. Provide an example where denormalisation plus triggers gives both speed and correctness.
40. Explain relational algebra’s `DIVISION` operation and its SQL equivalent.

---

## 41 – 60 Indexing Internals & Physical Storage

41. Contrast clustered vs non-clustered indexes in SQL Server; why does PostgreSQL only support one heap order?
42. How do B-tree and B+-tree differ, and why are most DB-indexes B+?
43. Explain what a *covering index* is and how an *index-only* scan works in Postgres.
44. Why can descending indexes be cheaper than ascending + sort?
45. Describe *heap-only tuple* optimisation (HOT) in PostgreSQL.
46. Explain *leaf node splitting* in B-trees and its impact on write amplification.
47. What is a *reverse key* index? Why might it reduce contention in Oracle RAC?
48. Show how *partial indexes* can dramatically cut size when many rows share default values.
49. Discuss bitmap indexes vs bitmap **join** indexes; use cases in DW workloads.
50. What is an *adaptive hash index* in MySQL/InnoDB and why can it thrash?
51. Compare GiST, GIN, and BRIN indexes in Postgres—strengths and query types.
52. Why can index compression hurt point-lookup latency yet help range scans?
53. How does *write-ahead logging* interact with indexes during bulk load?
54. Explain *fillfactor* and how lowering it benefits heavy update workloads.
55. Why are `LIKE 'abc%'` searches index-friendly but `LIKE '%abc'` usually not (classic B-tree)?
56. Outline the algorithm behind *hash partitioning* and its effect on secondary indexes.
57. Describe index *hinting* and when you might override the optimiser.
58. Explain how *index skip-scan* works in Oracle or MySQL 8.0.
59. Compare *covering indexes* vs *materialised views* for query acceleration.
60. What metrics would you monitor to detect index bloat?

---

## 61 – 80 Transaction Management & Concurrency Control

61. Define ACID precisely; illustrate each property with a SQL example.
62. Compare pessimistic locking (two-phase) and optimistic concurrency control (MVCC).
63. Explain the four ANSI isolation levels and a phenomena each prevents.
64. How does snapshot isolation in PostgreSQL differ from repeatable-read in MySQL?
65. What are *write skew* and *phantom reads*? Give SQL sequences that exhibit them.
66. Why is serializable isolation often implemented with predicate locks?
67. Discuss lock escalation in SQL Server: triggers, thresholds, and side effects.
68. Contrast row-level vs page-level locks and their trade-offs.
69. Explain purpose and usage of `SELECT … FOR UPDATE SKIP LOCKED`.
70. How does PostgreSQL’s *serializable snapshot isolation* detect conflicts?
71. Describe *deadlock detection* cycles and resolution strategy in InnoDB.
72. Provide an example where setting a transaction’s `lock_timeout` prevents blocking but still guarantees correctness.
73. What is *write intent* locking in SQL Server?
74. Show how *implicit* transactions differ from *autocommit* in MySQL.
75. Explain *gap locks* and why they are necessary in RR isolation for range queries.
76. Give an example where toggling `read committed snapshot` changes result consistency.
77. Why might long-running, read-only analytics queries cause bloat in MVCC systems?
78. Outline *two-phase commit* for distributed transactions (XA) and its failure modes.
79. How do savepoints assist in complex error handling?
80. Explain *time-travel* queries in temporal databases and their MVCC interaction.

---

## 81 – 100 Query Optimisation & Execution Plans

81. Define cardinality estimation and its role in join-order selection.
82. How do cost-based optimisers use statistics histograms?
83. Give three reasons a query might choose a nested-loops join over hash join.
84. Explain pipelined vs materialised execution of sub-plans.
85. Show how parameter sniffing can result in sub-optimal plans. How do you mitigate it?
86. Why can correlated sub-queries lead to **N+1 scans**?
87. Discuss `EXPLAIN (ANALYZE, BUFFERS)` in PostgreSQL and key metrics to watch.
88. Provide a scenario where *join re-ordering* is blocked by an outer join.
89. How do *semi-joins* and *anti-joins* appear in execution plans?
90. Describe *hash aggregate* vs *sort aggregate* operators.
91. When would an optimiser choose a bitmap index AND/OR over a B-tree intersection?
92. Explain concept of *pipeline breaker* and its performance cost.
93. How does `SET enable_nestloop = off` help diagnose bad plans in Postgres?
94. Why does adding a seemingly unused filter sometimes speed up a query?
95. What is a *covering index scan fallback to heap* and why may it happen?
96. Describe adaptive query plans (e.g., Oracle 12c adaptive joins).
97. How does SQL Server’s *batch mode* on rowstore improve OLAP queries?
98. Explain the role of parallel query flags / hints and when to disable parallelism.
99. How can lateral join plans avoid recalculating correlated sub-queries?
100. Give an example of using a materialised CTE or temp table to assist the optimiser.

---

## 101 – 120 Window Functions & Analytical SQL

101. Define the components of a window clause (`PARTITION BY`, `ORDER BY`, `frame_spec`).
102. Show how to compute a running total that resets per partition.
103. Explain difference between `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` vs `RANGE` frames.
104. Provide a query that ranks rows, skipping ties vs giving same rank (`RANK` vs `DENSE_RANK`).
105. How can you use `LAG`/`LEAD` to detect gaps in sequences?
106. Derive a 7-day moving average with gaps excluded (use `ROWS` frame).
107. Explain `NTILE` and its use in bucket analysis.
108. Why might `LAST_VALUE` require `frame_rows` beyond `CURRENT ROW` to behave intuitively?
109. Show how window functions can emulate a pivot without aggregation.
110. Compare query plans for equivalent `GROUP BY` vs window aggregates.
111. Demonstrate median calculation using `PERCENTILE_CONT` window function.
112. Explain cumulative distinct count using `COUNT(DISTINCT)` in a window (which DBMS support this?).
113. Provide an example of nested windows (Snowflake) or window over window.
114. How does `ROWS BETWEEN 3 PRECEDING AND 1 FOLLOWING` differ logically from `ROWS 4 PRECEDING`?
115. Show a query producing a *sparse* vs *dense* ordered set.
116. Discuss frame exclusion (`EXCLUDE TIES`) in Postgres 14+.
117. How can `FIRST_VALUE` be used in SCD-type queries?
118. Outline limitations of window functions inside views with `ORDER BY` removed.
119. Describe incremental window aggregation for streaming systems (conceptually).
120. Explain why mixing window and group aggregates in one select list can be invalid.

---

## 121 – 140 Stored Code, Triggers & Procedural Extensions

121. Compare procedural languages: T-SQL, PL/pgSQL, PL/SQL, and MySQL 8 stored programs.
122. What are autonomous transactions and risks they introduce?
123. Explain mutating-table errors in Oracle triggers and work-arounds.
124. Outline security definer vs security invoker functions in PostgreSQL.
125. Show how to write an exception block that catches deadlock error codes.
126. Compare statement-level vs row-level triggers; performance trade-offs.
127. How do *instead-of* triggers support updatable views?
128. Discuss deterministic vs non-deterministic functions and their impact on materialised views.
129. Explain *set-returning functions* in Postgres and their placement in FROM vs SELECT.
130. Provide an example of recursive stored procedure causing stack overflow and prevention techniques.
131. What is *plan caching* for stored routines and how can parameter sniffing still occur?
132. Describe mixing procedural loops with set-based operations – performance pitfalls.
133. Give an example of a trigger that enforces complex cross-row business logic better handled in application code.
134. How do you guarantee idempotence in trigger logic to handle retried transactions?
135. Describe how *generated columns* can replace certain triggers.
136. Explain why UDFs returning tables (TVFs) can help modularise logic in Postgres.
137. Discuss dependency tracking between views/functions and schema changes.
138. What is the danger of `SET NOCOUNT ON` missing in older T-SQL procs?
139. Provide an example of using PL/pgSQL `PERFORM` to discard result of a SELECT.
140. How can stored code be version-controlled reliably in DevOps pipelines?

---

## 141 – 160 Relational Security & Governance

141. Contrast discretionary vs mandatory access controls in RDBMS.
142. Explain row-level security policies in PostgreSQL and how they differ from SQL Server’s RLS.
143. What are *fine-grained auditing* features in Oracle?
144. Discuss column-level encryption (`Always Encrypted`) in SQL Server.
145. Outline risks of SQL injection despite parameterised queries (e.g., order-by injection).
146. Give an example of a privilege escalation path via view ownership chains.
147. Explain GRANT … WITH GRANT OPTION and its governance challenges.
148. Why should application users be mapped to roles not directly to superuser?
149. Describe Transparent Data Encryption (TDE) and its performance impact.
150. How do virtual private databases mask data to specific users?
151. Explain SQL Server *dynamic data masking* vs encryption.
152. Show how to audit SELECT statements without overwhelming storage (e.g., sampling or policy-based).
153. What is *SQL firewall* in MySQL enterprise edition?
154. Discuss GDPR/right-to-be-forgotten implementation at row level.
155. Why can definer-rights functions bypass RLS, and how to mitigate?
156. Explain certificate vs password authentication for DB connections.
157. Describe security implications of using temporary tables.
158. How can you whitelist permissible commands in restricted PL/pgSQL languages (`plperlu`, `plsandbox`)?
159. Compare cross-database ownership chaining in SQL Server with Postgres’ search\_path.
160. Discuss protecting against *timing attacks* in query responses.

---

## 161 – 180 Backup, Recovery & High Availability

161. Contrast logical vs physical backups.
162. Explain PostgreSQL base backup + WAL archiving restore procedure.
163. What is *point-in-time recovery* and how is it executed in SQL Server?
164. Describe hot vs warm vs cold standby servers.
165. How do incremental backups differ between MySQL’s InnoDB vs Percona XtraBackup?
166. What is *snapshot isolation backup* in cloud-managed databases (Aurora)?
167. Explain Oracle RMAN block-level corruption detection.
168. Discuss RPO/RTO trade-offs for financial OLTP systems.
169. Provide a scenario where replication lag causes stale reads and mitigation.
170. How does synchronous replication differ from semi-synchronous in MySQL Group Replication?
171. Show failover workflow for PostgreSQL Patroni cluster.
172. What are *availability groups* in SQL Server and how do they support read scale-outs?
173. Explain `pg_rewind` and its limitations.
174. Describe *log-shipping* and why it may be chosen over streaming replication.
175. What is split-brain and how do consensus algorithms (Raft) prevent it in distributed SQL?
176. Outline backup validation strategies (checksum, restore verify).
177. Discuss backup encryption at rest and in transit.
178. How do you ensure consistent backups when foreign key constraints span multiple DBs?
179. Explain PITR combined with logical decode for audit reconstruction.
180. Describe the impact of DDL on streaming replication and how to avoid issues.

---

## 181 – 200 Distributed & Cloud-Native SQL

181. Define *sharding* and distinguish between horizontal and vertical partitioning.
182. Explain *two-phase commit* overhead in distributed shards.
183. Describe Google Spanner’s TrueTime and how it guarantees external consistency.
184. Discuss *geo-partitioning* and data-locality optimizer hints.
185. What consistency guarantees does Amazon Aurora provide across availability zones?
186. Compare CockroachDB serializable isolation vs PostgreSQL.
187. Explain *partition-pruning* and its importance in distributed query plans.
188. Provide an example query that suffers from data movement between shards.
189. How does Vitess route queries for MySQL shards?
190. Discuss secondary indexes in a sharded environment—global vs local indexes.
191. What is *read-your-writes* and why can it break under asynchronous cross-region replication?
192. Show how *logical replication* differs from physical in Postgres (useful for bi-directional sync).
193. Explain *hybrid logical clocks* (HLC) in YugabyteDB.
194. What is a *follower read* in distributed systems and its staleness bound?
195. How do cloud services enforce auto-vacuum or compaction for MVCC?
196. Discuss cost models for distributed joins in Azure SQL Hyperscale.
197. Explain FaunaDB’s Calvin protocol vs Spanner-like approaches.
198. Provide best practices for multi-tenant SaaS schema designs (separate DB, separate schema, shared schema with tenant\_id).
199. Why might you choose serverless Postgres (Neon, Aurora Serverless v2) for bursty workloads?
200. Describe network latency’s effect on commit times in global clusters and mitigation (parallel commits, paxos-optimised).

---

## 201 – 220 Data Warehousing & Columnar Extensions

201. Contrast OLTP vs OLAP workload characteristics.
202. Explain columnar storage benefits for analytical scans.
203. How do bitmap encoding and dictionary encoding save space in columnar engines?
204. Describe *zone maps* and how they assist predicate push-down.
205. Compare star schema surrogate keys vs natural keys for BI.
206. What is *slowly changing dimension* type 2 and its typical SQL implementation?
207. Show how window functions replace procedural loops in ETL for DW.
208. Discuss `CREATE TABLE … PARTITION BY RANGE` in BigQuery vs Postgres declarative partitioning.
209. Explain parallel copy/load paths (Redshift COPY, Snowflake “PUT/COPY INTO”).
210. Why is *clustered materialised view* important in BigQuery?
211. Provide steps for query performance tuning in a columnar warehouse (distribution style, sort keys).
212. How does delta encoding combine with run-length encoding?
213. Describe ACID support in Snowflake’s multicluster architecture.
214. Explain data lakehouse pattern and table formats (Iceberg, Delta, Hudi) vis-à-vis classic RDBMS.
215. Compare push-down aggregation in Presto/Trino vs MPP warehouses.
216. What is *adaptive concurrency scaling* in Redshift?
217. Discuss *data skipping indexes* (Databricks) and similarities to Postgres BRIN.
218. Provide SQL to build a surrogate key by hash in ELT pipelines.
219. Explain concept of *ELT* vs *ETL* and why modern DW prefers ELT.
220. Why is `ANALYZE`/`VACUUM` still necessary in some columnar DBMS?

---

## 221 – 240 ETL, Integration & Tooling

221. What advantages do change-data-capture (CDC) tools offer over periodic snapshot loads?
222. Describe logical decoding in PostgreSQL and how Debezium uses it.
223. Compare `COPY` vs multi-row `INSERT` in bulk loads.
224. How does SQL Server’s *BULK LOGGED* recovery model facilitate bulk loads with minimal logging?
225. Explain SCD Type 1 update vs Type 2 history in ETL SQL.
226. Discuss idempotent load patterns with staging tables and MERGE.
227. What is *upsert* and how does `INSERT … ON CONFLICT` differ from `MERGE` in other DBMS?
228. Provide a pattern for deduplication using window functions inside ETL.
229. Explain advantages of external tables for semi-structured data ingestion.
230. How can you import hierarchical JSON into relational tables using SQL only?
231. Discuss *data quality* constraints and how you’d automate checks (NULL%, range, referential).
232. Describe cross-database queries (federated) in PostgreSQL FDW.
233. What is the role of `COPY FREEZE` for immutable cold-storage partitions?
234. Outline the water-marking strategy to capture last-successful-load time in incremental ETL.
235. Compare S3 `SELECT` offload vs reading raw files into Snowflake.
236. Explain *schema evolution* and its challenges in batch ingestion.
237. Provide SQL for pivoting source rows into JSON for API output.
238. How does `INSERT … RETURNING` (Postgres) help diminish round-trips in micro-services?
239. Describe multi-table mapping in CDC to maintain referential order.
240. Explain late-arriving dimension handling in SQL ETL.

---

## 241 – 260 Vendor-Specific Intricacies

241. What is Oracle’s *index-organized table* and when is it superior to heap?
242. Explain SQL Server’s *filtered statistics* and their maintenance.
243. Discuss MySQL *doublewrite buffer* and its crash-safety role.
244. How do PostgreSQL extension hooks allow custom data types (e.g., PostGIS geometry)?
245. Compare PostgreSQL logical vs physical replication slots.
246. Describe *In-memory OLTP* (Hekaton) architecture in SQL Server.
247. Explain Oracle standby redo and Fast-Start Failover (FSFO).
248. What is MySQL’s `GROUP BY` extension allowing nonaggregated columns, and why is it dangerous?
249. Show how `table partition exchange` can implement near-zero-downtime loads (Oracle).
250. Explain MariaDB’s *columnstore* vs InnoDB; when to mix engines.
251. Discuss Postgres `parallel_hash` join algorithm introduced in v13.
252. Why do SQL Server page splits cause fragmentation, and how mitigated?
253. Provide use-case for PostgreSQL `pg_stat_statements`.
254. Compare `innodb_flush_log_at_trx_commit` settings in MySQL for durability.
255. Explain Oracle’s *result cache* and invalidation triggers.
256. What are *invisible indexes* in MySQL 8 and how to use them for tuning tests?
257. Describe Postgres `session_replication_role` and its replication-trigger bypass.
258. How does DB2’s *pureScale* ensure cluster consistency?
259. Explain SAP HANA’s delta-store merge concept.
260. Discuss `WAIT EVENT` instrumentation differences between Oracle and Postgres.

---

## 261 – 280 SQL in Modern Application Architectures

261. Why might ORMs issue inefficient N+1 query patterns and how to detect them?
262. Discuss *optimistic locking* tokens compared to DB rowversion columns.
263. Explain connection pooling and the danger of long-running idle transactions.
264. Provide a pattern for *saga* coordination in micro-services using SQL outbox.
265. Why is idempotency critical for retry logic and how to implement via unique constraints.
266. Compare UUID strategy (`uuid_generate_v4`) vs ULID or KSUID for sharded systems.
267. Describe *query pagination* pitfalls using `OFFSET` / `LIMIT` vs keyset pagination.
268. How does server-side cursor streaming work in Postgres?
269. Explain *prepared statement* attack surfaces and plan cache pollution.
270. Discuss *read replicas* for analytics and eventual consistency issues.
271. Provide SQL to implement soft deletes with `deleted_at` but keep uniqueness constraints.
272. Explain challenges of storing encrypted PII and still indexing for LIKE searches.
273. Describe JSONB indexing strategies in Postgres (`GIN` with path\_ops).
274. Why does OR mapping sometimes force anti-pattern of multi-table inheritance?
275. Provide an example using `ROW LEVEL LOCK` hints from application code.
276. Discuss pitfalls of time zone storage for multi-tenant SaaS (UTC vs local).
277. How do ORMs translate eager vs lazy loading into JOIN vs separate queries?
278. Explain application-driven sharding key discovery.
279. Why is online schema change (OSC) necessary and how does pt-osc (Percona) work?
280. Compare database-first vs code-first migration workflows and their rollback strategy.

---

## 281 – 300 Theory, Standards & Future Directions

281. Summarise the key differences between SQL-92, SQL-99, SQL-2003, and SQL-2016 standards.
282. Explain relational calculus vs relational algebra foundations behind SQL.
283. Why is NULL-intolerant predicate logic controversial among theoreticians?
284. Discuss the CAP theorem in the context of distributed SQL engines.
285. Compare graph query extensions (Cypher-in-SQL, Postgres `pg_pathman` + `pgRouting`).
286. What is *relational lattice* and how might it simplify optimisation rules?
287. Explain *materialised path* vs closure table for hierarchical queries.
288. Describe *anchor modelling* for agile data warehousing.
289. Provide pros/cons of temporal & bitemporal relational models.
290. Outline SQL extensions for machine learning inference (e.g., BigQuery ML, PostgreSQL `plpython` models).
291. What are *table functions with lateral correlation* and why do they bridge gaps with procedural dataflows?
292. Discuss ISO/IEC standard JSON in SQL (SQL-2016) and forthcoming polymorphic table types.
293. Explain how probabilistic databases challenge classic relational constraints.
294. Describe *incremental view maintenance* algorithms used in modern stream-relational systems (Materialize, RisingWave).
295. Contrast Apache Calcite’s relational algebra framework vs DB-native optimisers.
296. What is *SQL over HTTP* (RESTful) philosophy and its security concerns.
297. Analyse effects of *persistent memory (PMEM)* on redo-log and buffer-manager design.
298. Predict how **serverless relational** offerings change cost-based optimisation (burst pricing).
299. Why are *lattice-based access controls* being explored for multi-tenant clouds?
300. Debate whether SQL’s declarative paradigm will remain dominant versus API-driven data interfaces for microservices.

---
<!---
**How to Study:**
*Work through one block at a time: draft concise answers or run demo scripts in your favourite RDBMS. Focus on the why behind each concept, and back findings with execution-plan experiments wherever possible.* 
-->
