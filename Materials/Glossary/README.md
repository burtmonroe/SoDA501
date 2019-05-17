# SoDA 501 Glossary

> **Jingle-jangle fallacies** refer to the erroneous assumptions that two different things are the same because they bear the same name (jingle fallacy) or that two identical or almost identical things are different because they are labeled differently (jangle fallacy). <https://en.wikipedia.org/wiki/Jingle-jangle_fallacies>



**A/B testing**


**accuracy**


**ACI** Advanced Cyber-infrastructure. The cluster computing resources provided on Penn State's campus by ICS.


**activation function**


**active learning**


**additive smoothing**


**adjacency matrix**


**admixture model** (see "mixed membership model")


**administrative data / records**


**adversarial (learning, examples, network)**


**affine transformation**


**affinity**


**aggregation**


**AI** (see "artificial intelligence")


**AIC** Akaike Information Criterion. "AIC is an estimator of the relative quality of statistical models for a given set of data." (Wikipedia). Related and often compared to BIC. Asymptotically equivsalent to leave-one-out cross-validation in some models, including OLS.


**Airflow / Apache Airflow**


**algorithmic bias / algorithmic fairness / machine bias**


**algorithmic confounding**


**AlphaGo**


**amplified asking**


**ANN** -- Artificial neural net (see "neural net")


**anonymous function**


**Apache Software Foundation**


**API** Application programming interface.


**artificial intelligence (AI)**


**association**


**ATE** Average treatment effect.


**AUC** Area under curve. A diagnostic for the performance of a binary classifier. Can refer to area under the Receiver Operating Characteristic curve ("ROC curve") or area under Precision-Recall curve ("PR curve"). Usual advice is to use ROC AUC except in cases of class imbalance, in which case PR AUC is preferred. That is, PR curve is preferable when one case (say, the positive case) is much rarer than the other, or where you care much more about one case than the other. (This is debatable given mathematical relationships between the two.) ROC curve plots true positive rate (aka "sensitivity", aka "recall", aka "hit rate", aka "probability of detection") against false positive rate (aka "fall-out," aka "probability of false alarm," = 1- "specificity") for different thresholds. PR curve plots precision (aka "positive predictive value") against recall. Each is a summary of how the "confusion matrix" changes as a function of decision threshold. A perfect classifier will have an AUC of 1.


**autoencoder**


**auxiliary data**


**auxiliary information**


**Avro / Apache Avro**


**awk**


**AWS** Amazon Web Services.


**Azure / Microsoft Azure**


**B-tree** "... a B-tree is a self-balancing tree data structure that maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time. The B-tree is a generalization of a binary search tree in that a node can have more than two children ... It is commonly used in databases and file systems." (https://en.wikipedia.org/wiki/B-tree). See explainer on "data structures."


**backpropagation**


**bag of words**


**bagging**


**balance**


**bash**


**basis**


**batch**


**Bayes / Bayesian (statistics, estimation, theory, updating)**


**Beam**


**behavioral drift**


**behavioral science**


**Berkeley Data Analytics Stack (BDAS)**


**between-subjects design**


**betweenness centrality**


**bias**


**bias-variance tradeoff**


**BIC** Bayesian Information Criterion


**big data**


**Big O notation**


**BigQuery** see "Google BigQuery"


**Bigram** An n-gram of length two. (See "n-gram.")


**Bigtable**


**bit sampling**


**black box**


**blind source separation**


**block diagonal**


**blocking**


**blue-team dynamics** (see "red-team dynamics")


**Bonferroni correction**


**Boolean**


**boosting**


**bot**


**breadth-first search**


**Bridges** XSEDE computing resource for large datasets, run by Pittsburgh Supercomputing Center (PSC)


**calibration**


**canonical correlation analysis (CCA)**


**CAPTCHA**


**capture-recapture**


**case-folding**


**cast**


**causal discovery**


**causal effect**


**causal inference**


**causal graph**


**CCA** (see "canonical correlation analysis")


**centered**


**centrality**


**change of basis**


**characteristic matrix**


**chunk**


**citizen science**


**classification / classifier**


**Clojure**


**cloud**


**cluster analysis**


**cluster computing**


**clustered sampling**


**CNN** (see "convolutional neural net")


**coding**


**collaborative filtering**


**collective intelligence**


**column-oriented database / column store**


**Common Rule / Revised Common Rule**


**community / community detection**


**compositional / compositional data**


**compression**


**computational social science**


**confounder / confounding**


**confusion matrix**


**conjugate prior**


**concurrent validity**


**constituency parse**


**construct validity**


**content validity**


**control (group, variable)**


**convergent validty**


**convolution**


**convolutional neural net (CNN)**


**core-sets**


**correlation**


**correspondence analysis**


**cosine similarity**


**Couchbase**


**CouchDB**


**counterfactual**


**covariance / covariance matrix**


**coverage bias**


**coverage error**


**cross entropy**


**cross product**


**curse of dimensionality**


**cross-validation**


**crowdsourcing**


**csv**


**CUR decomposition**


**DAG** (see Directed acyclic graph)


**data augmentation**


**data deluge**


**data editing / statistical data editing**


**data exhaust**


**data fusion**


**data lineage**


**data mapping**


**data mining**


**data privacy**


**data profiling**


**data provenance**


**data science**


**data squashing**


**data streams**


**data structure**


**data type**


**data wrangling / cleaning / scrubbing / munging**


**database**


**database management system**


**Dataflow**


**data-intensive**


**data.table**


**DCA** Discrete Component Analysis.


**de-anonymization**


**decomposition**


**de-duplication**


**deep learning**


**degeneracy (of labels; of ERGMs)**


**degree**


**degree centrality**


**dehydrated data**


**delimited file**


**demand effect**


**dependency parse**


**design matrix**


**DFT** Discrete Fourier Transform (see Fourier)


**difference-in-differences (DID)**


**differential privacy**


**digital traces / digital footprints / digital fingerprints**


**dimensionality reduction**


**directed acyclic graph / DAG**


**dirty (data)**


**discriminant validity**


**dissimilarity**


**distance**


**distributed computing/processing**


**distributed data collection**


**distributed representation**


**distributed sensors**


**Docker**


**document-oriented database**


**document-term-matrix / document-frequency matrix**


**dot product**


**DOM**


**double centered**


**double sampling** Used in at least three distinct ways in the sampling literature. One describes a two-phase sampling procedure in which a sample is taken and then, if inconclusive, a second sample is taken. (cite).

The second, as used by Thompson, is the more relevant for SoDA. This describes a two-phase procedure in which a sample is taken to measure some variable auxiliary to our variable of interest, and then a smaller sample of those is taken to measure our (presumably more expensive or intrusive) variable of interest. The auxiliary variable is presumed to occur at a constant ratio to the variable of interest, so then we can use the auxiliary variable to improve our estimate of the variable of interest. This is very similar to the sort of procedure Salganik calls *amplified asking* in which an expensive survey is combined with cheaper "big data," except that the relationship between the two is estimated by a more general process of supervised learning rather than a ratio assumption.

A third type of double sampling is also a two-phase sampling procedure. Here we want to do stratified sampling, but don't know the size of the strata. An initial sample is used to estimate the size of a strata, and then a second stratified sample is taken to measure the variable of interest.


**drift**


**drill**


**dropout**


**DTM** (see "document-term matrix")


**dummy observations** see "pseudo-observations."


**dummy variable**


**duplicate detection**


**ecological momentary assessment (EMA)**


**econometrics**


**edit distance**


**edge**


**Eigen- (value, vector, decomposition)**


**Eigenvector centrality**


**elastic net**


**Elasticsearch**


**EM algorithm**


**embarrassingly parallel / parallelizable**


**embedding(s)**


**encouragement design**


**enriched asking**


**ensembling / ensembles**


**entity**


**entity resolution / disambiguation**


**entropy**


**environment**


**epoch**


**ERGM** (see "Exponential Random Graph Model.") 


**ETL** (see "Extract, transform, load")


**Euclidean (distance, norm)**


**event data**


**Exceed OnDemand** Software -- remote access client for X-windowing, recommended / provided by ICS for use on the ACI systems. Not to be confused with "XSEDE", the consortium of remote computing resources funded by NSF (and generally facilitiated by ICS-ACI).


**exclusion restriction**


**experiment**


**Exponential Random Graph Model (ERGM)**  A generative model of link formation in networks. Extensions include TERGM, GERGM.


**exponential family**


**external validity**


**Extract, Transform, Load (ETL)**


**F1**


**face validity**


**factor analysis (FA)**


**factorial design**


**factorization (of a matrix)**


**feature**


**feature engineering**


**feature extraction**


**feature learning**


**feature selection**


**feedforward networks**


**FFT** Fast Fourier Transform (see Fourier).


**field**


**field view (of geographic data)**


**filter**


**first-class function / first-class citizen (in programming)**


**fixed effects**


**flat file**


**floating point**


**forecasting**


**formal theory** - used in political science to mean theorizing through mathematically coherent models ... usually means use of microeconomics-style models or game theory to model some political phenomenon. Main usage is to distinguish from another subfield of political science, political theory, which in most instances bears closer resemblance to philosophy than economics. 


**Fourier (anaylsis, operator, series, transform) / Fast Fourier Transform (FFT) / Discrete Fourier Transform (DFT)**


**FP-tree**


**frame**


**frame population**


**Frobenius norm**


**functional programming**


**Fundamental Problem of Causal Inference**


**games with a purpose / gamification**


**GAN** (see "generative adversarial network")


**garden of forking paths**


**Gaussian**


**generalization error**


**generative adversarial network (GAN)**


**generative model**


**geometric mean**


**GIA** Geographic Information Analysis


**Gibbs sampling**


**Giraph / Apache Giraph**


**GIS** Geographic Information System


**git / github**


**GloVe**


**gold standard**


**Google BigQuery**


**Google Books**


**Google Correlate**


**Google Flu**


**Google Ngram Viewer**


**Google Trends**


**GPU** Graphics Processing Unit


**gradient boosting**


**gradient descent** see also "stochastic gradient descent"


**Gram matrix**


**granularity**


**graph**


**graph mining**


**graphical database**


**graphical models**


**grid computing**


**ground truth**


**grouped summaries**


**H2O**


**Hamming distance**


**Hadamard product**


**Hadoop**


**harmonic mean**


**hash / hashing / hash table**


**Haskell**


**Hawthorne effect**


**HBase**


**HCI** Human-computer interaction.


**HDFS** Hadoop distributed file system. (also, around here, "Human Development and Family Studies")


**Hessian**


**heterogeneity**


**heterogeneous treatment effects**


**heteroskedasticity**


**hidden layer / hidden nodes**


**Hidden Markov Model (HMM)**


**hierarchical / hierarchy**


**higher-order functions*


**Hilbert space**


**HITS**


**Hive**


**homogeneous-response-propensities-within-group assumption**


**homophily**


**honeytrap (for web scrapers)**


**human computing / human-in-the-loop computation**


**human subjects**


**hyperparameters**


**hypothesis space**


**ICA** (see Independent Component Analysis)


**ICS** Penn State's Institute for CyberScience. Administer the ACI (advanced cyber infrastructure) systems on campus.


**IDE** Integrated development environment.


**Impala**


**idempotent**


**identification / identification problem**


**identity matrix**


**ill-posed (problem)**


**image**


**image filter / image kernel**


**imputation**


**incidence matrix**


**Independent Component Analysis (ICA)**


**index**


**indicator variable**


**indirect effects**


**influence maximization**


**information**


**information retrieval**


**information theory**


**informed consent**


**ingestion (of data)**


**inner product**


**instance construction**


**instance detection**


**instrument**


**instrumental variable (IV)**


**integer**


**integrated / integration**


**intention to treat (ITT)**


**internal validity**


**interrupted time series**


**inverse (of a matrix)**


**inverse problem**


**inverted index**


**intervening variable**


**invertible**


**IR** Usually in this field "information retrieval." Also used for "international relations", "infrared"


**IRB** Institutional Review Board.


**IRT** (see "item response theory")


**item nonresponse**


**item response theory (IRT) / IRT model**


**Jaccard similarity**


**Javascript**


**Jetstream**


**Johnson-Lindenstrauss lemma**


**join**


**JSON**


**Julia**


**Jupyter / Jupyter notebooks**


**k-means**


**k-NN (k Nearest Neighbors)**


**k-shingle**


**Kagglification**


**Keras**


**kernel**


**kernel density estimation**


**Kernel PCA (KPCA)**


**kernel smoothing**


**kernel trick**


**key**


**key-value pair**


**key-value store**


**KL-divergence** (see "Kullback-Leibler divergence")


**Kullback-Liebler divergence** (aka "relative entropy")


**L1-norm / L1 regulariation**


**L2-norm / L2 regularization**


**lambda operator / function**


**Laplace (distribution / prior)**


**Laplacian (of a network)**


**Laplacian eigenmaps**


**LASSO**


**latency**


**Latent Dirichlet Allocation (LDA)**


**Latent Semantic Analysis (LSA)**


**Latent Semantic Indexing (LSI)**


**latent variables**


**layer**


**layer view (of geographic data)**


**LDA** either "Latent Dirichlet Allocation" or (Fisher's) "Linear Discriminant Analysis"


**Leaflet**


**leakage (of data)**


**leave-one-out cross-validation (LOOCV)**


**lemma / lemmatization**


**levels of measurement**


**lifelong learning**


**likelihood**


**linear subspace**


**linear transformation**


**link**


**linkage ("record linkage")**


**linked data**


**list**


**list experiment**


**literate programming**


**load / loadings**


**local average treatment effect (LATE)** aka "complier average causal effect (CACE)"


**locality-sensitive hashing**


**locally linear embedding (LLE)**


**logarithm**


**logistic regression / logit**


**long data**


**longitudinal**


**loss function**


**LSTM**


**Lucene (Apache Lucene)**


**Luigi**


**machine bias** (see "algorithmic bias")


**machine learning**


**makefile**


**manifest**


**manifold**


**Mahout**


**map**


**map/reduce, MapReduce**


**MariaDB**


**Markov (process, chain, model)**


**Markov Chain Monte Carlo / MCMC**


**matching** In causal inference, matching is a process for analysis of observational data, under which treated units are matched with control units that are otherwise similar by some measure on observable pretreatment variables. In information sciences, matching is another term for "record linkage."


**max norm**


**MaxEnt**


**MCMC** (see "Markov Chain Monte Carlo")


**MDS** (see "multidimensional scaling")


**measurement model**


**mechanism**


**mediator / mediating variable**


**melt**


**Memcached**


**merge**


**metadata**


**metric**


**MinHash**


**minimum spanning tree**


**missing data / missingness**


**mixed membership model** (also called an "admixture model")


**mixed models**


**mixture model**


**model-based**


**moderator / moderating variable**


**Modifiable Areal Unit Problem (MAUP)**


**moments**


**MongoDB**


**Moore-Penrose pseudoinverse**


**morpheme / morphology**


**MovieLens**


**MPI** Message Passing Interface.


**MRP / "Mister P"** Multilevel regression and poststratification.


**MTurk**


**multidimensional scaling (MDS)**


**multilevel modeling**


**multiple comparisons**


**multiple imputation**


**multiple systems estimation (MSE)**


**multiscale**


**multithreaded**


**multivariate statistics**


**MusicLab**


**mutual information (MI), pointwise mutual information (PMI), positive pointwise mutual information (PPMI)**


**MWE** Multiword entity. (See "entity.")


**MySQL**


**Naive Bayes (classifier)**


**name matching**


**named entity recognition / NER**


**natural experiment**


**natural language processing / NLP**


**nearest neighbors**


**negative sampling**


**Neo4j**


**NER** (see "named entity recognition")


**Netflix Prize**


**neural net / artifical neural net (ANN)**


**n-gram**


**NLP** (See "natural language processing")


**No Free Lunch Theorem**


**node**


**noise**


**noncompliance**


**non-metric (distance / similarity function)**


**non-negative matrix factorization (NMF)**


**nonparametric**


**non-probability sample**


**nonreactive (measure)**


**non-rectangular data**


**nonresponse bias**


**non-stationary (time series)**


**norm / normal / normalize / normalization**


**normal distribution*


**normal form**


**NoSQL**


**notebooks**


**nowcasting**


**NP-hard / NP-complete**


**numerical computation**


**NumPy stack**


**object**


**object-oriented programming**


**object view (of geographic data)**


**observational (data, design)**


**OCR** Optical character recognition. The task of turning images into alphabetic (or similar) characters.


**OLS** Ordinary least squares


**one-hot**


**online** Could be used in regular meaning like "on the web" but can also be used like "streaming" as a modifier to "data" or "algorithm" or "processing" to indicate data is being processed sequentially, with one pass through the data, or even in real-time as data is generated.


**open data**


**open science**


**OpenStreetMap**


**operationalize / operationalization**


**ORC** Optimized Row Columnar


**orthogonal**


**orthogonalize**


**orthonormal**


**OSEMN workflow**


**out-of-sample prediction**


**over-determined**


**over-fitting**


**p-hacking**


**Pachyderm**


**PageRank**


**pandas**


**panel data**


**parallel worlds design**


**parametric**


**Parquet**


**parse tree**


**parsing**


**Pasteur's quadrant**


**path**


**pattern**


**pattern recognition**


**pdf** (stands for "portable desription format.") "Where data goes to die." -- Simon Rogers (Data editor, Google News Lab).


**Penrose inverse**


**persistence**


**perturb-and-observe experiment**


**pickle (pkl)**


**pipe / pipe operator**


**pipelines**


**pivot table**


**placebo / placebo test**


**PMI** Pointwise mutual information. (See "mutual information")


**population drift**


**POS (in NLP)** Part of speech. A "POS tagger" attempts to annotate the tokens/words of input sentences / text with their part of speech.


**positive definite**


**posterior**


**PostGIS**


**PostgreSQL**


**posting**


**post-SQL**


**post-stratification**


**potential outcomes model**


**PPMI** Positive pointwise mutual information. (See "mutual information")


**precision**


**prediction**


**preprocessing**


**pre-registration**


**principal angle**


**Principal Component Analysis (PCA)**


**prior**


**privacy-preserving data-mining**


**probabilistic data structures**


**Procrustes (analysis)**


**profiling (of code)**


**profiling (of data)** "Data profiling is the process of examining the data available from an existing information source ... and collecting statistics or informative summaries about that data." (Wikipedia) This is an information science concept, where it is also known as "data archeology", "data assessment". I know of no literature overlap, but it is essentially the same as the precursors to the process of "data editing" used by official statistics agencies. Individual records profiling can consist of syntactic profiling (making sure entries fit broad format constraints) or semantic profiling (making sure entries make sense). Set-based profiling involves examining the distribution of values of a variable/field, and parallels the social science practice of providing "descriptive statistics." See Rattenbury, et al. (2017) Principles of Data Wrangling. 


**projection**


**provenance (of data)**


**pruning**


**pseudo-inverse**


**pseudo-observations / pseudo-counts**


**quasi-experiment**


**QGIS**


**query**


**R**


**random forests**


**random projections**


**randomized controlled experiment / trial**


**raster**


**RCT** randomized controlled trial.


**RDD** in context of causal inference with observational data, see "regression discontinuity design." In context of computing with Spark, see "resilient distributed dataset."


**RDF (Resource description format)**


**RDS (R Data Serialization format)**


**reactivity**


**recall**


**recast**


**recommender system**


**reconstruction error**


**record**


**record linkage** - (see "linkage")


**recurrent neural net (RNN)**


**red-team dynamics / blue-team dynamics**


**Redis**


**redundancy**


**regression**


**regression discontinuity design (RDD)**


**regression to the mean**


**regression trees**


**regex** (see "regular expression")


**regular expressions**


**regularization / regularize**


**reinforcement learning**


**relation / relational data / relational database**


**reliability**


**ReLU (Rectified Linear Unit)**


**remote sensing**


**replace, refine, reduce**


**replicability**


**report**


**representation (of data)**


**representative / representativeness**


**reproducibility**


**repurposing (of data)**


**resilient distributed dataset (RDD)**


**respondent-driven sampling**


**RESTful / REST API**


**ridge regression**


**RMSE** Root mean squared error.


**RNN** (see "recurrent neural net")


**robots.txt**


**ROC curve**


**rotation**


**RStudio**


**SAC** (see "Split-Apply-Combine")


**sample population**


**sampling**


**sampling error**


**sampling frame**


**Scala**


**scale**


**scaling**


**scatter matrix**


**schema**


**scikit-learn**


**script**


**SciPy**


**segmentation**


**selection bias**


**semantic web**


**semantics**


**semi-parametric**


**semi-structured data**


**semi-supervised learning**


**sensitive / sensitivity**


**sensors**


**SEO** Search engine optimization.


**sequence modeling**


**serialization (of data) / serialization formats**


**SGD** (see "stochastic gradient descent")


**shapefile**


**shell**


**shingle / shingling**


**shrinkage**


**sigmoid (activation function)**


**signal processing**


**simplex**


**singular**


**singular value decomposition (SVD)**


**sketches**


**smoothing**


**SNA** Social network analysis


**snakemake**


**social**


**social data**


**social data analytics**


**social data stack**


**social network** I'd prefer if this were only used to refer to networks in which the nodes are people or groups of people and the edges indicate some kind of "social" relationship. It is sometimes used in reference to networks more generally. It is also, of course, a term of art used to refer to social media sites or platforms, e.g., Facebook. 


**social science**


**softmax**


**Software Carpentry**


**Solr (Apache)**


**Spark**


**sparse coding**


**sparse matrix**


**sparsity**


**spatial**


**spatial autocorrelation**


**spectrum / spectral theory**


**spider trap**


**spillover**


**split-apply-combine (SAC)**


**spurious**


**SQL**


**SQLite**


**stationarity**


**statistical conclusion validity**


**statistical disclosure limitations**


**statistical learning**


**stemming / stem**


**STM** Structural Topic Model


**stochastic gradient descent / SGD**


**strata**


**stratified sampling**


**streaming / stream processing** Could be used in regular meaning like "streaming video" but can also mean data is being passed through an algorithm sequentially, with the algorithm updating after each observation.


**structural**


**structured data**


**supervised learning**


**support vector machine (SVM)**


**SUTVA**


**SVD** (see "singular value decomposition")


**SVM** (see "support vector machine")


**systematic sample**


**systemic drift**


**tensor**


**TensorFlow**


**tesselation**


**test data / test set** (see "training data")


**tf.idf / tf-idf**


**threats to validity**


**tidy data / tidyverse**


**Tikhonov regularization**


**tile**


**topic model**


**toponymy** The study of place names.


**total survey error**


**trace**


**training data/set; test data/set; validation data/set**


**transfer learning**


**transition matrix**


**transparency**


**transportability**


**treatment / treatment effect**


**triangle inequality**


**Trifacta Wrangler**


**TSCS** Time series - cross-section


**t-SNE**


**tsv**


**Tucker decomposition**


**Turkers**


**uncertainty**


**unfolding** Lots of disciplines have something they call "unfolding" (e.g., music, biochemistry). Here, it's most likely to refer to a type of data analysis closely related to multidimensional scaling, which maps individuals and objects over which they have preferences in the same space. The next most likely usage is as "deconvolution", the reversing of a convolution operation.


**Unicode**


**unit nonresponse**


**unobtrusive (measure)**


**unpivot**


**unsupervised learning**


**uptake rate**


**user-attribute inference**


**UTF-8**


**"V"s of big data** Traditional "three Vs of big data" - volume, velocity, variety. There are many "fourth V's" including Monroe's "five Vs of big data social science"


**validation / validity**


**variance**


**variational inference / variational Bayesian methods**


**variety** One of the conventional "three Vs" of "Big Data"


**varimax / VARIMAX**


**vector**


**vector quantization (VQ)**


**vector space / vector space model**


**VEM** Variational Expectation Maximization


**version control**


**vertex**


**vinculation** - the tendency for social data to display interconnectedness (e.g., tied through network edges, exhibiting spatial correlation) that complicates inference and/or is itself the target of inference. Vinculated data may be small in N, but still require computationally intensive methods. One of Monroe's (2013) "five Vs." A [vinculum](https://www.dictionary.com/browse/vincula) is a "bond" or "tie"; used in anatomy, chemistry, and math.


**virtual machine**


**virtualization**


**visual analytics**


**Voronoi diagram / tesselation** aka "proximity polygons"


**wavelet**


**weak instrument**


**web driver**


**weights**


**WEIRD** Western, Educated, Industrialized, Rich, and Democratic. A critique of the typical pool of participants for lab experiments.


**wide data**


**within-subjects design**


**word2vec** Also doc2vec, sense2vec, skip-gram, CBOW, negative sampling


**XML**


**XPath**


**XSEDE** "Extreme Science and Engineering Discovery Environment"


**YAML**

**zone**
