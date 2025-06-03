# Codeaware RAG

> [!Note]
> Currently, this project is in progress and not all sections are complete.

This projects aims to try out different retrievers for effective retrieval for questions about code.
The project is currently in progress, therefore the 

## Next Steps

- [x] Decide on Dataset that for evaluation -> Create a custom dataset based on Flask (see [validation.csv](data/validation.csv))
- [x] Develop an extensible RAG system that serves as a baseline -> Baseline using huggingface sentence-transformers is implemented
- [x] Extend baseline to support autotokenizer and other models (unixcoder)
- [x] Validate questions
- [x] Generate other metrics (Line, Method, Class based metrics)
- [ ] Add validation run with dataset from huggingface to validate the metrics and my implementation ?
- [ ] Compare different embedding models (e.g. CodeBert, UnixCoder, ...)
- [ ] Extend the RAG system with [tree-sitter](https://tree-sitter.github.io/tree-sitter/) and/or [multispy](https://github.com/microsoft/multilspy) 
- [ ] Use a LLM to generate the dataset for a given repository
- [ ] Category for questions


## Dataset

### Evaluated Datasets
Datasets that **could** be used for the evaluation:

- [Codesearchnet](https://huggingface.co/datasets/sentence-transformers/codesearchnet) 
  - Originally published as a challenge [CodeSearchNet Challenge](https://arxiv.org/pdf/1909.09436v3)
  - Pros
    - Mapping of a [NL-Query](https://github.com/github/CodeSearchNet/blob/master/resources/queries.csv) to code
    - Code annotations by experts
    - Can be used to compare retrievers (Embeddings vs. deterministic retrievers)
    - Multiple languages (detailed picture)
    - Very Large Dataset (>1.3M Examples) to search in
  - Cons
    - Task are not suitable for end-to-end testing as generating the code would likely be easy for current LLMs
    - Queries are short and do not represent typical LLM inputs
    - Multiple languages (harder to implement or filter is necessary)
    - No queries that require multiple code snippets to be solved 

- [CodeXGLUE NL-code-search-Adv](https://huggingface.co/datasets/google/code_x_glue_tc_nl_code_search_adv)
  - Mapping of docstring to function. The docstring serves as the query the function and code as the answer.
  - Pros
    - Full repository search (TODO Check!)
    - Large Dataset (>280k Examples)
  - Cons
    - No queries that require multiple code snippets to be solved 
    - Currently only python coding questions

### Custom Dataset
For our evaluation we decided to use a custom dataset.
This is because no dataset is available that fits our needs.
We want to evaluate the retrievers on questions require a deep understanding of the code and the ability to retrieve multiple code snippets.
At least some of the questions should require a full repository search to find the correct answer.
This means that the queries should be longer and more complex than the queries in the CodeSearchNet or CodeXGlue datasets.
Additionally, we want to evaluate the retrievers on questions that require a full repository search and provide them with the in-line documentation of the code.

We decided to create a custom dataset based on the [Flask repository](https://github.com/pallets/flask).
This dataset contains hand-crafted questions.
The first 10 questions are inspired by common questions asked in StackOverflow about Flask.
The other questions are questions that require a more deep understanding of the code.
The dataset is stored in the [validation.csv](data/validation.csv) file.

#### Shortcomings
Our dataset has multiple shortcomings:
- The dataset is small (only 18 questions)
- There are multiple code snippets that have similar/identical information (e.g. abstract functions, decorators, ...). 
  This means even during the creation of the dataset it was hard to find the correct reference answer.
  We tried to mitigate this problem by manually comparing the validation answers and the code snippets retrieved by the UniXcoder-retriever.
- The dataset covers only one repository (Flask)


## Embedding
Dependent on the approach the usage of embeddings could be not necessary.
If the retriever is based on a deterministic approach (e.g. tree-sitter) the embeddings are not necessary.
Dependent on the retriever other database index structures like inverted indexes could be used.

To improve the embeddings we can try out two different approaches:

### Models
The following models are currently supported as embeddings and can be used as a baseline:

- [UniXcoder](https://huggingface.co/microsoft/unixcoder-base)
- [SFR-Embedding](https://huggingface.co/Salesforce/SFR-Embedding-Code-400M_R) Model is available in multiple sizes.
- [CodeT5-Plus](https://huggingface.co/Salesforce/codet5p-110m-embedding) Model is available in multiple sizes.
- [codesage-large-v2](https://huggingface.co/CodeSage/codesage-large-v2)
- [codesage-large](https://huggingface.co/CodeSage/codesage-large)
- [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed)
- [inf-retriever-v1-1.5b](https://huggingface.co/infly/inf-retriever-v1-1.5b)
- [multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
- [bilingual-embedding-large](https://huggingface.co/Lajavaness/bilingual-embedding-large)

### Data Loaders
Both dataset does not focus on the preparation of the code snippets.
Therefore, it would be interesting to see how the performance of the retriever changes when the code snippets are preprocessed with tree-sitter or multispy.
By using the repositories from CodeXGLUE, we could create the snippets on our own, add additional metadata (e.g path) and use the queries from CodeSearchNet to evaluate the retrievers.


## Retrievers

### Baseline Retrievers
For the baseline we implemented a retriever that uses the huggingface ecosystem.
We currently support the [sentence-transformers](https://www.sbert.net/) and [Transformers](https://huggingface.co/docs/transformers/en/index).
The advantage of sentence-transformers is that it is optimized for sentence embeddings and provides a simple interface to use.
It may be possible to only use the transformers library, but this is currently not our focus.


### Custom Retrievers
We have two types of retrievers we can implement.
Embedding based retrievers and deterministic retrievers.
Our focus is on the deterministic retrievers, but we can use the embedding based retrievers as a baseline.

Possible approaches for the deterministic retrievers are:
- [tree-sitter](https://tree-sitter.github.io/tree-sitter/)
  - Pros
    - TODO
  - Cons
    - TODO
- [multispy](https://github.com/microsoft/multilspy) 
  - Pros
    - TODO
  - Cons
    - TODO

- TODO how does the python splitter work (whole class vs function)

- how many results is needed to get the correct answer
- pass in k (is the correct answer in the top k results) to the retriever
- Line vs function vs class based metrics

- create custom dataset for different categories (dependencies, ...)