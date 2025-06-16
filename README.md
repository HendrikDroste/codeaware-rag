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
- [x] Compare different embedding models
- [ ] Create a retriever base class
- [ ] Custom splitter for code snippets
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


### Custom Retriever Types

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

### Retriever Base idea
A first idea for a retriever is to create embeddings for all functions and methods in the codebase.
Based on the distance between the query and the embeddings, we can retrieve the most relevant functions and methods.
As metadata,, we use the path to the file and the start and end line of the function or method.
This allows us to create snippets that contains an abstraction of one functionality with the relevant documentation.
For the idea it is crucial to compare this with the RecursiveCharacterTextSplitter from LangChain.
The python text splitter is based on simple regular expressions matching.
In the class there is a defined [list of patterns](https://github.com/langchain-ai/langchain/blob/b149cce5f8a68e0122ef590c98b3ec8e229586cc/libs/text-splitters/langchain_text_splitters/character.py#L345) that are used to split the code into smaller chunks.
On the highest level, it splits the code into classes.
If the desired chunk size is not reached, it splits the code into methods and functions.
This loop continues with other expressions until the desired chunk size is reached.
As this logic is relatively simple, we expect that with a more complex splitter like tree-sitter or multispy we can improve the quality of the snippets.

### Retriever Extension 1
This extension uses a graph database to store the location of the function/method definitions in the codebase.
We use the graph database to store edges from function/method definitions and their usages.
This allows us to understand the usage of the function/method in the codebase.

A high level query could look like this:
1. We use the list with all function/method definitions to find the k most relevant functions/methods based on the query.
2. We use the graph database to find all usages of the k most relevant functions/methods.
3. Rerank the k most relevant functions/methods based on the definitions and the usages.

### Retriever Extension 2
This extension uses an agent approach to find the relevant code snippets.
The agent has functions to query the graph database.

### Retriever Extension 3
If we have access to the git history we can use the git history to better understand the codebase.
As a git commit is often on a more abstract level we can use the git history to answer more complex questions.
Furthermore, we can use the diff files to identify related code snippets who are not directly connected via the graph database.

### Retriever Extension 4
Add debugging information to the code snippets.
This can include the stack trace, the variables and their values, and the function/method calls.
This could improve the understanding for the usage of objects.


- TF/IDF