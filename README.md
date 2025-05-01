# Codeaware RAG

[!Note]
Currently, this project is in progress and not all sections are complete.

T

This projects aims to try out different retrievers for effective retrieval for questions about code.
The project is currently in progress, therefore the 

## Next Steps

- [ ] Decide on Dataset that for evaluation
- [ ] Develop an extensible RAG system that serves as a baseline (currently under development)
- [ ] Extend the RAG system with [tree-sitter](https://tree-sitter.github.io/tree-sitter/) and/or [multispy](https://github.com/microsoft/multilspy) 


## Dataset

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

## Embedding
Dependent on the approach the usage of embeddings could be not necessary.
If the retriever is based on a deterministic approach (e.g. tree-sitter) the embeddings are not necessary.
Dependent on the retriever other database index structures like inverted indexes could be used.

To improve the embeddings we can try out two different approaches:

### Models
These embedding models that **could** be used:

- [CodeBert](https://github.com/microsoft/CodeBERT)


### Data Loaders
Both dataset does not focus on the preparation of the code snippets.
Therefore, it would be interesting to see how the performance of the retriever changes when the code snippets are preprocessed with tree-sitter or multispy.
By using the repositories from CodeXGLUE, we could create the snippets on our own, add additional metadata (e.g path) and use the queries from CodeSearchNet to evaluate the retrievers.


## Retrievers
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