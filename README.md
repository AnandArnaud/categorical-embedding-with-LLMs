# Categorical Embedding with LLMs

This project aims to extract a representation vector of high-cardinality categorical data using Large Language Models (LLMs) such as BERT. We will compare our results with the baseline performance of TableVectorizer from dirty-cat.

## Datasets

The datasets used in this project can be found in the datasets folder. Additionally, we also use datasets directly from the dirty_cat module.

## Code Structure

The functions file contains all the necessary functions to run the main code (main.py). The main.py file allows running all models to compare the performance of:

* TableVectorizer
* Embeddings extracted from BERT using only the values of high-cardinality categorical variables to replace the high-cardinality categorical data and using the last four hidden states of BERT
* Embeddings extracted from BERT using both the values and column names of high-cardinality categorical variables to replace the high-cardinality categorical data, but only using the last hidden state of BERT
* Embeddings extracted from BERT using both the values and column names of high-cardinality categorical variables to replace the high-cardinality categorical data, using the concatenation of the last four hidden states of BERT

The categorical data is given in the form of sentences as inputs to BERT.

## Results

The results can be found in a CSV file located in the results folder.
