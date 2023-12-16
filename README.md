# Conditional Generation for NER

## Requirements

Python Package Requirements are:
 - openai
 - transformers
 - datasets
 - tqdm
 - pinecone

## How to Use

### Data Setup

In order to create JSONs for CONLL and WNUT, use the data.py file.
In order to use KNN, one will have to setup a Pinecone Index. To do so, use the embed.py file.

### Running Methods

Use the respective files for each method
 - baseline.py
 - knn.py
 - taskmod.py 
 - domain_context.py
 - qa.py

Refer to the notebooks for each method if something is amiss or not working as we spent the majority of our time using those.

### Results

Use metric.py or metric.ipynb

