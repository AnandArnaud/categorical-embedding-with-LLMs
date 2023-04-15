import pandas as pd
import numpy as np
import torch
import transformers
import time
import datetime
import os
import gc

from tqdm import tqdm
from transformers import BertConfig, BertModel, AutoTokenizer
from sklearn.model_selection import cross_val_score

from functions.utils import get_pipeline, get_scoring

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def initialize_bert(model_name="bert-base-uncased"):
    """
    Initializes a pre-trained BERT model and tokenizer from the transformers library.

    Returns:
        tuple: A tuple of two elements. The first element is the BERT model and the second element is the tokenizer.
    """

    # Load the BERT configuration and model
    bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
    bert_model = BertModel.from_pretrained(model_name, config=bert_config)

    # Load the BERT tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)

    return bert_model, bert_tokenizer



def get_high_cardinality_categorical_columns(X, threshold=40):
    """
    Return a list of high cardinality categorical columns in the input DataFrame X.

    Args:
        X (pandas.DataFrame): The input DataFrame.
        threshold (int): The threshold for high cardinality. Default is 40.

    Returns:
        list: A list of high cardinality categorical column names.
    """

    high_cardinality_columns = [col for col in X.select_dtypes(include=["object", "category"]).columns if X[col].nunique() > threshold]
    
    return high_cardinality_columns



def create_sentence_list_from_high_cardinality_columns(X, strategy='val_and_col'):
    """
    Create a list of sentences from the high cardinality categorical columns in the input DataFrame.

    Args:
        X (pandas.DataFrame): The input DataFrame from which to extract the sentences.
        strategy (str): The strategy to use for generating the sentences. Valid options are "val" to use only
            the values of the high cardinality columns, and "val_col" (default) to use both the column names and
            values.

    Raises:
        ValueError: If the specified strategy is not valid.
        ValueError: If no high cardinality categorical columns are found in the input DataFrame.

    Returns:
        list: A list of sentences, where each sentence corresponds to a row in the input DataFrame and contains the
        names of the high cardinality columns and their corresponding values.
    """

    # Get the high cardinality categorical columns
    high_cardinality_columns = get_high_cardinality_categorical_columns(X)
    if not high_cardinality_columns:
        raise ValueError("No high cardinality categorical columns found in input DataFrame.")

    if strategy == "val":
        # Create a list of sentences for each row with only the values of the high cardinality columns
        sentences = [' '.join([f"{val}" for val in row]) for row in X[high_cardinality_columns].values]

    elif strategy == "val_col":        
        # Create a list of sentences for each row with both the column names and values of the high cardinality columns
        sentences = [' '.join([f"{col} {val}" for col, val in zip(high_cardinality_columns, row)]) for row in X[high_cardinality_columns].values]

    elif strategy == "val_col_sep":        
    # Create a list of sentences for each row with both the column names and values of the high cardinality columns
        sentences = [' '.join([f"{col} is {val}." for col, val in zip(high_cardinality_columns, row)]) for row in X[high_cardinality_columns].values]
       
    else:
        raise ValueError("Invalid strategy specified. Valid options are 'only_value' and 'value_and_column'.")
    
    return sentences



def convert_sentences_to_bert_input(sentences, tokenizer, max_length):
    """
    Convert a list of sentences to BERT-compatible input format.

    Args:
        sentences (list[str]): A list of sentences.
        tokenizer: A BERT tokenizer.
        max_length (int): The maximum length of the tokenized sentences.

    Returns:
        input_ids (torch.Tensor): The input IDs for BERT.
        attention_masks (torch.Tensor): The attention masks for BERT.
    """

    # Validate input
    if not sentences:
        raise ValueError("Sentences argument must not be empty")

    # Encode the sentences
    # Convert the sentences to BERT-compatible input format
    input_ids = []
    attention_masks = []
    for sentence in sentences:
        # Encode the sentence
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length + 2,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Add the encoded sentence to the list
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks



def get_embeddings(X, model, tokenizer, sentence_strategy):

    # Get the high cardinality categorical columns
    high_cardinality_columns = get_high_cardinality_categorical_columns(X)
    X_high_cardinality = X[high_cardinality_columns]

    # Get the low cardinality and non-categorical categorical columns
    other_variables_columns = [col for col in X.columns if col not in high_cardinality_columns]
    X_other_variables = X[other_variables_columns]
    X_other_variables = X_other_variables.reset_index(drop=True)

    # Create a list of sentences for each row
    sentences = create_sentence_list_from_high_cardinality_columns(X_high_cardinality, strategy=sentence_strategy)

    # Convert sentences to input IDs and attention masks
    max_length = max(len(s) for s in sentences)
    print(max_length)
    input_ids, attention_masks = convert_sentences_to_bert_input(sentences, tokenizer, max_length)

    if len(X) > 100000:
        batch_size = 1  # Define the batch size
        n_samples = len(input_ids)
        last = []
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size)):
                # Get a batch of inputs and pass them through the model
                batch_input_ids = input_ids[i:i+batch_size]
                batch_attention_masks = attention_masks[i:i+batch_size]
                batch_outputs = model(batch_input_ids, attention_mask=batch_attention_masks)[2][-1][:,0,:]
                last.append(batch_outputs)
                del(batch_input_ids)
                del(batch_attention_masks)
                del(batch_outputs)
                gc.collect()
        
        categorical_variables_embeddings = torch.cat(last, dim=1)       
    
    elif len(X) > 50000:
        batch_size = 2  # Define the batch size
        n_samples = len(input_ids)
        last = []
        lastt = []
        lasttt = []
        lastttt = []
        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size)):
                # Get a batch of inputs and pass them through the model
                batch_input_ids = input_ids[i:i+batch_size]
                batch_attention_masks = attention_masks[i:i+batch_size]
                batch_outputs = model(batch_input_ids, attention_mask=batch_attention_masks)[2][-1:-5:-1]
                last.append(batch_outputs[0][:,0,:])
                lastt.append(batch_outputs[1][:,0,:])
                lasttt.append(batch_outputs[2][:,0,:])
                lastttt.append(batch_outputs[3][:,0,:])
                del(batch_input_ids)
                del(batch_attention_masks)
                del(batch_outputs)
                gc.collect()

        last = torch.cat(last, dim=0)
        lastt = torch.cat(lastt, dim=0)
        lasttt = torch.cat(lasttt, dim=0)
        lastttt = torch.cat(lastttt, dim=0)
        categorical_variables_embeddings = torch.cat((last, lastt, lasttt, lastttt), dim=1)
    
    else:
        # Pass the inputs through the model to obtain the representation vectors
        with torch.no_grad():
            categorical_variables_embeddings = model(input_ids, attention_mask=attention_masks)[2][-1:-5:-1]
        categorical_variables_embeddings = torch.cat((categorical_variables_embeddings[0][:,0,:], categorical_variables_embeddings[1][:,0,:], categorical_variables_embeddings[2][:,0,:], categorical_variables_embeddings[3][:,0,:]), dim=1)

    return categorical_variables_embeddings

def extract_embeddings(sentence_strategy, embedding_strategy, dataset_name):
    dir = 'embeddings'
    file_name = dataset_name + "_" + sentence_strategy + "_embeddings.pt"
    file_path = os.path.join(dir, file_name)

    if embedding_strategy == "last":
        embeddings = torch.load(file_path)[:,:768]
    
    elif embedding_strategy == "last_four":
        embeddings = torch.load(file_path)
    
    else:
        raise ValueError("Invalid embedding strategy. Choose between 'last' and 'last_four'")
    
    n_components = len(embeddings[0])
    col_names = [f"embedding_{i+1}" for i in range(n_components)]
    df_embeddings = pd.DataFrame(data=embeddings, columns=col_names)

    return df_embeddings

def run_model(dataset, dataset_name, embeddings, sentence_strategy):
    if embeddings.shape[1] == 3072:
        embedding_strategy = "last_four"
    else:
        embedding_strategy = "last"
    
    strategy = sentence_strategy + " " + embedding_strategy
    strategy = "BERT : " + strategy


    X = dataset.X
    y = dataset.y

    target_type = y.dtype.name
    pipeline = get_pipeline(target_type)
    scoring = get_scoring(target_type)

    # Get the high cardinality categorical columns
    high_cardinality_columns = get_high_cardinality_categorical_columns(X)
    X_high_cardinality = X[high_cardinality_columns]

    # Get the low cardinality and non-categorical categorical columns
    other_variables_columns = [col for col in X.columns if col not in high_cardinality_columns]
    X_other_variables = X[other_variables_columns]
    X_other_variables = X_other_variables.reset_index(drop=True)

    X = pd.concat([X_other_variables, embeddings], axis=1, ignore_index=True)

    start = time.time()
    print("running...")
    scores = cross_val_score(pipeline, X, y, scoring=scoring, n_jobs=-1)
    print(np.mean(scores))
    end = time.time()
    time_delta = datetime.timedelta(seconds = end-start)

    result = pd.DataFrame({
        'dataset_name': [dataset_name],
        "strategy" : [strategy],
        'mean_score': [scores.mean()],
        'std_score': [scores.std()],
        "compute_time" : [time_delta]
    })

    return result