from datasets import load_dataset, Dataset
from transformers import T5Tokenizer
import pdb
import pandas as pd

model_checkpoint = "google/t5-v1_1-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, cache_dir="../cache/transformers/")
label_dict = {
                # 'axb': {0: 'entailment', 1: 'not_entailment'},
                # 'axg': {0: 'entailment', 1: 'not_entailment'},
                'boolq': {0: 'false', 1: 'true'},
                'cb': {0: 'entailment', 1: 'contradiction', 2: 'neutral'}, 
                'copa': {0: 'choice1', 1: 'choice2'},
                'multirc': {0: 'False', 1: 'True'},
                'rte': {0: 'entailment', 1: 'not_entailment'}, 
                'wic': {0: 'False', 1: 'True'},
            }

def io_formatting(row, dataset_name):
    '''
    if dataset_name == 'axb':
        return {"input": f"sentence1: {row['sentence1']} sentence2: {row['sentence2']}", "output": label_dict[dataset_name][row['label']]}
    elif dataset_name == 'axg':
        return {"input": f"hypothesis: {row['hypothesis']} premise: {row['premise']}", "output": label_dict[dataset_name][row['label']]}
    '''
    if dataset_name == 'boolq':
        return {"input": f"question: {row['question']} passage: {row['passage']}", "output": label_dict[dataset_name][row['label']]}
    elif dataset_name == 'cb':
        return {"input": f"hypothesis: {row['hypothesis']} premise: {row['premise']}", "output": label_dict[dataset_name][row['label']]}
    elif dataset_name == 'copa':
        return {"input": f"choice1: {row['choice1']} choice2: {row['choice2']} premise: {row['premise']} question: {row['question']}", "output": label_dict[dataset_name][row['label']]}
    elif dataset_name == 'multirc':
        return {"input": f"question: {row['question']} answer: {row['answer']} paragraph: {row['paragraph']}", "output": label_dict[dataset_name][row['label']]}
    elif dataset_name == 'rte':
        return {"input": f"hypothesis: {row['hypothesis']} premise: {row['premise']}", "output": label_dict[dataset_name][row['label']]}
    elif dataset_name == 'wic':
        return {"input": f"word: {row['word']} sentence1: {row['sentence1']} sentence2: {row['sentence2']}", "output": label_dict[dataset_name][row['label']]}
    elif dataset_name == 'wsc':
        span2_text = '*' + row['span2_text'] + '*'
        text_split = row['text'].split(" ")
        text_split[row['span2_index']] = span2_text
        text_processed = " ".join(text_split)
        return {"input": f"text: {text_processed}", "output": row['span1_text']}

# Define a custom function to tokenize each element in the 'text' column
def tokenize_text(text):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512)
    return pd.Series([encoding['input_ids'], encoding['attention_mask']])

def dataset_loader(dataset_name):
    raw_dataset = load_dataset("super_glue", dataset_name, cache_dir="../cache/datasets/")
    df_train = raw_dataset['train'].to_pandas()
    df_valid = raw_dataset['validation'].to_pandas()

    df_train_list = []
    df_valid_list = []

    for i, row in df_train.iterrows():
        df_train_list.append(io_formatting(row, dataset_name))
    for i, row in df_valid.iterrows():
        df_valid_list.append(io_formatting(row, dataset_name))
    df_train = pd.DataFrame(df_train_list)
    df_valid = pd.DataFrame(df_valid_list)

    # encode the inputs and outputs
    df_train[['input_ids', 'attention_mask']] = df_train['input'].apply(tokenize_text)
    df_train[['labels', 'label_attention_mask']] = df_train['output'].apply(tokenize_text)
    df_valid[['input_ids', 'attention_mask']] = df_valid['input'].apply(tokenize_text)
    df_valid[['labels', 'label_attention_mask']] = df_valid['output'].apply(tokenize_text)
    
    df_train = df_train.drop('label_attention_mask', axis=1)
    df_valid = df_valid.drop('label_attention_mask', axis=1)

    dataset_train = Dataset.from_pandas(df_train)
    dataset_valid = Dataset.from_pandas(df_valid)

    return dataset_train, dataset_valid