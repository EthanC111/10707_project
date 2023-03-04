import transformers
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
import pdb
import pandas as pd
import json

model_checkpoint = "google/t5-v1_1-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, cache_dir="../cache/transformers/")

# Define a custom function to tokenize each element in the 'text' column
def tokenize_text(text):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512)
    return pd.Series([encoding['input_ids'], encoding['attention_mask']])

def rte_dataset_loader():
    rte_dataset = load_dataset("super_glue", 'rte', cache_dir="../cache/datasets/")
    
    labels = torch.tensor(rte_dataset['validation']['label'])
    df_train = rte_dataset['train'].to_pandas()
    df_valid = rte_dataset['validation'].to_pandas()

    label_dict = {0: 'entailment', 1: 'not_entailment'}
    df_train_list = []
    df_valid_list = []

    for i, row in df_train.iterrows():
        df_train_list.append({"input": f"hypothesis: {row['hypothesis']} premise: {row['premise']}", "output": label_dict[row['label']]})
    for i, row in df_valid.iterrows():
        df_valid_list.append({"input": f"hypothesis: {row['hypothesis']} premise: {row['premise']}", "output": label_dict[row['label']]})
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

class training():
    def __init__(self, tokenized_datasets):
        self.tokenized_datasets = tokenized_datasets
        self.eval_labels = self.tokenized_datasets["test"]["output"]
        pass

    def model_init(self):
        return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint, cache_dir="../cache/transformers/")

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        count = 0
        for j in range(len(self.eval_labels)):
            if decoded_preds[j] == self.eval_labels[j]:
                count += 1
        return {"accuracy": count / len(decoded_preds)}

    def train(self):
        batch_size = 8  # set the batch size
        model_name = "t5-v1_1-small-superglue-rte"
        model_dir = f"./cache/models/{model_name}"
        log_dir = f"./cache/logs/{model_name}"

        args = Seq2SeqTrainingArguments(
            output_dir=model_dir,
            evaluation_strategy="steps",
            eval_steps=200,
            logging_dir=log_dir,
            logging_strategy="steps",
            logging_steps=50,
            log_level="info",
            save_strategy="steps",
            save_steps=200,
            learning_rate=1e-4,#3e-4, 4e-5, 8e-5
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 4,
            # weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs= 15, #20
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            #report_to="tensorboard"
        )

        data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=20)

        trainer = Seq2SeqTrainer(
            model_init=self.model_init,
            args=args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        logs = trainer.state.log_history
        with open(log_dir + '_logs.txt', 'w') as f:
            json.dump(logs, f)
        pdb.set_trace()

class testing():
    def __init__(self, tokenized_datasets):
        self.tokenized_datasets = tokenized_datasets
        self.eval_labels = self.tokenized_datasets["test"]["output"]
        pass

    def test(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "t5-v1_1-small-superglue-rte"
        model_dir = f"./cache/models/{model_name}"
        model_dir = f"./cache/models/{model_name}/checkpoint-6000"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
        
        batch_size = 64  # set the batch size
        outputs = []    # create an empty list to store the outputs
        count = 0

        for i in range(0, len(self.tokenized_datasets["test"]), batch_size):
            batch = self.tokenized_datasets["test"][i:i+batch_size]   # get a batch of examples
            output_ids = model.generate(torch.tensor(batch["input_ids"]).to(device), attention_mask=torch.tensor(batch["attention_mask"]).to(device), early_stopping=True)  # generate the output
            output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # decode the output
            # pdb.set_trace()
            for j in range(len(batch['input'])):
                outputs.append({"input": batch['input'][j], "output_ref":  batch['output'][j], "output_text": output_text[j]})
                if batch['output'][j] == output_text[j]:
                    count += 1
            # pdb.set_trace()

        print({"accuracy": count / len(self.tokenized_datasets["test"])})
        # write the outputs to a JSON file
        with open("./cache/outputs_rte.json", "w") as f:
            json.dump(outputs, f)


if __name__ == "__main__":
    tokenized_train_dataset, tokenized_test_dataset = rte_dataset_loader()
    train_obj = training({'train': tokenized_train_dataset, 'test':tokenized_test_dataset})
    train_obj.train()
    # test_obj = testing({'train': tokenized_train_dataset, 'test':tokenized_test_dataset})
    # test_obj.test()