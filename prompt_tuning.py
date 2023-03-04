from transformers import (
    T5Tokenizer, 
    get_scheduler, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_constant_schedule
)
from transformers.optimization import Adafactor, AdafactorSchedule
import torch
import datasets
import random
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset, Features, Sequence, Value
from torch.utils.data import DataLoader

from model import T5PromptTuningLM
# from optimizers import get_optimizer



class Config:
    # Same default parameters as run_clm_no_trainer.py in tranformers
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py

    num_train_epochs = 20
    weight_decay = 1e-5
    learning_rate = 0.3
    lr_scheduler_type = "linear"
    max_train_steps = num_train_epochs
    max_seq_length=512
    adam_epsilon=1e-8
    warmup_steps=500
    train_batch_size=16
    eval_batch_size=16
    gradient_accumulation_steps=2
    n_gpu=1
    early_stop_callback=False
    fp_16=False # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1' # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0 # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42
    data_dir="" # path for data files
    output_dir="" # path to save the checkpoints
    
    # Prompt-tuning
    # number of prompt tokens
    n_prompt_tokens = 22
    # If True, soft prompt will be initialized from vocab 
    # Otherwise, you can set `random_range` to initialize by randomization.
    init_from_vocab = True
    # random_range = 0.5



def tokenize_text(text):
    encoding = tokenizer(text, padding='max_length', truncation=True, max_length=512)
    return pd.Series([encoding['input_ids'], encoding['attention_mask']])

def rte_dataset_loader():
    rte_dataset = load_dataset("super_glue", 'rte', cache_dir="datasets/")
    
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


model_checkpoint = "google/t5-v1_1-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
train_dataset, eval_dataset = rte_dataset_loader()
eval_labels = eval_dataset["output"]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions, labels[0][:20])
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    count = 0
    for j in range(len(decoded_preds)):
        if j < 10:
            print(j, decoded_preds[j], eval_labels[j])
        if decoded_preds[j] == eval_labels[j]:
            count += 1
    return {"accuracy": count / len(decoded_preds)}


def train_trainer(args, model, train_dataset, eval_dataset, optimizer, lr_scheduler):
    model_name = "t5-v1_1-small-rte-" + str(args.n_prompt_tokens)
    if args.init_from_vocab:
        model_name += "-init"

    model_dir = f"./cache/models/{model_name}"
    log_dir = f"./cache/logs/{model_name}"

    args_ = Seq2SeqTrainingArguments(
            output_dir=model_dir,
            evaluation_strategy="steps",
            eval_steps=100,
            logging_dir=log_dir,
            logging_strategy="steps",
            logging_steps=5,
            log_level="info",
            save_strategy="steps",
            save_steps=2000,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_total_limit=3,
            num_train_epochs=args.num_train_epochs,
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

    data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=512)

    trainer = Seq2SeqTrainer(
            model=model,
            args=args_,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, lr_scheduler)
        )

    trainer.train()

def train(model, optimizer, lr_scheduler, dataset, batch_size, gradient_accumulation_steps):
    dataset.shuffle()

    model.train()                    
    train_loss = []
    num_batch = 0
    optimizer.zero_grad()

    progress_bar = tqdm(range(len(dataset) // batch_size))

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size] 
        tensor_batch = {}
        for k, v in batch.items():
            if k in ['input_ids', 'attention_mask', 'labels']:
                tensor_batch[k] = torch.tensor(v).to(device)

        outputs = model(**tensor_batch)
        loss = outputs.loss
        loss.backward()

        if (num_batch + 1) % gradient_accumulation_steps == 0:
            optimizer.step()  
            optimizer.zero_grad()

        progress_bar.update(1)
        num_batch += 1

        train_loss.append(loss.item())

    lr_scheduler.step()

    return np.mean(train_loss)

def eval(model, dataset, batch_size):
    model.eval()

    count = 0
    progress_bar = tqdm(range(len(dataset) // batch_size))

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size] 
        tensor_batch = {}
        for k, v in batch.items():
            if k in ['input_ids']:
                tensor_batch[k] = torch.tensor(v).to(device)

        output_ids = model.generate(**tensor_batch)
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for j in range(len(batch['output'])):
            if i == 0:
                print(j, batch['output'][j], output_text[j])
            if batch['output'][j] == output_text[j]:
                count += 1

        progress_bar.update(1)

    return count / len(dataset)


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.cuda.manual_seed_all(seed)

    args = Config()

    # Initialize GPT2LM with soft prompt
    model = T5PromptTuningLM.from_pretrained(
        "google/t5-v1_1-small",
        n_tokens=args.n_prompt_tokens,
        initialize_from_vocab=args.init_from_vocab
    )

    if torch.cuda.is_available():
        model.cuda()

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight" or 'lm_head' in n],
            "weight_decay": args.weight_decay,
        }
    ]
    
    for n, p in model.named_parameters():
        if n == "soft_prompt.weight" or 'lm_head' in n:
            print(n)
    

    optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False,  relative_step=False)
    # optimizer = Adafactor(optimizer_grouped_parameters, scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    # lr_scheduler = AdafactorSchedule(optimizer)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps * len(train_dataset) // args.train_batch_size,
    )
    # lr_scheduler = get_constant_schedule(optimizer)

    for ep in range(args.num_train_epochs):
        train_trainer(args, model, train_dataset, eval_dataset, optimizer, lr_scheduler)
        # loss = train(model, optimizer, lr_scheduler, train_dataset, args.train_batch_size, args.gradient_accumulation_steps)
        acc = eval(model, eval_dataset, args.eval_batch_size)

        print("[Epoch " + str(ep) + "] Train Loss:", "\t Eval Acc:", acc, "\t LR:", optimizer.param_groups[0]['lr'])
