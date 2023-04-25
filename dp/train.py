from transformers import (
    T5Tokenizer, 
    get_scheduler, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_constant_schedule
)
from transformers.optimization import Adafactor, AdafactorSchedule
import torch
import random
from tqdm.auto import tqdm
import numpy as np
from preprocessing import dataset_loader

from model import T5PromptTuningLM
from optimizers import get_optimizer

import os
os.environ["WANDB_DISABLED"] = "true"

class Config:
    # Same default parameters as run_clm_no_trainer.py in tranformers
    # https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py

    num_train_epochs = 100
    weight_decay = 1e-5
    learning_rate = 0.3
    lr_scheduler_type = "linear"
    max_train_steps = num_train_epochs
    max_seq_length=512
    adam_epsilon=1e-8
    warmup_steps=500
    train_batch_size=32
    eval_batch_size=32
    gradient_accumulation_steps=1
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
    n_prompt_tokens = 20
    # If True, soft prompt will be initialized from vocab 
    # Otherwise, you can set `random_range` to initialize by randomization.
    init_from_vocab = True
    # random_range = 0.5


model_checkpoint = "google/t5-v1_1-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions!= -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    count = 0
    for j in range(len(decoded_preds)):
        if decoded_preds[j] == decoded_labels[j]:
            count += 1
    return {"accuracy": count / len(decoded_preds)}


def train_trainer(args,dataset_name, model, train_dataset, eval_dataset, optimizer, lr_scheduler, noise):
    model_name = "t5-v1_1-small-" + dataset_name + "-" + str(args.n_prompt_tokens) + "-" + str(noise) 
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
            logging_steps=100,
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

    data_collator = DataCollatorForSeq2Seq(tokenizer, max_length=20)

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



if __name__ == '__main__':
    
    args = Config()
    dataset_name = "rte" #'boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc'
    train_dataset, eval_dataset = dataset_loader(dataset_name)
    eval_outputs = eval_dataset["output"]
    eval_labels = eval_dataset["labels"]

    for noise in [1e-2,1e-1,1]:
        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed) 
        torch.cuda.manual_seed_all(seed)

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
                # "weight_decay": args.weight_decay,
            }
        ]
        
        ### DP ###
        config = {"params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight" or 'lm_head' in n], "lr": args.learning_rate, "scale_parameter": False, "relative_step": False} #"weight_decay": args.weight_decay, 
        DP_config =  {"l2_norm_clip": 1, "noise_multiplier": noise,  "microbatch_size": args.train_batch_size, "num_microbatches":10}
        optimizer = get_optimizer("Adafactor", config, DP_config)
       
        ### Non-DP ###
        # optimizer = Adafactor(optimizer_grouped_parameters, lr=args.learning_rate, scale_parameter=False,  relative_step=False)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.max_train_steps * len(train_dataset) // args.train_batch_size,
        )

        train_trainer(args, model, train_dataset, eval_dataset, optimizer, lr_scheduler, "noise"+str(noise))
