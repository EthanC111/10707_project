import argparse
from dataset_preprocessing import dataset_loader
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import pdb
import json

model_checkpoint = "google/t5-v1_1-small"
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, cache_dir="../cache/transformers/")

class training():
    def __init__(self, dataset_name, tokenized_datasets):
        self.dataset_name = dataset_name
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

    def run(self):
        batch_size = 8  # set the batch size
        saved_model_name = "t5-v1_1-small-superglue-" + self.dataset_name
        model_dir = f"./cache/models/{saved_model_name}"
        log_dir = f"./cache/logs/{saved_model_name}"

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
        # pdb.set_trace()

class testing():
    def __init__(self, dataset_name, tokenized_datasets):
        self.dataset_name = dataset_name
        self.tokenized_datasets = tokenized_datasets
        self.eval_labels = self.tokenized_datasets["test"]["output"]
        pass

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        saved_model_name = "t5-v1_1-small-superglue-" + self.dataset_name
        model_dir = f"./cache/models/{saved_model_name}"
        model_dir = f"./cache/models/{saved_model_name}/checkpoint-6000"
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', choices=['boolq', 'cb', 'copa', 'multirc', 'rte', 'wic', 'wsc'], help='Name of the dataset')
    args = parser.parse_args()

    tokenized_train_dataset, tokenized_test_dataset = dataset_loader(args.dataset_name)

    train_obj = training(args.dataset_name, {'train': tokenized_train_dataset, 'test':tokenized_test_dataset})
    train_obj.run()
    # test_obj = testing(args.dataset_name, {'train': tokenized_train_dataset, 'test':tokenized_test_dataset})
    # test_obj.run()
