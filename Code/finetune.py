import os
import argparse  
from datetime import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, T5Tokenizer, T5ForConditionalGeneration, T5PreTrainedModel, T5Model, IntervalStrategy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

def load_data(dataset_name, seed):  
    # Get the TMPDIR path
    tmpdir = os.environ['TMPDIR']

    # Determine the file path based on the dataset_name
    if dataset_name == 'Omni':
        file_path = f'{tmpdir}/OmniScience_processed.csv'
        # Load data
        df = pd.read_csv(file_path, index_col=0)
    elif dataset_name == 'Emtree':
        file_path = f'{tmpdir}/Emtree_RMC_processed.csv'
        # Load data
        df = pd.read_csv(file_path, index_col=0)
    elif dataset_name == 'Mesh':
        file_path = f'{tmpdir}/Mesh.csv'
        # Load data
        df = pd.read_csv(f'{tmpdir}/Mesh.csv', delimiter=";")
    else:
        raise ValueError(f"Invalid dataset_name: {dataset_name}. Choose from 'Omni', 'Emtree', or 'Mesh'.")

    # Process data based on the dataset_name
    if dataset_name == 'Mesh':
        df.drop(df.tail(3).index, inplace=True)  # Drop last 3 rows
        df['ambiguous'] = df['UMLS'] > 1  # Create 'ambiguous' column
        df = df[['Term', 'ambiguous', 'UMLS']]  # Keep only the desired columns
        df = df.rename(columns={"Term": "label", "ambiguous": "ambiguous"})
    elif dataset_name in ['Omni', 'Emtree']:
        df['label'] = "Category: " + df['superLabel'] + "\nTerm: " + df['label']

    # Split the data into train, val and test sets
    df_train, df_temp = train_test_split(df, test_size=0.70, random_state=seed)  
    df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=seed)
    return df_train, df_val, df_test


def load_model(model_name, device):
    if model_name == 'BERT':
        # Load the trained model
        model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)

        # Define a new model with a dense layer for dimensionality reduction
        class BertWithReducer(nn.Module):
            def __init__(self, bert_model):
                super(BertWithReducer, self).__init__()
                self.bert = bert_model
                self.reducer = nn.Linear(768, 2)

            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden_state = outputs.hidden_states[-1][:, 0, :]
                logits = self.reducer(last_hidden_state)
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                    return (loss,logits)
                else:
                    softmax_output = nn.functional.softmax(logits, dim=-1)
                    return softmax_output

        # Replace the original BERT model with the new model
        model = BertWithReducer(model)
        model = model.to(device)
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    elif model_name in ['T5_base', 'T5_large']:
        model_size = 't5-base' if model_name == 'T5_base' else 't5-large'
        class T5ForSequenceClassification(torch.nn.Module):
            '''
            Custom T5 Model setup for sequence classification;
            Logic derived from HuggingFace BART
            '''
            def __init__(self, model):
                super(T5ForSequenceClassification, self).__init__()

                # load model
                self.l1 = model

                # final classification layer
                self.classifier = torch.nn.Linear(self.l1.config.d_model, 2)

            def forward(self, input_ids, attention_mask, labels=None):
                #print(f"input_ids shape: {input_ids.shape}, attention_mask shape: {attention_mask.shape}, labels shape: {labels.shape if labels is not None else 'N/A'}")
                # generate outputs
                outputs = self.l1(input_ids=input_ids,
                                attention_mask=attention_mask,
                                decoder_input_ids=input_ids,
                                output_hidden_states=False)

                # last hidden decoder layer
                hidden_states = outputs.last_hidden_state

                # just the eos token
                eos_mask = input_ids.eq(self.l1.config.eos_token_id)
                if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
                    raise ValueError("All examples must have the same number of <eos> tokens.")

                # final hidden state of final eos token sent into classifier
                sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]
                logits = self.classifier(sentence_representation)
                #print(f"logits shape: {logits.shape}, logits: {logits}")

                # Compute loss
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1,2), labels.view(-1))
                    return {"loss": loss, "logits": logits}
                else:
                    return {"logits": logits}

        # Initialize the model
        model = T5ForSequenceClassification(T5Model.from_pretrained(model_size)).to(device)
        # Load the tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_size)
    else:
        raise ValueError(f"Invalid model_name: {model_name}. Choose from 'BERT', 'T5_base', or 'T5_large'.")
    return model, tokenizer

class AmbigDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, labels, attention_mask=None, model_name='BERT'):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_mask = attention_mask
        self.model_name = model_name

    def __getitem__(self, idx):
        item = {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx], 'labels': self.labels[idx]}
        return item

    def __len__(self):
        return len(self.labels)
    
def process_data(df, model_name, tokenizer, device, seed, undersample_flag=False, undersample_ratio=0.1):  
    if undersample_flag:   
        # Define the undersampling strategy
        undersample = RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=seed)

        # Fit and apply the transform
        X_resampled, y_resampled = undersample.fit_resample(df[['label']], df['ambiguous'])

        # Combine X_resampled and y_resampled back into a single DataFrame
        df = pd.concat([X_resampled, y_resampled], axis=1)

    texts = df['label'].tolist()
    labels = df['ambiguous'].astype(int).tolist()

    if model_name in ['T5_base', 'T5_large']:
        task_prefix = "binary classification: "
        #tokenize input
        text_encodings = tokenizer([task_prefix + sequence for sequence in texts], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    elif model_name == 'BERT':
        #tokenize input
        text_encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}. Choose from 'BERT', 'T5_base', or 'T5_large'.")

    input_ids, attention_mask = text_encodings.input_ids, text_encodings.attention_mask

    labels = torch.tensor(labels).to(device)

    #create Dataset objects
    dataset = AmbigDataset(input_ids, labels, attention_mask, model_name)

    return dataset

def compute_metrics_bert(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'roc_auc': auc,
        'f1': f1
    }

def compute_metrics_t5(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)[0],
        'recall': precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)[1],
        'roc_auc': roc_auc_score(labels, predictions),
        'f1': precision_recall_fscore_support(labels, predictions, average='binary', zero_division=1)[2]

    }


class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        outputs = model(**inputs)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

def train_model(train_dataset, val_dataset, model, model_name, dataset_name, timestamp): 
    # Define the training arguments
    if model_name in ['T5_base', 'T5_large']:
        training_args = TrainingArguments(
            output_dir = f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/tuned_{model_name.lower()}_{dataset_name.lower()}_{timestamp}',
            num_train_epochs=3,              # total number of training epochs
            per_device_train_batch_size=4,  # batch size per device during training
            per_device_eval_batch_size=16,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=250,
            dataloader_pin_memory=False,
            evaluation_strategy=IntervalStrategy.EPOCH, # evaluation is done at end of each epoch
            save_strategy=IntervalStrategy.STEPS, # model is saved every 'save_steps' steps
            save_steps=int(1e9), # set to a large number to effectively disable saving
            save_safetensors=False
        )
    elif model_name == 'BERT':
        training_args = TrainingArguments(
            output_dir = f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/tuned_{model_name.lower()}_{dataset_name.lower()}_{timestamp}',
            num_train_epochs=10,              # total number of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            per_device_eval_batch_size=64,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
            dataloader_pin_memory=False,
            load_best_model_at_end=True,     # load the best model at the end of training
            metric_for_best_model='f1',      # use f1 for best model
            greater_is_better=True,          # higher f1 is better
            evaluation_strategy='epoch',     # evaluate every epoch
            save_strategy='epoch',           # save every epoch
            eval_steps=500,                  # number of steps for evaluation
            save_steps=int(1e9)  
        )
    else:
        raise ValueError(f"Invalid model_name: {model_name}. Choose from 'BERT', 'T5_base', or 'T5_large'.")

    # Initialize the trainer
    if model_name in ['T5_base', 'T5_large']:
        trainer = CustomTrainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics_t5,  # the callback that computes metrics of interest
        )
    elif model_name == 'BERT':
        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics_bert  # the callback that computes metrics of interest
        )

    # Train the model
    trainer.train()
    return trainer

def evaluate_model(trainer, test_dataset, model_name, dataset_name, timestamp):  
    # Evaluate the model on the test dataset  
    metrics = trainer.evaluate(test_dataset)

    # Create a DataFrame from the metrics
    metrics_df = pd.DataFrame({
        'Dataset': [dataset_name],
        'Transformer Model': [model_name],
        'Accuracy': [metrics['eval_accuracy']], 
        'Precision': [metrics['eval_precision']], 
        'Recall': [metrics['eval_recall']],  
        'ROC AUC': [metrics['eval_roc_auc']],
        'F1 Score': [metrics['eval_f1']]
    })

    # # Print the DataFrame
    # print(metrics_df)

    # Save the DataFrame to a CSV file
    metrics_df.to_csv(f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/finetune_metrics_{model_name.lower()}_{dataset_name.lower()}_{timestamp}.csv', index=False)

    return metrics

def save_model(trainer, model_name, dataset_name, timestamp):

    # Save the model
    output_dir = f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/results_{model_name.lower()}_{dataset_name.lower()}_{timestamp}'
    trainer.save_model(output_dir)




#========================================================================

def main():  
    parser = argparse.ArgumentParser(description="Finetune a transformer model for text classification.")  
    parser.add_argument('--model_name', type=str, default='BERT', choices=['BERT', 'T5_base', 'T5_large'], help='Model name')  
    parser.add_argument('--dataset_name', type=str, default='Emtree', choices=['Omni', 'Emtree', 'Mesh'], help='Dataset name')  
    parser.add_argument('--undersample_flag', action='store_true', help='Flag to undersample data')  
    parser.add_argument('--undersample_ratio', type=float, default=0.1, help='Undersample ratio')  
    parser.add_argument('--smote_flag', action='store_true', help='Flag to apply SMOTE')  
    parser.add_argument('--smote_ratio', type=float, default=0.1, help='SMOTE ratio')  
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  
    args = parser.parse_args()  

    # Generate a consistent timestamp at the start of the script    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')  

    print(f"Finetuning with model_name={args.model_name}, dataset_name={args.dataset_name}, undersample_flag={args.undersample_flag}, smote_flag={args.smote_flag}, seed={args.seed}")    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print("Using " + ("cuda" if torch.cuda.is_available() else "cpu"))    

    # Load data    
    df_train, df_val, df_test = load_data(args.dataset_name, args.seed)    

    # Load model and tokenizer    
    model, tokenizer = load_model(args.model_name, device)    

    # Create Datasets for training    
    train_dataset = process_data(df_train, args.model_name, tokenizer, device, args.seed, undersample_flag=args.undersample_flag, undersample_ratio=args.undersample_ratio)    
    val_dataset = process_data(df_val, args.model_name, tokenizer, device, args.seed)    
    test_dataset = process_data(df_test, args.model_name, tokenizer, device, args.seed)    

    # Train the model    
    trainer = train_model(train_dataset, val_dataset, model, args.model_name, args.dataset_name, timestamp)    

    # Evaluate model    
    metrics = evaluate_model(trainer, test_dataset, args.model_name, args.dataset_name, timestamp)    
    print(metrics)    

    # save_model(trainer, args.model_name, args.dataset_name, timestamp)  


if __name__ == "__main__":    
    main()



# ==================================================================================================
# ==================================================================================================
# Main to run 5 times everything with different seed to calculate confidence intervals

# def main():
#     parser = argparse.ArgumentParser(description="Finetune a transformer model for text classification.")
#     parser.add_argument('--model_name', type=str, default='BERT', choices=['BERT', 'T5_base', 'T5_large'], help='Model name')
#     parser.add_argument('--dataset_name', type=str, default='Emtree', choices=['Omni', 'Emtree', 'Mesh'], help='Dataset name')
#     parser.add_argument('--undersample_flag', action='store_true', help='Flag to undersample data')
#     parser.add_argument('--undersample_ratio', type=float, default=0.1, help='Undersample ratio')
#     parser.add_argument('--smote_flag', action='store_true', help='Flag to apply SMOTE')
#     parser.add_argument('--smote_ratio', type=float, default=0.1, help='SMOTE ratio')
#     args = parser.parse_args()  

#     # Generate a consistent timestamp at the start of the script    
#     timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')  

#     print(f"Finetuning with model_name={args.model_name}, dataset_name={args.dataset_name}, undersample_flag={args.undersample_flag}, smote_flag={args.smote_flag}")    

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
#     print("Using " + ("cuda" if torch.cuda.is_available() else "cpu"))    

#     # List of seeds for different runs  
#     seeds = [42, 43, 44, 45, 46]  #,47,48,49,50,51

#     # Create an empty DataFrame to store the aggregated metrics  
#     aggregated_metrics_df = pd.DataFrame(columns=['Seed', 'Dataset', 'Transformer Model', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'F1 Score'])  

#     for seed in seeds:    
#         print(f"\nRunning with seed={seed}")    

#         # Load data with the current seed    
#         df_train, df_val, df_test = load_data(args.dataset_name, seed)    

#         # Load model and tokenizer    
#         model, tokenizer = load_model(args.model_name, device)    

#         # Create Datasets for training    
#         train_dataset = process_data(df_train, args.model_name, tokenizer, device, seed, undersample_flag=args.undersample_flag, undersample_ratio=args.undersample_ratio)    
#         val_dataset = process_data(df_val, args.model_name, tokenizer, device, seed)    
#         test_dataset = process_data(df_test, args.model_name, tokenizer, device, seed)

#         # Train the model  
#         trainer = train_model(train_dataset, val_dataset, model, args.model_name, args.dataset_name, timestamp)  

#         # Evaluate model  
#         metrics = evaluate_model(trainer, test_dataset, args.model_name, args.dataset_name, timestamp)  
#         print(metrics)  

#         # Add the current run's metrics to the aggregated DataFrame  
#         current_metrics_df = pd.DataFrame({  
#             'Seed': [seed],  
#             'Dataset': [args.dataset_name],  
#             'Transformer Model': [args.model_name],  
#             'Accuracy': [metrics['eval_accuracy']],  
#             'Precision': [metrics['eval_precision']],  
#             'Recall': [metrics['eval_recall']],  
#             'ROC AUC': [metrics['eval_roc_auc']],  
#             'F1 Score': [metrics['eval_f1']]  
#         })  
#         aggregated_metrics_df = pd.concat([aggregated_metrics_df, current_metrics_df], ignore_index=True)  

#     # Save the aggregated metrics to a CSV file  
#     aggregated_metrics_df.to_csv(f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/aggregated_finetune_metrics_{args.model_name.lower()}_{args.dataset_name.lower()}_{timestamp}.csv', index=False)  

# if __name__ == "__main__":    
#     main()