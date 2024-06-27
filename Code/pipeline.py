import os  
import argparse  
from tqdm import tqdm  
import datetime  
import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.preprocessing import StandardScaler  
from sklearn.ensemble import RandomForestClassifier 
from sklearn.dummy import DummyClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score  
import xgboost as xgb  
from sklearn.naive_bayes import GaussianNB  
from transformers import AutoTokenizer, BertForSequenceClassification  
from transformers import T5Tokenizer, T5EncoderModel  
import torch  

from imblearn.over_sampling import SMOTE  
from imblearn.under_sampling import RandomUnderSampler  
from sklearn.svm import SVC  


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
        print("\n")  
        df['label'] = "Category: " + df['superLabel'] + "\nTerm: " + df['label']  

    # Split the data into train and test sets  
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=seed)  

    return df_train, df_test  


def load_model(model_name, device):  
    if model_name == 'BERT':  
        model = BertForSequenceClassification.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)  
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')  
    elif model_name in ['T5_base', 'T5_large']:  
        model_size = 't5-base' if model_name == 'T5_base' else 't5-large'  
        model = T5EncoderModel.from_pretrained(model_size)  
        tokenizer = T5Tokenizer.from_pretrained(model_size)  
    else:  
        raise ValueError(f"Invalid model_name: {model_name}. Choose from 'BERT', 'T5_base', or 'T5_large'.")  
    model = model.to(device)  # Move the model to the GPU if available  
    model.eval()  
    return model, tokenizer  


def train_model(classifier, train_embed, train_labels):  
    # Train the model and return it  
    classifier.fit(train_embed, train_labels)  
    return classifier  


def evaluate_model(classifier, test_embed, test_labels, model_name, dataset_name, classifier_name):    
    # Evaluate the model and return metrics    
    test_pred = classifier.predict(test_embed)    

    # Calculate metrics    
    accuracy = accuracy_score(test_labels, test_pred)    
    precision = precision_score(test_labels, test_pred)    
    recall = recall_score(test_labels, test_pred)    
    f1 = f1_score(test_labels, test_pred)    
    roc_auc = roc_auc_score(test_labels, classifier.predict_proba(test_embed)[:, 1])  # get the probability estimates of the positive class    

    return dataset_name, model_name, classifier_name, accuracy, precision, recall, roc_auc, f1


def process_data(df, model_name, model, tokenizer, device, dataset_name, seed, undersample_flag=False, undersample_ratio=0.1):  
    # Process data and return tokenized and embedded data  

    if dataset_name in ['Omni', 'Emtree']:  
        df['label'] = "Category: " + df['superLabel'] + "\nTerm: " + df['label']  

    if undersample_flag:  
        # Define the undersampling strategy  
        undersample = RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=seed)  

        # Fit and apply the transform  
        X_resampled, y_resampled = undersample.fit_resample(df[['label']], df['ambiguous'])  

        # Combine X_resampled and y_resampled back into a single DataFrame  
        df = pd.concat([X_resampled, y_resampled], axis=1)  

    texts = df['label'].tolist()  
    labels = df['ambiguous'].astype(int).tolist()  

    # Tokenize input  
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)  

    # Get the embeddings  
    embeddings = []  
    with torch.no_grad():  
        for i in tqdm(range(encodings.input_ids.size(0)), desc='Processing data'):  
            inputs = {key: val[i].unsqueeze(0) for key, val in encodings.items()}  
            outputs = model(**inputs)  
            if model_name == 'BERT':  
                embeddings.append(outputs.hidden_states[-1][:, 0, :].squeeze().cpu().numpy())  
            elif model_name in ['T5_base', 'T5_large']:  
                embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())  
            else:  
                raise ValueError(f"Invalid model_name: {model_name}. Choose from 'BERT', 'T5_base', or 'T5_large'.")  

    # Convert the list of embeddings to a numpy array  
    embeddings = np.array(embeddings)  

    return embeddings, labels  


# Apply SMOTE if flag is set  
def apply_smote(embeddings, labels, seed, smote_flag=False, smote_ratio=0.1):  
    if smote_flag:  
        smote = SMOTE(sampling_strategy=smote_ratio, random_state=seed)  
        embeddings, labels = smote.fit_resample(embeddings, labels)  
    return embeddings, labels  


def main():
    parser = argparse.ArgumentParser(description="Run text classification pipeline.")
    parser.add_argument('--model_name', type=str, default='BERT', choices=['BERT', 'T5_base', 'T5_large'], help='Model name')
    parser.add_argument('--dataset_name', type=str, default='Emtree', choices=['Omni', 'Emtree', 'Mesh'], help='Dataset name')
    parser.add_argument('--undersample_flag', action='store_true', help='Flag to undersample data')
    parser.add_argument('--undersample_ratio', type=float, default=0.1, help='Undersample ratio')
    parser.add_argument('--smote_flag', action='store_true', help='Flag to apply SMOTE')
    parser.add_argument('--smote_ratio', type=float, default=0.2, help='SMOTE ratio')
    parser.add_argument('--prediction_flag', action='store_true', help='Flag to make predictions')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  
    args = parser.parse_args()

    # Generate a consistent timestamp at the start of the script    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    
    seed = args.seed
    print(f"Running pipeline with model_name={args.model_name}, dataset_name={args.dataset_name}, undersample_flag={args.undersample_flag}, undersample_ratio={args.undersample_ratio}, smote_flag={args.smote_flag}, prediction_flag={args.prediction_flag}")  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    print(f"Using {device}")    
    # Load data with the current seed  
    df_train, df_test = load_data(args.dataset_name, seed)      

    # Load model and tokenizer  
    model, tokenizer = load_model(args.model_name, device)  

    # Process data  
    print("\nTrain set:")  
    train_embed, train_labels = process_data(df_train, args.model_name, model, tokenizer, device, args.dataset_name, seed, undersample_flag=args.undersample_flag, undersample_ratio=args.undersample_ratio)  
    print("\nTest set:")  
    test_embed, test_labels = process_data(df_test, args.model_name, model, tokenizer, device, args.dataset_name, seed)  

    # Apply SMOTE if flag is set  
    train_embed, train_labels = apply_smote(train_embed, train_labels, seed, smote_flag=args.smote_flag, smote_ratio=args.smote_ratio)  

    # Scale the data  
    scaler = StandardScaler()  
    train_embed_scaled = scaler.fit_transform(train_embed)  
    test_embed_scaled = scaler.transform(test_embed)  

    if args.prediction_flag:  
        # For XGBoost model  
        clf = xgb.XGBClassifier()  
        clf.fit(train_embed_scaled, train_labels)  

        # Make predictions on the test set  
        test_pred = clf.predict(test_embed_scaled)  

        # Get the probability estimates of the positive class  
        test_pred_proba = clf.predict_proba(test_embed_scaled)[:, 1]  # get the probability estimates of the positive class  

        # Create a DataFrame with results  
        # Create a copy of df_test  
        df_results = df_test.copy()  

        # Add the new columns to df_results  
        df_results['Predicted Label'] = test_pred  
        df_results['Probability Estimate'] = test_pred_proba  

        df_results_filtered = df_results  
        # Calculate metrics  
        accuracy = accuracy_score(test_labels, test_pred)  
        precision = precision_score(test_labels, test_pred)  
        recall = recall_score(test_labels, test_pred)  
        f1 = f1_score(test_labels, test_pred)  
        roc_auc = roc_auc_score(test_labels, clf.predict_proba(test_embed_scaled)[:, 1])  # get the probability estimates of the positive class  

        # Print the metrics  
        print("Accuracy:", accuracy)  
        print("Precision:", precision)  
        print("Recall:", recall)  
        print("ROC AUC:", roc_auc)  
        print("F1 Score:", f1)  


        df_results_filtered.to_csv(f'Predictions/df_predictions_xgboost_bert_omni_{timestamp}.csv', index=False)  
    else:  
        # Create an empty DataFrame to store the metrics    
        metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'F1 Score'])    

        classifiers = [    
            ('Majority Class', DummyClassifier(strategy="most_frequent")),    
            ('Stratified', DummyClassifier(strategy="stratified")),    
            ('Logistic Regression', LogisticRegression(max_iter=5000)),    
            ('Random Forest', RandomForestClassifier()),    
            ('XGBoost', xgb.XGBClassifier()),    
            ('Naive Bayes', GaussianNB()),    
            ('SVC', SVC(probability=True))    
        ]    
        for classifier_name, classifier in classifiers:    
            print(classifier_name)    
            trained_classifier = train_model(classifier, train_embed_scaled, train_labels)    
            args.dataset_name, args.model_name, classifier_name, accuracy, precision, recall, roc_auc, f1 = evaluate_model(trained_classifier, test_embed_scaled, test_labels, args.model_name, args.dataset_name, classifier_name)    

            # Create a DataFrame for the current model's metrics    
            current_metrics_df = pd.DataFrame({    
                'Dataset': [args.dataset_name],    
                'Transformer Model': [args.model_name],    
                'Method': [classifier_name],    
                'Accuracy': [accuracy],    
                'Precision': [precision],    
                'Recall': [recall],    
                'ROC AUC': [roc_auc],    
                'F1 Score': [f1]    
            })    

            # Print current metrics in a tabular format    
            print(current_metrics_df.to_string(index=False))    

            # Add the current model's metrics to the DataFrame    
            metrics_df = pd.concat([metrics_df, current_metrics_df], ignore_index=True)    

        # Print the DataFrame    
        print(metrics_df)    

        # Save the DataFrame to a CSV file    
        metrics_df.to_csv(f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/pipeline_metrics_final_{args.model_name.lower()}_{args.dataset_name.lower()}_{timestamp}.csv', index=False)    

if __name__ == "__main__":  
    main()






# # =============================================================================================================================
# # Main to run 5 times everything with different seed to calculate confidence intervals
# def main():    
#     parser = argparse.ArgumentParser(description="Run text classification pipeline.")    
#     parser.add_argument('--model_name', type=str, default='BERT', choices=['BERT', 'T5_base', 'T5_large'], help='Model name')    
#     parser.add_argument('--dataset_name', type=str, default='Emtree', choices=['Omni', 'Emtree', 'Mesh'], help='Dataset name')    
#     parser.add_argument('--undersample_flag', action='store_true', help='Flag to undersample data')    
#     parser.add_argument('--undersample_ratio', type=float, default=0.1, help='Undersample ratio')    
#     parser.add_argument('--smote_flag', action='store_true', help='Flag to apply SMOTE')    
#     parser.add_argument('--smote_ratio', type=float, default=0.1, help='SMOTE ratio')    
#     parser.add_argument('--prediction_flag', action='store_true', help='Flag to make predictions')    
#     args = parser.parse_args()    

#     # Generate a consistent timestamp at the start of the script    
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    

#     print(f"Running pipeline with model_name={args.model_name}, dataset_name={args.dataset_name}, undersample_flag={args.undersample_flag}, undersample_ratio={args.undersample_ratio}, smote_flag={args.smote_flag}, prediction_flag={args.prediction_flag}")    

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
#     print(f"Using {device}")    

#     # List of seeds for different runs    
#     seeds = [42, 43, 44, 45, 46]    

#     # Create an empty DataFrame to store the aggregated metrics    
#     aggregated_metrics_df = pd.DataFrame(columns=['Seed', 'Model', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'F1 Score'])    

#     for seed in seeds:    
#         print(f"\nRunning with seed={seed}")    

#         # Load data with the current seed    
#         df_train, df_test = load_data(args.dataset_name, seed)    

#         # Load model and tokenizer    
#         model, tokenizer = load_model(args.model_name, device)    

#         # Process data    
#         print("\nTrain set:")    
#         train_embed, train_labels = process_data(df_train, args.model_name, model, tokenizer, device, args.dataset_name, seed, undersample_flag=args.undersample_flag, undersample_ratio=args.undersample_ratio)    
#         print("\nTest set:")    
#         test_embed, test_labels = process_data(df_test, args.model_name, model, tokenizer, device, args.dataset_name, seed)    

#         # Apply SMOTE if flag is set    
#         train_embed, train_labels = apply_smote(train_embed, train_labels, seed, smote_flag=args.smote_flag, smote_ratio=args.smote_ratio)    

#         # Scale the data    
#         scaler = StandardScaler()    
#         train_embed_scaled = scaler.fit_transform(train_embed)    
#         test_embed_scaled = scaler.transform(test_embed)    

#         if args.prediction_flag:    
#             # For XGBoost model    
#             clf = xgb.XGBClassifier()    
#             clf.fit(train_embed_scaled, train_labels)    

#             # Make predictions on the test set    
#             test_pred = clf.predict(test_embed_scaled)    

#             # Get the probability estimates of the positive class    
#             test_pred_proba = clf.predict_proba(test_embed_scaled)[:, 1]  # get the probability estimates of the positive class    

#             # Create a DataFrame with results    
#             df_results = df_test.copy()    

#             # Add the new columns to df_results    
#             df_results['Predicted Label'] = test_pred    
#             df_results['Probability Estimate'] = test_pred_proba    

#             df_results_filtered = df_results    
#             # Calculate metrics    
#             accuracy = accuracy_score(test_labels, test_pred)    
#             precision = precision_score(test_labels, test_pred)    
#             recall = recall_score(test_labels, test_pred)    
#             f1 = f1_score(test_labels, test_pred)    
#             roc_auc = roc_auc_score(test_labels, clf.predict_proba(test_embed_scaled)[:, 1])  # get the probability estimates of the positive class    

#             # Print the metrics    
#             print("Accuracy:", accuracy)    
#             print("Precision:", precision)    
#             print("Recall:", recall)    
#             print("ROC AUC:", roc_auc)    
#             print("F1 Score:", f1)    

#             df_results_filtered.to_csv(f'Predictions/df_predictions_xgboost_bert_omni_{timestamp}_seed_{seed}.csv', index=False)    
#         else:    
#             # Create an empty DataFrame to store the metrics    
#             metrics_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'F1 Score'])    

#             classifiers = [    
#                 ('Majority Class', DummyClassifier(strategy="most_frequent")),    
#                 ('Stratified', DummyClassifier(strategy="stratified"))    
#                 # ('Logistic Regression', LogisticRegression(max_iter=5000)),    
#                 # ('Random Forest', RandomForestClassifier()),    
#                 # ('XGBoost', xgb.XGBClassifier()),    
#                 # ('Naive Bayes', GaussianNB()),    
#                 # ('SVC', SVC(probability=True))    
#             ]    
#             for classifier_name, classifier in classifiers:    
#                 print(classifier_name)    
#                 trained_classifier = train_model(classifier, train_embed_scaled, train_labels)    
#                 args.dataset_name, args.model_name, classifier_name, accuracy, precision, recall, roc_auc, f1 = evaluate_model(trained_classifier, test_embed_scaled, test_labels, args.model_name, args.dataset_name, classifier_name)    

#                 # Create a DataFrame for the current model's metrics    
#                 current_metrics_df = pd.DataFrame({    
#                     'Seed': [seed],    
#                     'Dataset': [args.dataset_name],    
#                     'Transformer Model': [args.model_name],    
#                     'Method': [classifier_name],    
#                     'Accuracy': [accuracy],    
#                     'Precision': [precision],    
#                     'Recall': [recall],    
#                     'ROC AUC': [roc_auc],    
#                     'F1 Score': [f1]    
#                 })    

#                 # Print current metrics in a tabular format    
#                 print(current_metrics_df.to_string(index=False))    

#                 # Add the current model's metrics to the DataFrame    
#                 metrics_df = pd.concat([metrics_df, current_metrics_df], ignore_index=True)    

#             # Print the DataFrame    
#             print(metrics_df)    

#             # Add the current run's metrics to the aggregated DataFrame    
#             aggregated_metrics_df = pd.concat([aggregated_metrics_df, metrics_df], ignore_index=True)    

#     # Save the aggregated metrics to a CSV file    
#     aggregated_metrics_df.to_csv(f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/aggregated_pipeline_metrics_{args.model_name.lower()}_{args.dataset_name.lower()}_{timestamp}.csv', index=False)    


# if __name__ == "__main__":    
#     main()


# #=============================================================================================================================
#Main to test different undersampling ratios.

# def main():
#     parser = argparse.ArgumentParser(description="Run text classification pipeline.")
#     parser.add_argument('--model_name', type=str, default='BERT', choices=['BERT', 'T5_base', 'T5_large'], help='Model name')
#     parser.add_argument('--dataset_name', type=str, default='Emtree', choices=['Omni', 'Emtree', 'Mesh'], help='Dataset name')
#     parser.add_argument('--undersample_flag', action='store_true', help='Flag to undersample data')
#     parser.add_argument('--undersample_ratio', type=float, default=0.1, help='Undersample ratio')
#     parser.add_argument('--smote_flag', action='store_true', help='Flag to apply SMOTE')
#     parser.add_argument('--smote_ratio', type=float, default=0.1, help='SMOTE ratio')
#     parser.add_argument('--prediction_flag', action='store_true', help='Flag to make predictions')
#     args = parser.parse_args()  

#     print(f"Running with model_name={args.model_name}, dataset_name={args.dataset_name}, undersample_flag={args.undersample_flag}, smote_flag={args.smote_flag}, prediction_flag={args.prediction_flag}")  

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
#     print(f"Using {device}")  
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  

#     # Load data  
#     df_train, df_test = load_data(args.dataset_name)  

#     # Count the occurrences of each class  
#     class_counts = df_train['ambiguous'].value_counts()  

#     # Calculate and print the ratio  
#     ratio = class_counts.min() / class_counts.max()  
#     print("Original ratio in training set: ", ratio)  

#     # Load model and tokenizer  
#     model, tokenizer = load_model(args.model_name, device)  

#     # Create an empty DataFrame to store the metrics  
#     metrics_df = pd.DataFrame(columns=['UnderSample Ratio', 'Dataset', 'Transformer Model', 'Method', 'Accuracy', 'Precision', 'Recall', 'ROC AUC', 'F1 Score'])  

#     for undersample_ratio in args.undersample_ratios:  # Loop over undersample ratios  
#         print(f"\nRunning with undersample_ratio={undersample_ratio}")  

#         # Process data  
#         print("\nTrain set:")  
#         train_embed, train_labels = process_data(df_train, args.model_name, model, tokenizer, device, args.dataset_name, undersample_flag=args.undersample_flag, undersample_ratio=undersample_ratio)  
#         print("\nTest set:")  
#         test_embed, test_labels = process_data(df_test, args.model_name, model, tokenizer, device, args.dataset_name)  

#         # Apply SMOTE if flag is set  
#         train_embed, train_labels = apply_smote(train_embed, train_labels, smote_flag=args.smote_flag, smote_ratio=args.smote_ratio)  

#         # Scale the data  
#         scaler = StandardScaler()  
#         train_embed_scaled = scaler.fit_transform(train_embed)  
#         test_embed_scaled = scaler.transform(test_embed)  

#         classifiers = [  
#             ('Logistic Regression', LogisticRegression(max_iter=5000)),  
#             ('Random Forest', RandomForestClassifier()),  
#             ('XGBoost', xgb.XGBClassifier()),  
#             ('Naive Bayes', GaussianNB()),  
#             ('SVC', SVC(probability=True))  
#         ]  
#         for classifier_name, classifier in classifiers:  
#             print(classifier_name)  
#             trained_classifier = train_model(classifier, train_embed_scaled, train_labels)  
#             dataset_name, model_name, classifier_name, accuracy, precision, recall, roc_auc, f1 = evaluate_model(trained_classifier, test_embed_scaled, test_labels, args.model_name, args.dataset_name)  

#             # Create a DataFrame for the current model's metrics  
#             current_metrics_df = pd.DataFrame({  
#                 'UnderSample Ratio': [undersample_ratio],  # Add undersample_ratio to the metrics  
#                 'Dataset': [dataset_name],  
#                 'Transformer Model': [model_name],  
#                 'Method': [classifier_name],  
#                 'Accuracy': [accuracy],  
#                 'Precision': [precision],  
#                 'Recall': [recall],  
#                 'ROC AUC': [roc_auc],  
#                 'F1 Score': [f1]  
#             })  

#             # Print current metrics in a tabular format  
#             print(current_metrics_df.to_string(index=False))  
#             current_metrics_df.to_csv(f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/current_pipeline_metrics_final_{args.model_name.lower()}_{args.dataset_name.lower()}_{timestamp}.csv', index=False)  
#             # Add the current model's metrics to the DataFrame  
#             metrics_df = pd.concat([metrics_df, current_metrics_df], ignore_index=True)  

#     # Print the DataFrame  
#     print(metrics_df)  

#     # Save the DataFrame to a CSV file  
#     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  
#     metrics_df.to_csv(f'/home/dpapadopoulos/dsls-papadopoulos-ambiguity-scoring-thesis/Results/pipeline_metrics_final_{args.model_name.lower()}_{args.dataset_name.lower()}_{timestamp}.csv', index=False)

