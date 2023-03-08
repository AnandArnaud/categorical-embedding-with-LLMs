import pandas as pd

from functions.utils import save_scores_to_csv
from functions.data_loader import load_datasets, fetch_colleges
from functions.SuperVectorizer_baseline import run_baseline_model
from functions.bert_encoder import run_model_using_bert_embeddings


all_datasets = load_datasets()

baseline_results_df = run_baseline_model(all_datasets)

results_using_bert_embeddings_df1 = run_model_using_bert_embeddings(all_datasets, 
                                                                        sentence_strategy="only_value", 
                                                                        embedding_strategy="last_four_hidden_state_concatenate"
                                                                    )

results_using_bert_embeddings_df2 = run_model_using_bert_embeddings(all_datasets, 
                                                                        sentence_strategy="value_and_column", 
                                                                        embedding_strategy="last_four_hidden_state_concatenate"
                                                                    )

results_using_bert_embeddings_df3 = run_model_using_bert_embeddings(all_datasets, 
                                                                        sentence_strategy="value_and_column", 
                                                                        embedding_strategy="last_hidden_state"
                                                                    )

all_results_df = pd.concat([baseline_results_df,
                            results_using_bert_embeddings_df1,
                            results_using_bert_embeddings_df2,
                            results_using_bert_embeddings_df3
                            ],
                            axis = 0)

all_results_df = all_results_df.sort_values(["dataset_name", "strategy"])

save_scores_to_csv(all_results_df, "results.csv")
