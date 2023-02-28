from dirty_cat import datasets

from functions.utils import save_scores_to_csv
from functions.data_loader import load_datasets_from_dirty_cat
from functions.SuperVectorizer_baseline import run_baseline_model
from functions.bert_encoder import run_model_using_bert_embeddings


all_datasets = load_datasets_from_dirty_cat()

baseline_results_df = run_baseline_model(all_datasets)
save_scores_to_csv(baseline_results_df, "baseline_results.csv")


results_using_bert_embeddings_df = run_model_using_bert_embeddings(all_datasets, 
                                                                        sentence_strategy="only_value", 
                                                                        embedding_strategy="last_four_hidden_state_concatenate"
                                                                    )
save_scores_to_csv(results_using_bert_embeddings_df, "bert_only_value_four_hidden_state_results.csv")


results_using_bert_embeddings_df = run_model_using_bert_embeddings(all_datasets, 
                                                                        sentence_strategy="value_and_column", 
                                                                        embedding_strategy="last_four_hidden_state_concatenate"
                                                                    )
save_scores_to_csv(results_using_bert_embeddings_df, "bert_value_col_foour_hidden_state_results.csv")


results_using_bert_embeddings_df = run_model_using_bert_embeddings(all_datasets, 
                                                                        sentence_strategy="value_and_column", 
                                                                        embedding_strategy="last_hidden_state"
                                                                    )
save_scores_to_csv(results_using_bert_embeddings_df, "bert_value_col_last_hidden_state_results.csv")