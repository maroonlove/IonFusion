# Figure scripts

These Python scripts draw figures from private CSV inputs. Input data,
credentials, server paths, and generated figures are not included.
Dependencies: numpy, pandas, matplotlib.

Set DRAW_FONT_PATH only if a local font file is needed. Outputs go under
outputs/ by default. Keep input CSV files outside Git.

- plot_go_aupr_by_depth.py: set DRAW_DATA_DIR with go_aupr and go_label_stats subdirectories.
- fmax_zhuzhuangtu.py: pass a CSV with method, class, seq_identity, Fmax, AUPR, ICAUPR, DPAUPR.
- ablation_zhexian.py: pass a directory with ablation_evolution.csv and ablation_mechanisms.csv; each needs Model, Accuracy, Precision, Recall, F1-score.
- zhuzhuang_leida.py: pass a directory with table1_macro_comparison.csv, table2_per_class_precision.csv, table3_per_class_recall.csv, table4_ablation_macro.csv, table5_per_class_f1score.csv.

The three scripts that accept a CSV or directory argument support --help.
For the GO depth script, set DRAW_DATA_DIR before running it.
