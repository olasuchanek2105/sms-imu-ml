from preprocessing.build_cnn_dataset import build_cnn_dataset, load_subject_target_mapping


mapping = load_subject_target_mapping("data/ankieta_score_and_target.csv")

X, y, groups = build_cnn_dataset(
    raw_folder_path="data/cut_to_same_length",
    subject_target_mapping=mapping
)