import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def show_misclassified(y_true_tensor, preds_tensor, idx_array, detection_str, encoder):
    """Print misclassified examples using the index‑map CSV.

    Arguments:
        y_true_tensor: ground truth labels as torch tensor
        preds_tensor: predicted labels as torch tensor (same length)
        idx_array: numpy array of original indices corresponding to the
                   samples in y_true_tensor/preds_tensor (e.g. idx_test)
        detection_str: value of `detection` parameter used to pick dataset
        encoder: sklearn LabelEncoder instance for inverse transforming labels
    """
    base_dir_root = r"C:\BP\pythonProject1\data_rysy"
    if detection_str == '':
        base_dir = os.path.join(base_dir_root, "rys_trening_data_Beno")
    elif detection_str == '_detected':
        base_dir = os.path.join(base_dir_root, "rys_trening_data_Beno_detected")
    elif detection_str == '_detected_manual':
        base_dir = os.path.join(base_dir_root, "rys_trening_data_Beno_detected_manual", "rys_trening_data_Beno_detected_manual")
    else:
        raise ValueError(f"unknown detection value: {detection_str}")

    csv_path = os.path.join(base_dir, "index_map.csv")
    index_map = pd.read_csv(csv_path)

    wrong = (preds_tensor != y_true_tensor).nonzero(as_tuple=True)[0]
    print(f"Number wrong: {len(wrong)}")
    for test_idx in wrong.tolist():
        true_label = encoder.inverse_transform([y_true_tensor[test_idx].item()])[0]
        pred_label = encoder.inverse_transform([preds_tensor[test_idx].item()])[0]
        original_idx = idx_array[test_idx]
        image_path = index_map.at[original_idx, 'path']
        print(f"Index {test_idx} (Orig {original_idx}): {image_path}")
        print(f"  True: {true_label}, Predicted: {pred_label}")
        print()