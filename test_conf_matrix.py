
import torch
from torchmetrics.functional import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define a synthetic binary classification dataset
# Class 0: Dog, Class 1: Cat
true_labels = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0, 0, 1])  # Ground truth
pred_labels = torch.tensor([0, 1, 0, 0, 1, 1, 1, 0, 0, 1])  # Model predictions

# Compute the confusion matrix
num_classes = 2  # Binary classification: Dog (0), Cat (1)
val_recalls, val_precisions = [], []
best_val_precision = 0.0
best_epoch = 0
for i in range(5):
    conf_matrix = confusion_matrix(pred_labels.cpu(), true_labels.cpu(), task="binary", num_classes=num_classes)

    # Print the confusion matrix
    print("Epoch ", i)
    print(f"\n Confusion Matrix of epoch {i}:")
    print(conf_matrix)

# # Extract precision and recall
# true_positives = conf_matrix[1, 1]  # Correctly predicted Cats
# false_positives = conf_matrix[0, 1]  # Dogs predicted as Cats
# false_negatives = conf_matrix[1, 0]  # Cats predicted as Dogs
# true_negatives = conf_matrix[0, 0]  # Correctly predicted Dogs

# precision_cat = (true_positives / (true_positives + false_positives)).item()
# recall_cat = (true_positives / (true_positives + false_negatives)).item()

# precision_dog = (true_negatives / (true_negatives + false_negatives)).item()
# recall_dog = (true_negatives / (true_negatives + false_positives)).item()

# print("\nPrecision and Recall per Class:")
# print(f"Class Dog (0): Precision = {precision_dog:.2f}, Recall = {recall_dog:.2f}")
# print(f"Class Cat (1): Precision = {precision_cat:.2f}, Recall = {recall_cat:.2f}")


    true_positives = conf_matrix.diag()
    total_predicted = conf_matrix.sum(dim=0)
    total_actual = conf_matrix.sum(dim=1)

    # Class-wise precision and recall
    if i == 2:
        class_precisions = (true_positives / total_predicted).nan_to_num(0)
        class_recalls = (true_positives / total_actual).nan_to_num(0)
    else:
        class_precisions = (true_positives / total_predicted).nan_to_num(0) - torch.tensor([0.2, 0.1])
        class_recalls = (true_positives / total_actual).nan_to_num(0) - torch.tensor([0.2, 0.2])

    # print("Precisions shape: ", precisions.size())
    # print("Precisions: \n", precisions)
    val_precision = class_precisions.mean().item()
    val_recall = class_recalls.mean().item()
    val_precisions.append(class_precisions)
    val_recalls.append(class_recalls)
    print(f"Precision: {val_precision:.2f}, Recall: {val_recall:.2f}")

    if val_precision > best_val_precision:
        best_epoch = i
        best_val_precision = val_precision


print("Best model at epoch ", best_epoch)
print("val_precisions shape: ", len(val_precisions))
print("val_precisions: ", val_precisions[best_epoch].mean())


# conf_matrix = conf_matrix
# print(type(conf_matrix))
# conf_matrix_np = conf_matrix.cpu().numpy()
# fig, ax = plt.subplots(figsize=(6, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Dog', 'Cat'], yticklabels=['Dog', 'Cat'], ax=ax)
# ax.set_xlabel('Predicted')
# ax.set_ylabel('True')
# ax.set_title('Confusion Matrix')
# plt.show()