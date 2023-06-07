import torch
import numpy as np
from seqeval.metrics import f1_score, accuracy_score


def evaluate(model, test, tag_values):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions , true_labels = [], []
    for batch in test:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)

        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]

    print(f'Test_Accuracy: {accuracy_score(pred_tags, valid_tags)} | Test_F1 score: {f1_score([pred_tags], [valid_tags])}')

