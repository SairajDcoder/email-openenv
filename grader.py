def grade(predictions, ground_truth):
    correct = 0
    for p, g in zip(predictions, ground_truth):
        if p == g:
            correct += 1

    return round(correct / len(ground_truth), 2)