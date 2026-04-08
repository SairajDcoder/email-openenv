def grade(predictions, ground_truth):
    if not ground_truth or len(ground_truth) == 0:
        return 0.5

    correct = 0
    for p, g in zip(predictions, ground_truth):
        if p == g:
            correct += 1

    score = correct / len(ground_truth)

    # Clamp to strictly between 0 and 1 (not 0.0 and not 1.0)
    if score <= 0.0:
        score = 0.01
    if score >= 1.0:
        score = 0.99

    return round(score, 2)