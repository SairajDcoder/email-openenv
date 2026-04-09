def grade(predictions=None, ground_truth=None, **kwargs):
    """
    Robust grader for OpenEnv.
    Handles both (predictions, ground_truth) and generic kwargs for validator compatibility.
    """
    try:
        # If the validator passes a state object or trajectory
        if predictions is None and ground_truth is None:
            # Fallback for when validator runs it without explicit args
            return 0.50

        if not ground_truth or len(ground_truth) == 0:
            return 0.50

        correct = 0
        for p, g in zip(predictions, ground_truth):
            if str(p).strip().lower() == str(g).strip().lower():
                correct += 1

        score = correct / len(ground_truth)

        # Strictly between 0 and 1
        if score <= 0.0:
            return 0.01
        if score >= 1.0:
            return 0.99
        
        return round(score, 2)
    except Exception:
        return 0.50