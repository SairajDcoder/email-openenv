from pydantic import BaseModel
import random

class Observation(BaseModel):
    email_text: str
    step: int

class Action(BaseModel):
    action: str  # classify:spam / reply:hi / ignore / escalate

class SmartEmailEnv:
    def __init__(self):

        # EASY
        self.tasks_easy = [
            {"email": "Win a free iPhone now!!!", "label": "spam", "expected_action": "ignore"},
            {"email": "Team meeting at 5 PM", "label": "important", "expected_action": "reply"}
        ]

        # MEDIUM
        self.tasks_medium = [
            {"email": "Discount offer just for you", "label": "spam", "expected_action": "ignore"},
            {"email": "Project deadline tomorrow", "label": "important", "expected_action": "reply"},
            {"email": "Weekly newsletter", "label": "normal", "expected_action": "ignore"}
        ]

        # HARD
        self.tasks_hard = [
            {"email": "URGENT: Account verification needed", "label": "important", "expected_action": "escalate"},
            {"email": "Claim your reward now by clicking here", "label": "spam", "expected_action": "ignore"},
            {"email": "Update your notification preferences", "label": "normal", "expected_action": "ignore"}
        ]

        self.current = None
        self.step_count = 0
        self.difficulty = None   # ✅ added

    def reset(self):
        self.step_count = 0

        self.difficulty = random.choice(["easy", "medium", "hard"])

        if self.difficulty == "easy":
            self.current = random.choice(self.tasks_easy)
        elif self.difficulty == "medium":
            self.current = random.choice(self.tasks_medium)
        else:
            self.current = random.choice(self.tasks_hard)

        return Observation(email_text=self.current["email"], step=0)

    def step(self, action: Action):
        self.step_count += 1
        reward = 0

        correct_label = self.current["label"]
        expected_action = self.current["expected_action"]

        try:
            # CLASSIFY
            if action.action.startswith("classify"):
                parts = action.action.split(":")
                if len(parts) < 2:
                    reward = 0.1
                else:
                    predicted = parts[1].strip()
                    if predicted == correct_label:
                        reward = 0.8
                    else:
                        reward = 0.2

            # IGNORE / ESCALATE
            elif action.action in ["ignore", "escalate"]:
                if action.action == expected_action:
                    reward = 0.8
                else:
                    reward = 0.2

            # REPLY
            elif action.action.startswith("reply"):
                if expected_action == "reply":
                    reward = 0.9
                else:
                    reward = 0.1

            else:
                # ❌ invalid action
                reward = 0.1

        except Exception:
            reward = 0.1

        done = self.step_count >= 2

        return (
            Observation(email_text=self.current["email"], step=self.step_count),
            reward,
            done,
            {"difficulty": self.difficulty}   # ✅ added
        )

    def state(self):
        return {
            "email": self.current,
            "difficulty": self.difficulty
        }