import os
from openai import OpenAI
from env import SmartEmailEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN is required")

if "huggingface" in API_BASE_URL:
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=HF_TOKEN)
else:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def get_action(email):
    prompt = f"""
    You are an email assistant.

    Steps:
    1. classify email as spam/important/normal
    2. choose action (ignore/reply/escalate)

    Email: {email}

    Output format:
    classify:<label>
    OR
    reply:<text>
    OR
    ignore
    OR
    escalate
    """

    if hasattr(client, "chat_completion"):
        res = client.chat_completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )

    return res.choices[0].message.content.strip().lower().split("\n")[0]


def run():
    env = SmartEmailEnv()
    obs = env.reset()

    print(f"[START] task=email-agent env=openenv model={MODEL_NAME}")

    rewards = []
    step_count = 0

    try:
        while True:
            step_count += 1

            action_str = get_action(obs.email_text)

            action = Action(action=action_str)

            obs, reward, done, info = env.step(action)

            rewards.append(f"{reward:.2f}")

            print(f"[STEP] step={step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

            if done:
                success = sum([float(r) for r in rewards]) > 0
                break

    except Exception as e:
        print(f"[STEP] step={step_count} action=error reward=0.00 done=true error={str(e)}")
        success = False

    print(f"[END] success={str(success).lower()} steps={step_count} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run()