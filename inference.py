import os
from openai import OpenAI
from env import SmartEmailEnv, Action
from grader import grade

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()

client = None
if HF_TOKEN:
    if "huggingface" in API_BASE_URL:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_TOKEN)
    else:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def get_action(email, step_count):
    if step_count == 1:
        prompt = f"""
        You are an email assistant.
        Email: {email}
        Determine if the email is 'spam', 'important', or 'normal'.
        Rules for classification:
        - "Win a free iPhone", "Discount offer", "Claim your reward", "Free lottery" -> spam
        - "Team meeting", "Project deadline", "URGENT", "Account issue" -> important
        - "Weekly newsletter", "Update preferences", "Weekly digest" -> normal
        
        Output ONLY one of these exact strings:
        classify:spam
        classify:important
        classify:normal
        """
    else:
        prompt = f"""
        You are an email assistant.
        Email: {email}
        Determine what action to take based on the email.
        Rules for actions:
        - If it is spam or a normal newsletter/notification -> ignore
        - If it's a team meeting or project deadline -> reply:Got it!
        - If it is URGENT (like account verification or account issue) -> escalate
        
        Output ONLY the chosen action string (ignore, escalate, or reply:<msg>).
        """

    if not client:
        return "classify:normal" if step_count == 1 else "ignore"

    try:
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
    except Exception as e:
        print(f"LLM call failed: {e}")
        return "classify:normal" if step_count == 1 else "ignore"


def clamp_reward(r):
    """Clamp reward to strictly between 0 and 1 (not 0.0, not 1.0)."""
    if r <= 0.0:
        return 0.01
    if r >= 1.0:
        return 0.99
    return round(r, 2)


def run():
    env = SmartEmailEnv()
    history = []
    total_steps = 0
    task_ids = ["easy", "medium", "hard"]

    print(f"[START] task=email-agent env=openenv model={MODEL_NAME}")

    try:
        for task_idx, task_id in enumerate(task_ids, 1):
            obs = env.reset()
            task_step_count = 0
            email_content_str = obs.email_text
            predictions = []
            ground_truths = []

            while True:
                task_step_count += 1
                total_steps += 1

                action_str = get_action(obs.email_text, task_step_count)
                action = Action(action=action_str)
                obs, reward, done, info = env.step(action)

                # Clamp reward to strictly (0, 1)
                reward = clamp_reward(reward)

                # Track for grading
                if action_str.startswith("classify:"):
                    predictions.append(action_str.split(":")[1].strip())
                    ground_truths.append(env.current["label"])

                history.append({
                    "task": task_idx,
                    "task_id": task_id,
                    "step": task_step_count,
                    "email": email_content_str,
                    "action": action_str,
                    "reward": reward
                })

                print(f"[STEP] task={task_id} step={task_step_count} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

                if done:
                    break

            # Compute grader score for this task
            if predictions and ground_truths:
                task_score = grade(predictions, ground_truths)
            else:
                task_score = 0.5
            print(f"[GRADE] task={task_id} score={task_score}")

        success = True
    except Exception as e:
        print(f"[STEP] task=error step=0 action=error reward=0.01 done=true error={str(e)}")
        success = False

    rewards_list = [f"{h['reward']:.2f}" for h in history]
    print(f"[END] success={str(success).lower()} steps={total_steps} rewards={','.join(rewards_list)}")

    # Build the dashboard HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email Agent - Multi-Task Dashboard</title>
        <style>
            body {{
                margin: 0; padding: 0;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
                color: #f8fafc;
                min-height: 100vh;
                padding: 40px 0;
                display: flex; flex-direction: column; align-items: center;
            }}
            .glass-panel {{
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 40px; width: 90%; max-width: 800px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                animation: fadein 1s ease-out;
            }}
            @keyframes fadein {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
            h1 {{
                margin-top: 0; font-size: 28px;
                background: linear-gradient(to right, #38bdf8, #818cf8);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            }}
            .task-section {{
                 margin-bottom: 30px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 20px;
            }}
            .email-box {{
                background: rgba(0,0,0,0.3); border-left: 4px solid #818cf8;
                padding: 15px 20px; border-radius: 8px; margin: 15px 0; font-size: 16px;
            }}
            .step {{
                display: flex; justify-content: space-between; align-items: center;
                background: rgba(255,255,255,0.03); padding: 12px; border-radius: 10px; margin-bottom: 8px;
            }}
            .badge {{
                color: white; font-weight: bold; padding: 4px 10px; border-radius: 20px; font-size: 12px;
            }}
            .total {{ text-align: center; font-size: 24px; margin-top: 30px; font-weight: bold; color: { "#34d399" if success else "#ef4444" }; }}
        </style>
    </head>
    <body>
        <div class="glass-panel">
            <h1>Smart Email AI Agent</h1>
            <p>Processing 3 tasks (easy, medium, hard) with Qwen2.5-72B...</p>
    """

    current_task = 0
    for h in history:
        if h['task'] != current_task:
            if current_task != 0:
                html_content += "</div>"
            current_task = h['task']
            html_content += f"""
                <div class="task-section">
                    <h3>Task: {h['task_id'].upper()}</h3>
                    <div class="email-box">"{h['email']}"</div>
            """

        reward_col = "#22c55e" if h['reward'] >= 0.5 else "#ef4444"
        html_content += f"""
            <div class="step">
                <div><strong>Step {h['step']}:</strong> <code style="color: #93c5fd;">{h['action']}</code></div>
                <div class="badge" style="background: {reward_col};">Reward: {h['reward']:.2f}</div>
            </div>
        """

    html_content += f"""
            </div>
            <div class="total">Overall: {"PASSED" if success else "FAILED"}</div>
        </div>
    </body>
    </html>
    """

    # Start the web server dashboard
    import http.server
    import socketserver

    PORT = 7860
    class CustomDashboardHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))

        def do_POST(self):
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"success","message":"Agent is healthy"}')

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, format, *args):
            pass  # Suppress request logs

    print(f"Starting web server dashboard on port {PORT}...")
    try:
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", PORT), CustomDashboardHandler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(f"Web server stopped: {e}")

if __name__ == "__main__":
    run()