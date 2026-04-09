import os
from openenv_core.app import create_app

# The hackathon environment uses this app object for deployment
app = create_app(env_dir=".")

def main():
    import uvicorn
    # The hackathon environment usually expects the server on port 7860 or 8000
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
