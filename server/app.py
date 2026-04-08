import os
from openenv_core.app import create_app

def main():
    # create_app initializes the OpenEnv server for the environment in the current directory
    app = create_app(env_dir=".")
    import uvicorn
    # The hackathon environment usually expects the server on port 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
