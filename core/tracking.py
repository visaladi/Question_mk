# core/tracking.py
import os, mlflow, time, json, tempfile
from contextlib import contextmanager

def mlflow_init():
    uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)
    exp = os.getenv("MLFLOW_EXPERIMENT", "default")
    mlflow.set_experiment(exp)

@contextmanager
def start_run(run_name: str, tags: dict | None = None):
    mlflow_init()
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        t0 = time.time()
        try:
            yield
        finally:
            mlflow.log_metric("wall_time_sec", time.time() - t0)

def log_params(d: dict): 
    mlflow.log_params({k:str(v) for k,v in d.items()})

def log_metrics(d: dict): 
    mlflow.log_metrics(d)

def log_text(text: str, path: str): 
    mlflow.log_text(text, artifact_file=path)

def log_prompt(prompt: str, prompt_name: str = "prompt"):
    """Log prompt as both parameter and text artifact"""
    # Log as parameter (truncated if too long)
    prompt_preview = prompt[:250] + "..." if len(prompt) > 250 else prompt
    mlflow.log_param(f"{prompt_name}_preview", prompt_preview)
    
    # Log full prompt as text artifact
    mlflow.log_text(prompt, artifact_file=f"{prompt_name}.txt")

def log_conversation(messages: list, conversation_name: str = "conversation"):
    """Log conversation/chat history as JSON artifact"""
    log_json(messages, f"{conversation_name}.json")

def log_json(obj, path: str):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tmp.close()  # Close the file handle before writing
    with open(tmp.name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    mlflow.log_artifact(tmp.name, artifact_path=os.path.dirname(path) or None)
    os.unlink(tmp.name)

def log_artifact(path: str, artifact_subdir: str | None = None):
    mlflow.log_artifact(path, artifact_path=artifact_subdir)
