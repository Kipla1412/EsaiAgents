# agent utility, Singleton pattern
import mlflow
import time

class Tracker:
    def __init__(self, experiment_name= None, nested=True):
        #mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run(run_name=f"session_{int(time.time())}", nested=nested)
        print(f"[MLflow] Started run: {self.run.info.run_id}")

    def log_static(self, llm_config, agent_config):
        mlflow.log_param("llm_method", llm_config.get("method"))
        mlflow.log_param("llm_model", llm_config.get("path"))
        mlflow.log_param("temperature", llm_config.get("temperature"))
        mlflow.log_param("agent_description", agent_config.get("description"))
        mlflow.log_param("max_iterations", agent_config.get("max_iterations", "N/A"))
        mlflow.log_param("prompt_templates", agent_config.get("prompt_templates"))

    def log_turn(self, user_input, response, duration=None):
        mlflow.set_tag("latest_user_input", user_input)
        mlflow.set_tag("latest_response", response)
        mlflow.log_metric("input_length", len(user_input))
        mlflow.log_metric("response_length", len(response))
        if duration:
            mlflow.log_metric("turn_duration_sec", duration)

    def end_run(self):
        mlflow.end_run()
        print("[MLflow] Run ended successfully.")
