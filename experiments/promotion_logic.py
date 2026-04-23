import mlflow
from mlflow.tracking import MlflowClient

def promote_to_production_if_better(model_name="PakSentinel_Best_LogReg", metric="metrics.f1_weighted", improvement_threshold=0.01):
    """
    Task 6 Requirement:
    Moves a model from Staging to Production only if its F1-weighted 
    exceeds the current Production model by at least 1% (0.01).
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name("PakSentinel_FakeNews")
    if not experiment:
        print("Error: Experiment 'PakSentinel_FakeNews' not found.")
        return

    # 1. Fetch the best run
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=[f"{metric} DESC"]
    )

    if not runs:
        print("Error: No runs found. Please run ablation_study.py first.")
        return

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_run_f1 = best_run.data.metrics.get(metric.split('.')[-1], 0)

    print(f"\n--- Model Promotion Logic ---")
    print(f"Top Run ID: {best_run_id}")
    print(f"Top Run F1-Weighted: {best_run_f1:.4f}")

    # 2. Register this run as a model version and put it in Staging
    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(model_uri, model_name)
    
    # Delay to ensure DB processes the registry
    import time
    time.sleep(1)

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"-> Version {model_version.version} registered and moved to Staging.")

    # 3. Retrieve current Production model's metric
    production_versions = [v for v in client.search_model_versions(f"name='{model_name}'") if v.current_stage == "Production"]
    
    if not production_versions:
        print("-> No existing Production model. Promoting directly to Production!")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        return

    # Assuming the first one returned is the active production
    prod_version = production_versions[0]
    prod_run_id = prod_version.run_id
    prod_run = client.get_run(prod_run_id)
    prod_f1 = prod_run.data.metrics.get(metric.split('.')[-1], 0)

    print(f"-> Current Production Model (v{prod_version.version}) F1-Weighted: {prod_f1:.4f}")

    # 4. Check the 1% condition
    if best_run_f1 >= (prod_f1 + improvement_threshold):
        print(f"-> SUCCESS: Staging model outperforms Production by >= {improvement_threshold * 100}%!")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"-> Model v{model_version.version} officially PROMOTED to Production.")
    else:
        print(f"-> REJECTED: Staging model ({best_run_f1:.4f}) did not beat Production ({prod_f1:.4f}) by {improvement_threshold * 100}%. Left in Staging.")

if __name__ == "__main__":
    promote_to_production_if_better()