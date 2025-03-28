import os
import sys
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(parent_dir)

from ai_data_science_team.agents import DataCleaningAgent # type: ignore
from ai_data_science_team.agents import FeatureEngineeringAgent # type: ignore
from ai_data_science_team.agents import ModelSelectionAgent # type: ignore
from ai_data_science_team.agents import ModelTrainingAgent # type: ignore


def data_cleaning_agent_function(state):
    data_cleaning_agent = DataCleaningAgent(
        model = state['llm'], 
        log=state['LOG'], 
        log_path=state['LOG_PATH'],
        file_name="data_transformation.py", 
        function_name="data_cleaner",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        mlflow_run_id=state['mlflow_run_id']
    )
    raw_df=pd.read_csv(state['raw_df_loc'])
    processed_df_loc=state['processed_df_loc']
    data_cleaning_documentation=state['data_cleaning_documentation']

    data_cleaning_agent.invoke_agent(
        data_raw=raw_df,
        user_instructions="Don't remove outliers when cleaning the data.",
        max_retries=3,
        retry_count=0
    ) 

    with open(data_cleaning_documentation, 'w') as md_file:
        cleaning_steps = data_cleaning_agent.get_recommended_cleaning_steps(markdown=True)
        if hasattr(cleaning_steps, 'data'):
            md_file.write(cleaning_steps.data)
        else:
            md_file.write(str(cleaning_steps))
            print("Warning: Could not extract markdown content properly.")

    data_cleaning_agent.get_data_cleaned().to_csv(processed_df_loc, index=False)
    return {'cleaned_data':data_cleaning_agent.get_data_cleaned()}

def feature_engineering_agent_function(state):
    feature_engineering_agent = FeatureEngineeringAgent(
        model = state['llm'], 
        log=state['LOG'], 
        log_path=state['LOG_PATH'],
        file_name="feature_engineering.py", 
        function_name="feature_engineering",
        overwrite=True,
        human_in_the_loop=False,
        bypass_recommended_steps=False,
        bypass_explain_code=False,
        mlflow_run_id=state['mlflow_run_id']
    )
    df_cleaned = state['cleaned_data']
    X = df_cleaned.drop("Survived", axis=1)
    y = df_cleaned["Survived"]
    feature_engineering_agent.invoke_agent(
    data_features=X, 
    data_target=y,
    max_retries=3,
    retry_count=0
    )

    feature_engineered_df_loc=state['feature_engineered_df_loc']
    feature_engineering_documentation=state['feature_engineering_documentation']

    with open(feature_engineering_documentation, 'w') as md_file:
        feature_engineering_steps = feature_engineering_agent.get_recommended_feature_engineering_steps(markdown=True)
        if hasattr(feature_engineering_steps, 'data'):
            md_file.write(feature_engineering_steps.data)
        else:
            md_file.write(str(feature_engineering_steps))
            print("Warning: Could not extract markdown content properly.")

    feature_engineering_agent.get_data_engineered().to_csv(feature_engineered_df_loc, index=False)
    return {'feature_engineered_data':feature_engineering_agent.get_data_engineered()}

def model_selection_agent_function(state):
    model_selection_agent=ModelSelectionAgent(
        model = state['llm'], 
        log=state['LOG'], 
        log_path=state['LOG_PATH'],
        file_name="model_selection.py", 
        function_name="model_selection",
        mlflow_run_id=state['mlflow_run_id']
    )
    df = state['feature_engineered_data']
    X = df.drop("target", axis=1)
    y = df["target"]

    model_selection_agent.invoke_agent(
        user_prompt="I need a classification model that prioritizes recall.",
        data_features=X,
        data_target=y
    )

    model_selection_documentation=state['model_selection_documentation']
    with open(model_selection_documentation, 'w') as md_file:
        results = model_selection_agent.get_model_selection_results()
        md_file.write(str(results))

    best_model=model_selection_agent.get_model_selection_results()['best_model']
    return {'best_model':best_model}

def model_training_agent_function(state):
    model_training_agent = ModelTrainingAgent(
        model=state['llm'], 
        n_samples=state['n_samples'], 
        log=state['LOG'], 
        log_path=state['LOG_PATH'],
        mlflow_run_id=state['mlflow_run_id']
    )
    model_training_documentation = state['model_training_documentation']

    df = state['feature_engineered_data']

    model_training_agent.invoke_agent(
        user_prompt=f"Train a {state['best_model']} model for this classification task. Optimize for F1 score and use 5-fold cross-validation.",
        data_df=df,
        max_retries=3,
        retry_count=0
    )

    training_results = model_training_agent.get_model_training_results()

    with open(model_training_documentation, 'w') as md_file:
        md_content = "# Model Training Results\n\n"
        
        if 'trained_model' in training_results:
            md_content += "## Trained Model\n\n"
            md_content += f"```\n{training_results['trained_model']}\n```\n\n"
        
        if 'metrics' in training_results and isinstance(training_results['metrics'], dict):
            md_content += "## Performance Metrics\n\n"
            
            md_content += "| Metric | Value |\n"
            md_content += "|--------|-------|\n"
            
            for metric_name, metric_value in training_results['metrics'].items():
                if isinstance(metric_value, float):
                    formatted_value = f"{metric_value:.4f}"
                else:
                    formatted_value = str(metric_value)
                    
                md_content += f"| {metric_name} | {formatted_value} |\n"
            
            md_content += "\n"
        
        md_file.write(md_content)
