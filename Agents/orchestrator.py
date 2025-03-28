from langchain_openai import ChatOpenAI
import os
import pandas as pd


from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from agent_functions import data_cleaning_agent_function, feature_engineering_agent_function, model_selection_agent_function, model_training_agent_function
import mlflow


OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-2.5-pro-exp-03-25:free"

experiment_name = "ML_Data_Science_Pipeline"
mlflow.set_experiment(experiment_name)
with mlflow.start_run(run_name="full_pipeline_11") as run:
       run_id = run.info.run_id

# Graph state
class State(TypedDict):
    llm: ChatOpenAI
    n_samples: int
    raw_df_loc: str
    processed_df_loc: str
    data_cleaning_documentation: str
    cleaned_data: pd.DataFrame
    LOG: bool
    LOG_PATH: str
    feature_engineered_data: pd.DataFrame
    feature_engineered_df_loc: str
    feature_engineering_documentation: str
    model_selection_documentation: str
    best_model: str
    model_training_documentation: str
    mlflow_run_id: str



workflow = StateGraph(State)

# Add nodes
workflow.add_node("data_cleaning", data_cleaning_agent_function)
workflow.add_node("feature_engineering", feature_engineering_agent_function)
workflow.add_node("model_selection", model_selection_agent_function)
workflow.add_node("model_training", model_training_agent_function)

workflow.add_edge(START, "data_cleaning")
workflow.add_edge("data_cleaning", "feature_engineering")
workflow.add_edge("feature_engineering", "model_selection")
workflow.add_edge("model_selection", "model_training")
workflow.add_edge("model_training", END)

chain = workflow.compile()

# Save the graph visualization to a file instead of displaying it
# graph_image = chain.get_graph().draw_mermaid_png()
# with open("Agents/Documentation/workflow_graph.png", "wb") as f:
#     f.write(graph_image)
# print("Workflow graph saved to 'workflow_graph.png'")

# Uncomment to run the full workflow
state=chain.invoke({"llm":ChatOpenAI(
                        model=OPENROUTER_MODEL,
                        openai_api_key=OPENROUTER_API_KEY,
                        openai_api_base="https://openrouter.ai/api/v1",
                        max_tokens=10000,
                        temperature=0.0
                        ),
"raw_df_loc":"datasets/raw/tested.csv",
'processed_df_loc':"datasets/processed/tested_cleaned.csv",
'data_cleaning_documentation':"Agents/Documentation/data_cleaning_documentation.md",
'LOG':True,
'LOG_PATH':os.path.join(os.getcwd(), "src/data_processing"),
'feature_engineered_df_loc':"datasets/processed/feature_engineered_data.csv",
'feature_engineering_documentation':"Agents/Documentation/feature_engineering_documentation.md",
'model_selection_documentation':"Agents/Documentation/model_selection_documentation.md",
'model_training_documentation':"Agents/Documentation/model_training_documentation.md",
'n_samples':50,
'mlflow_run_id':run_id})

