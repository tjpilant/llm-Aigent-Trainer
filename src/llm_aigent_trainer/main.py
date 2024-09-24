import json
import os
from typing import Dict, Any, List
import time
import re
import argparse
import tempfile

import mlflow
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def initialize_project(project_name: str, model_name: str) -> None:
    """Initialize MLflow project."""
    mlflow.set_experiment(project_name)
    mlflow.start_run()
    mlflow.log_params({
        "base_model": model_name,
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0
    })

def process_markdown(file_path: str) -> List[Dict[str, List[Dict[str, str]]]]:
    """Process a Markdown file and extract sections for training."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into sections based on headers
    sections = re.split(r'\n#+\s', content)
    
    # Remove the first empty section if it exists
    if sections[0].strip() == '':
        sections = sections[1:]

    # Prepare the training data
    training_data = []
    for i in range(0, len(sections) - 1, 2):
        if i + 1 < len(sections):
            training_data.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant trained on Markdown documentation."},
                    {"role": "user", "content": sections[i].strip()},
                    {"role": "assistant", "content": sections[i + 1].strip()}
                ]
            })

    return training_data

def prepare_training_data(file_path: str) -> List[Dict[str, List[Dict[str, str]]]]:
    """Prepare training data from a Markdown file and log to MLflow."""
    training_data = process_markdown(file_path)
    
    # Log the training data to MLflow
    mlflow.log_dict({"training_data": training_data}, "training_data.json")
    
    return training_data

def initiate_fine_tuning(training_data: List[Dict[str, Any]], model_name: str) -> str:
    """Initiate fine-tuning process and log to MLflow."""
    # Create a temporary file to store the training data in JSONL format
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl') as temp_file:
        for item in training_data:
            json.dump({"messages": item["messages"]}, temp_file)
            temp_file.write('\n')
        temp_file_path = temp_file.name

    try:
        print(f"Uploading file: {temp_file_path}")
        with open(temp_file_path, 'rb') as file:
            uploaded_file = client.files.create(
                file=file,
                purpose='fine-tune'
            )
        
        print(f"File uploaded successfully. File ID: {uploaded_file.id}")
        
        job = client.fine_tuning.jobs.create(
            training_file=uploaded_file.id,
            model=model_name
        )
        
        # Log the fine-tuning job details to MLflow
        mlflow.log_param("fine_tuning_job_id", job.id)
        mlflow.log_artifact(temp_file_path, "training_file.jsonl")
        
        return job.id
    except Exception as e:
        print(f"An error occurred during fine-tuning: {str(e)}")
        raise
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

def check_fine_tuning_status(job_id: str) -> str:
    """Check the status of a fine-tuning job and log metrics to MLflow."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    mlflow.log_metrics({
        "trained_tokens": job.trained_tokens,
        "elapsed_time": job.elapsed_time
    })
    mlflow.log_param("fine_tuning_status", job.status)
    return job.status

def use_fine_tuned_model(model: str, input_text: str) -> Dict[str, Any]:
    """Use the fine-tuned model to process input."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant trained on Markdown documentation."},
                {"role": "user", "content": input_text}
            ],
            temperature=mlflow.get_param("temperature"),
            max_tokens=mlflow.get_param("max_tokens"),
            top_p=mlflow.get_param("top_p"),
            frequency_penalty=mlflow.get_param("frequency_penalty"),
            presence_penalty=mlflow.get_param("presence_penalty")
        )
        return {"output": response.choices[0].message.content}
    except Exception as e:
        mlflow.log_param("error", str(e))
        raise

def evaluate_output(input_text: str, output: str) -> float:
    """Evaluate the output based on the input."""
    # Simple evaluation: check if output is not empty and has some relation to input
    score = 1.0 if output and any(word in output.lower() for word in input_text.lower().split()) else 0.0
    return score

def ai_processing_workflow(input_text: str, model: str) -> Dict[str, Any]:
    """Main AI processing workflow."""
    processed_output = use_fine_tuned_model(model, input_text)
    evaluation_score = evaluate_output(input_text, processed_output["output"])
    return {
        'input': input_text,
        'output': processed_output["output"],
        'evaluation_score': evaluation_score
    }

def read_input_file(file_path: str) -> str:
    """Read input text from a file."""
    with open(file_path, 'r') as file:
        return file.read().strip()

def main(project_name: str, training_file: str, input_file: str, model_name: str) -> None:
    """Main function to run the AI processing workflow with fine-tuning."""
    initialize_project(project_name, model_name)

    try:
        # Prepare training data
        training_data = prepare_training_data(training_file)
        
        # Print the first few items of training data for verification
        print("Sample of prepared training data:")
        for item in training_data[:2]:  # Print first 2 items
            print(json.dumps(item, indent=2))
        
        print("Training data preparation completed.")
        
        # Initiate fine-tuning
        job_id = initiate_fine_tuning(training_data, model_name)
        print(f"Fine-tuning job initiated. Job ID: {job_id}")
        
        # Wait for fine-tuning to complete
        while True:
            status = check_fine_tuning_status(job_id)
            print(f"Fine-tuning status: {status}")
            if status == "succeeded":
                break
            elif status in ["failed", "cancelled"]:
                raise Exception(f"Fine-tuning failed with status: {status}")
            time.sleep(60)  # Check every minute
        
        # Get the fine-tuned model name
        job = client.fine_tuning.jobs.retrieve(job_id)
        fine_tuned_model = job.fine_tuned_model
        print(f"Fine-tuned model: {fine_tuned_model}")
        
        # Log the fine-tuned model to MLflow
        mlflow.log_param("fine_tuned_model", fine_tuned_model)
        
        # Process input with fine-tuned model
        input_text = read_input_file(input_file)
        result = ai_processing_workflow(input_text, fine_tuned_model)

        # Log the result to MLflow
        mlflow.log_params({
            'input_text': result['input'],
            'processed_output': result['output'],
            'evaluation_score': result['evaluation_score'],
        })

        print(f"Workflow completed for project: {project_name}")
        print("Check the MLflow dashboard for logged results.")
        print(f"Input: {result['input'][:100]}...")  # Print first 100 characters of input
        print(f"Output: {result['output'][:100]}...")  # Print first 100 characters of output
        print(f"Evaluation Score: {result['evaluation_score']}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        mlflow.log_param("error", str(e))

    finally:
        mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Aigent-Trainer")
    parser.add_argument("project_name", help="Name of the project")
    parser.add_argument("training_file", help="Path to the training data Markdown file")
    parser.add_argument("input_file", help="Path to the input file for processing")
    parser.add_argument("--model", default="gpt-4o-2024-08-06", help="Name of the pre-existing fine-tuned model (optional)")
    
    args = parser.parse_args()
    
    main(args.project_name, args.training_file, args.input_file, args.model)

# To run this script:
# 1. Ensure you have the required packages installed: pip install mlflow openai python-dotenv
# 2. Set up your .env file with your OpenAI API key:
#    OPENAI_API_KEY=your_openai_api_key_here
# 3. Prepare a training data Markdown file and an input text file
# 4. Run the script using: python main.py <project_name> <training_file.md> <input_file> [--model <model_name>]
# 5. Check the output to verify the correct processing of the Markdown file