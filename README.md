# LLM-Aigent-Trainer

A project for fine-tuning OpenAI's GPT-4 models, processing text from Markdown files, and tracking experiments with MLflow.

## Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)
- OpenAI API key with access to GPT-4 models

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/llm-Aigent-Trainer.git
   cd llm-Aigent-Trainer
   ```

2. Install Poetry if you haven't already:

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install project dependencies:

   ```bash
   poetry install
   ```

   This command will create a virtual environment and install all required packages, including mlflow, openai, and python-dotenv.

4. Create a `.env` file in the project root and add your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key_here
   ```

5. If you've previously installed the project dependencies or if you encounter any import errors, update them to the latest versions:

   ```bash
   poetry update
   ```

6. Verify that all dependencies are installed correctly:

   ```bash
   poetry run python -c "import mlflow, openai, dotenv; print(f'MLflow: {mlflow.__version__}, OpenAI: {openai.__version__}, python-dotenv: {dotenv.__version__}')"
   ```

   This should print the versions of mlflow, openai, and python-dotenv without any errors.

7. (Optional) If you're using VSCode, you may see warnings about unresolved imports. These warnings don't affect the script's execution. To resolve them:
   - Open the command palette (Ctrl+Shift+P or Cmd+Shift+P)
   - Type "Python: Select Interpreter" and choose the interpreter from your Poetry virtual environment
   - Restart the VSCode window or reload the workspace

## Usage

1. Activate the virtual environment:

   ```bash
   poetry shell
   ```

2. Prepare your data:
   - Create a Markdown file for training data. The script will process the Markdown file and use headers to split the content into input-output pairs for training. Format your Markdown file as follows:

     ```markdown
     # Input 1
     Your first input text here.

     # Output 1
     The desired output for the first input.

     # Input 2
     Your second input text here.

     # Output 2
     The desired output for the second input.
     ```

   - Prepare an input text file with the content you want to process using the fine-tuned model.

3. Run the main script:

   ```bash
   python src/llm_aigent_trainer/main.py <project_name> <training_file.md> <input_file> [--model <model_name>]
   ```

   Replace:
   - `<project_name>` with your desired project name
   - `<training_file.md>` with the path to your Markdown training data file
   - `<input_file>` with the path to your input text file
   - `<model_name>` (optional) with the name of a pre-existing fine-tuned model

   Example:

   ```bash
   python src/llm_aigent_trainer/main.py ptip freeperson031.md test_input.txt
   ```

   In this example:
   - 'ptip' is the project name
   - 'freeperson031.md' is the Markdown file containing the training data
   - 'test_input.txt' is the input file you want to process with the fine-tuned model

4. The script will:
   - If no model is specified or the default "gpt-4o-2024-08-06" is used:
     - Process the Markdown file (freeperson031.md) to create training data
     - Initiate fine-tuning using the processed training data
     - Wait for the fine-tuning process to complete
   - If a pre-existing model is specified:
     - Skip the fine-tuning process and use the specified model
   - Use the fine-tuned or specified model to process your input text (test_input.txt)
   - Log results to MLflow

5. After the fine-tuning process completes, the script will output the name of the fine-tuned model. It will look something like:

   ```plaintext
   Fine-tuned model: ft:gpt-4o-2024-08-06-TIMESTAMP:ORG-NAME
   ```

   Make sure to note down this model name for future use.

6. Check the MLflow dashboard for logged results and visualizations:

   ```bash
   mlflow ui
   ```

   Then open a web browser and go to [http://localhost:5000](http://localhost:5000)

## Accessing the Fine-tuned Model in OpenAI Playground

To use your fine-tuned model in the OpenAI playground:

1. Go to the [OpenAI playground](https://platform.openai.com/playground).
2. In the "Model" dropdown, scroll to the "Fine-tuned models" section.
3. You should see your fine-tuned model listed with the name noted earlier (e.g., ft:gpt-4o-2024-08-06-TIMESTAMP:ORG-NAME).
4. Select this model to use it in the playground.

## Project Structure

- `src/llm_aigent_trainer/main.py`: Main script containing the fine-tuning and AI processing workflow
- `pyproject.toml`: Poetry configuration file with project dependencies
- `.env`: Environment file for storing your OpenAI API key

## Customization

You can customize the AI processing workflow by modifying the functions in `main.py`:

- `process_markdown`: Adjust how the Markdown file is processed for training data
- `prepare_training_data`: Modify the preparation of training data
- `initiate_fine_tuning`: Adjust fine-tuning parameters
- `use_fine_tuned_model`: Modify how the fine-tuned model is used
- `evaluate_output`: Implement more sophisticated evaluation metrics
- `ai_processing_workflow`: Add additional processing steps or logging

## Integration with MLflow

This project uses MLflow for experiment tracking and visualization. The integration includes:

- Logging of OpenAI API calls
- Tracking of fine-tuning progress
- Logging of input texts, processed outputs, and evaluation scores
- Visualization of results in the MLflow dashboard

To learn more about MLflow and its features, visit their [GitHub repository](https://github.com/mlflow/mlflow) and [documentation](https://www.mlflow.org/docs/latest/index.html).

## Model Information

By default, this project uses the GPT-4o-2024-08-06 model for fine-tuning and inference. This is an advanced model from OpenAI's GPT-4 series, which offers improved performance and capabilities compared to earlier versions. Ensure that your OpenAI API key has access to this model before running the script.

You can also use a pre-existing fine-tuned model by specifying its name with the `--model` argument when running the script.

## Troubleshooting

If you encounter import errors or other issues:

1. Ensure you've activated the Poetry virtual environment: `poetry shell`
2. Verify that all dependencies are up to date: `poetry update`
3. If you encounter a "No module named 'mlflow'" or similar error:
   - Make sure you're running the script within the Poetry virtual environment
   - Try reinstalling the specific package: `poetry run pip install mlflow openai python-dotenv`
   - If the issue persists, try removing and reinstalling all dependencies:

     ```bash
     poetry env remove python
     poetry install
     ```

4. If using VSCode:
   - Select the correct Python interpreter:
     - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
     - Type "Python: Select Interpreter"
     - Choose the interpreter from the Poetry virtual environment
   - Rebuild the IntelliSense database:
     - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
     - Type "Python: Restart Language Server"
5. Restart your IDE after installing or updating dependencies
6. Make sure your OpenAI API key has access to the GPT-4o-2024-08-06 model or the model you're trying to use
7. If you're using a pre-existing fine-tuned model, ensure it exists in your OpenAI account and you have the correct permissions to access it
8. If you encounter issues with the training data format, double-check that your Markdown file follows the correct structure with alternating "Input" and "Output" headers

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
