#!/bin/bash

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null
then
    echo "Poetry is not installed. Please install it first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
poetry install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    echo "WANDB_API_KEY=your_wandb_api_key_here" >> .env
    echo "Please edit the .env file and add your OpenAI and Weights & Biases API keys."
else
    echo "Updating .env file..."
    if ! grep -q "WANDB_API_KEY" .env; then
        echo "WANDB_API_KEY=your_wandb_api_key_here" >> .env
        echo "Please add your Weights & Biases API key to the .env file."
    fi
fi

echo "Setup complete!"
echo "To run the project:"
echo "1. Edit the .env file and add your OpenAI and Weights & Biases API keys."
echo "2. Activate the virtual environment: poetry shell"
echo "3. Run the main script: python src/llm_aigent_trainer/main.py"
echo "4. Check the Weave and Weights & Biases dashboards for logged results."