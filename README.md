Multi-Agent Financial Analysis System
Group ID: 13
Team Member: Sethuraman Krishnasamy

This project is a part of the AAI-520 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

-- Project Status: Completed

Installation

This project is built in Python and requires several packages to run. It is recommended to use a virtual environment.

Clone the repository:

git clone [your-repository-url]
cd [your-repository-name]


Install the required packages:

pip install pandas numpy yfinance newsapi-python transformers torch matplotlib seaborn


(Note: torch installation may vary depending on your system's CUDA capabilities. The CPU version is sufficient for this project.)

Add your NewsAPI Key:
Open the NewsAgent class file and replace the placeholder API key with your own from NewsAPI.org.

self.api_key = "YOUR_NEWSAPI_KEY"


Run the analysis:
Execute the main script from your terminal to run the full analysis pipeline.

python main_script.py


Project Intro/Objective

The main purpose of this project is to develop and implement a Multi-Agent Financial Analysis system designed to reason, act, and deliver intelligent insights on stock investment opportunities. This system orchestrates multiple specialized AI agents, mirroring real-world quantitative research workflows in investment firms.

The goal is to move beyond monolithic scripts by creating a modular, agentic framework where each agent has a specific role (data ingestion, news analysis, quantitative scoring, etc.). The project aims to demonstrate the power of collaborative AI by synthesizing quantitative financial data with qualitative, LLM-driven news sentiment to produce a holistic and explainable investment thesis.

Partner(s)/Contributor(s)

[Name of Teammate 1]

[Name of Teammate 2]

Methods Used

Natural Language Processing (NLP): For sentiment analysis, text summarization, and cleaning.

Machine Learning: Utilized pre-trained transformer models for sentiment classification and summarization.

Data Visualization: To present analytical results and system performance metrics in an intuitive format.

Data Manipulation: Extensive use of Pandas for handling and analyzing financial time-series data.

Agentic AI Design: Implementation of Prompt Chaining, Routing, and Evaluator-Optimizer patterns.

Technologies

Python

Pandas & NumPy: For data manipulation and numerical computation.

Hugging Face transformers: For accessing and using pre-trained LLMs.

PyTorch: As the backend for the transformer models.

yfinance & newsapi-python: For financial data and news ingestion.

Matplotlib & Seaborn: For generating static plots and visualizations.

Project Description

Project Overview

This project implements a multi-agent system where specialized agents collaborate to perform a comprehensive stock analysis. The architecture is as follows:

ResearchAgent (Base Class): Provides all agents with core capabilities like logging and persistent memory.

OrchestratorAgent / FinancialAnalysisSystem: The central coordinator that manages the workflow.

DataAgent: Ingests quantitative data (stock prices, financials, technical indicators like RSI and MACD) from the Yahoo Finance API.

NewsAgent: A sophisticated agent that ingests news from NewsAPI.org and performs a multi-stage Prompt Chain:

Preprocessing: Cleans text.

Classification: Uses the cardiffnlp/twitter-roberta-base-sentiment-latest model with prompt engineering to determine sentiment.

Routing: Categorizes news for specialized sub-analyzers.

Summarization: Uses the facebook/bart-large-cnn model.

Evaluation: Scores its own analysis quality for a feedback loop.

AnalysisAgent: The synthesizer that receives data from the DataAgent and NewsAgent. It uses a weighted model to calculate an overall score and generate a final investment decision and reasoning.

VisualizationEngine: Generates all charts and plots.

Dataset

The system operates on real-time data fetched from two primary external sources:

Yahoo Finance API: Provides real-time stock prices, historical data, financial statements, and analyst recommendations.

NewsAPI.org: Provides recent news articles for any given stock symbol.

Analysis and Modeling

The core hypothesis is that a superior investment thesis can be formed by fusing quantitative metrics with qualitative sentiment analysis. The system explores this by:

Calculating scores for six key factors: Valuation, Profitability, Technicals, Sentiment, Analyst Ratings, and Market Position.

Weighting these scores to produce a final Overall Score.

Applying an Evaluator-Optimizer pattern where the quality of the NewsAgent's analysis (e.g., sentiment confidence) can boost or penalize the final score.

Generating a final, human-readable recommendation (e.g., "STRONG BUY", "HOLD") and a justification string.

Roadblocks and Challenges

A significant challenge was overcoming the inconsistency of the NewsAgent's sentiment analysis. Initial analysis showed that a simple keyword-based or generic LLM approach was insufficient. This was addressed by implementing a hybrid model that uses prompt engineering (adding financial context), LLM classification, and post-processing adjustments (applying financial heuristics) to produce more accurate, context-aware sentiment.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

