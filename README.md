Title Multi-Agent Financial Analysis System

Group ID: 13

Team Member: Sethuraman Krishnasamy

This project is a part of the AAI-520 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

Note: Jupyter Python notebook has to be downloaded and opened in Google Colab (due to compatibility issues with GitHub rendering interactive widgets and metadata).

Files:

1. AAI_520_FinalTeam_Project_Sethuraman_Krishnasamy.ipynb - Jupyter Python notebook has to be downloaded and opened in Google Colab (due to compatibility issues with GitHub rendering interactive widgets and metadata).
2. AAI_520_FinalTeam_Project_Sethuraman_Krishnasamy.pdf - PDF version of the Jupyter notebook
3. AAI-520 Final Team Project Reqport-Group13-Sethuraman_Krishnasamy.docx - Project Report
4. ReadNe:


I. Installation

This project is built in Python and requires several packages to run. It is recommended to use a virtual environment.

II. Clone the repository:

git clone [your-repository-url]
cd [your-repository-name]


III. Install the required packages:

1. pip install pandas numpy yfinance newsapi-python transformers torch matplotlib seaborn

(Note: torch installation may vary depending on your system's CUDA capabilities. The CPU version is sufficient for this project.)

2. Add your NewsAPI Key:
Open the NewsAgent class file and replace the placeholder API key with your own from NewsAPI.org.

self.api_key = "YOUR_NEWSAPI_KEY"


3. Run the main script:
Execute the main script from your terminal to run the full analysis pipeline.

python main_script.py


4. Project Introduction and Objective

The main purpose of this project is to develop and implement a Multi-Agent Financial Analysis system designed to reason, act, and deliver intelligent insights on stock investment opportunities. This system orchestrates multiple specialized AI agents, mirroring real-world quantitative research workflows in investment firms.

The goal is to move beyond monolithic scripts by creating a modular, agentic framework where each agent has a specific role (data ingestion, news analysis, quantitative scoring, etc.). The project aims to demonstrate the power of collaborative AI by synthesizing quantitative financial data with qualitative, LLM-driven news sentiment to produce a holistic and explainable investment thesis.

5: Contributor of Project

Sethuraman Krishnasamy (Main author)

6. Technologies used:

Natural Language Processing (NLP): For sentiment analysis, text summarization, and cleaning.

Machine Learning: Utilized pre-trained transformer models for sentiment classification and summarization.

Data Visualization: To present analytical results and system performance metrics in an intuitive format.

Data Manipulation: Extensive use of Pandas for handling and analyzing financial time-series data.

Agentic AI: Implementation of Prompt Chaining, Routing, and Evaluator-Optimizer patterns.


7. Language: Python
   
9. Libraries used:

Pandas & NumPy: For data manipulation and numerical computation.

Hugging Face transformers: For accessing and using pre-trained LLMs.

PyTorch: As the backend for the transformer models.

yfinance & newsapi-python: For financial data and news ingestion.

Matplotlib & Seaborn: For generating static plots and visualizations.


10. Project Overview:

This project implements a multi-agent system where specialized agents collaborate to perform a comprehensive stock analysis. The architecture is as follows:

a) ResearchAgent (Base Class): Provides all agents with core capabilities like logging and persistent memory.

b) OrchestratorAgent / FinancialAnalysisSystem: The central coordinator that manages the workflow.

c) DataAgent: Ingests quantitative data (stock prices, financials, technical indicators like RSI and MACD) from the Yahoo Finance API.

d) NewsAgent: A sophisticated agent that ingests news from NewsAPI.org and performs a multi-stage Prompt Chain:

e) Preprocessing: Cleans text.

f) Classification: Uses the cardiffnlp/twitter-roberta-base-sentiment-latest model with prompt engineering to determine sentiment.

g) Routing: Categorizes news for specialized sub-analyzers.

11. LLM MODEL USED: Uses the facebook/bart-large-cnn model.

12. Evaluation: Scores its own analysis quality for a feedback loop.

13. AnalysisAgent: The synthesizer that receives data from the DataAgent and NewsAgent. It uses a weighted model to calculate an overall score and generate a final investment decision and reasoning.

14. VisualizationEngine: Generates all charts and plots.
    
16. Dataset

The system operates on real-time data fetched from two primary external sources:

Yahoo Finance API: Provides real-time stock prices, historical data, financial statements, and analyst recommendations.

NewsAPI.org: Provides recent news articles for any given stock symbol.

17. Analysis and Modeling

The core hypothesis is that a superior investment thesis can be formed by fusing quantitative metrics with qualitative sentiment analysis. The system explores this by:

Calculating scores for six key factors: Valuation, Profitability, Technicals, Sentiment, Analyst Ratings, and Market Position.

Weighting these scores to produce a final Overall Score.

Applying an Evaluator-Optimizer pattern where the quality of the NewsAgent's analysis (e.g., sentiment confidence) can boost or penalize the final score.

Generating a final, human-readable recommendation (e.g., "STRONG BUY", "HOLD") and a justification string.

Roadblocks and Challenges

18. A significant challenge was overcoming the inconsistency of the NewsAgent's sentiment analysis. Initial analysis showed that a simple keyword-based or generic LLM approach was insufficient. This was addressed by implementing a hybrid model that uses prompt engineering (adding financial context), LLM classification, and post-processing adjustments (applying financial heuristics) to produce more accurate, context-aware sentiment.

19. License

This project is licensed under the MIT License. See the LICENSE file for more details.

