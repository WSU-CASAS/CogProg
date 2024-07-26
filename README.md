# CogProg: The Cognitive Prognosticator
Data and code associated with paper: Sprint, Schmitter-Edgecombe, Weaver, Wiese, and Cook. "CogProg: Utilizing Large Language Models to Forecast In-the-moment Health Assessment", 2024.

## Data
* Prompt/response pairs
    * `daily_ema_base_context`: Daily EMA prompts with base context
    * `daily_ema_timeofday_context`: Daily EMA prompts with extra time of day context
    * Note: do to participant privacy concerns, we are unable to release the journal EMA prompt/response pairs
* Numeric data
    * `daily_ema_numeric`: Daily EMA responses in numeric format, parsed by `numeric_methods.py` into format needed by `neural forecast` library

## Environment
* Anaconda w/Python 3.11.4
* Server w/4 80Gb H100s

## Experiments
* Smaller model fine tuning
    * Adapted PromptCast code from: https://github.com/HaoUNSW/PISA
* Llama-based methods
    * `pip install -r llama_requirements.txt`
        * Uses Hugging Face `transformer` library
    * Fine tuning
        * `llama_finetune.py`
    * Iter-CoT adaption
        * `llama_itercot.py`
        * Adapted algorithm from: https://arxiv.org/abs/2304.11657
* Numeric methods
    * `pip install -r numeric_requirements.txt`
        * Uses `neural forecast`, `sci-kit learn`, and `xgboost` libraries
    * `python numeric_methods.py daily_ema_numeric daily_ema_numeric_results`