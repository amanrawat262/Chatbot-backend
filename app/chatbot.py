from pathlib import Path
from langchain_community.chat_models import ChatOpenAI
from sqlalchemy import create_engine
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.schema import HumanMessage
from sqlalchemy.sql import text
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import uuid
import numpy as np
import os
import io
import json
import base64
import requests
from forecasting_model import get_weather_data, direct_forecast_sktime_lgbm_JM, get_forecasted_weather, calculate_total_cost
from train_test import data_prep_test, data_prep_train
import sqlite3


PROJECT_DIR = "/datadisk0/chatbot/tushar/ChatBot"
DB_PATH = os.path.join(PROJECT_DIR, "session_data.db")

base_dir ="/datadisk0/chatbot/tushar/ChatBot/rout"
graph_dir = os.path.join(base_dir, "static", "graphs")
os.makedirs(graph_dir, exist_ok=True)

pg_uri = "postgresql://postgres:Esya%401234@10.159.1.39:5432/aiml_test"
engine=create_engine(pg_uri)
db = SQLDatabase.from_uri(pg_uri, schema="AEML")

API_KEY = "Your Api Key"
model_for_checkQ = "Your Api Key"
model_for_sql = "Your Api Key"
model_for_query_validity = "Your Api Key"
model_for_forecast = "Your Api Key"
model_for_graph = "Your Api Key"
model_for_explanation = "Your Api Key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


# This function will generate SQl Query and execute on Database
def query_and_execute(user_input):
    # URL for the LLaMA 3.3 model (check DeepInfra for the exact model endpoint)
    MODEL_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"

    # Define the request payload
    payload = {
        "model": model_for_sql,  # Change model name if needed
        "messages": [
            {"role": "system",
            "content": """You are an expert SQL query generator. Your job is to create SQL queries for the given User Input.
I am using PostgreSQL, and my schema name is "AEML"."""},
            {"role": "user",
            "content": f"""My schema structure is as follows:
 
Tables and Sample Data:
m_model (model_id: bigint, model_name: text)
(1, 'Skymet_Fore_Weather_3yrs_lb')
(2, 'Skymet_Act_Weather_3yrs_lb')
(9, 'MeteoSource_Act_Weather_3yr_lb')

m_weather_service (sp_id: integer, service_provider_name: text)
(1, 'company1')
(2, 'company2')
 
t_actual_demand (datetime: timestamp without time zone, block: bigint, demand: double precision)
('2021-01-01 00:00:00',96, 884.31)
('2021-01-01 00:15:00',1, 829.94)
 
t_actual_weather (datetime: timestamp without time zone, temp: double precision, humidity: double precision, sp_id: bigint, block: bigint)
('2021-01-01 00:00:00', 26.06, 68.36, 1, 96)
('2021-01-01 00:15:00', 25.93, 69.26, 2, 1)
 
t_forecasted_demand (datetime: timestamp without time zone,block: bigint, forecasted_demand: double precision, model_id: bigint)
('2024-01-01 00:00:00',96, 1157.32, 1)
('2024-01-01 00:15:00',1, 1123.39, 1)
 
t_forecasted_weather (datetime: timestamp without time zone, temp: double precision, humidity: double precision, sp_id: bigint, block: bigint)
('2021-01-01 00:00:00', 25.08, 78.88, 1, 96)
('2021-01-01 00:15:00', 24.9, 78.52, 2, 1)

t_holidays (date: timestamp without time zone, name: text, normal_holiday: bigint, extended_weekend: bigint, special_day: bigint, flag_off: bigint, post_holiday: bigint, day_name: text)
('2021-01-01', "New Year's Day", 1, 0, 0, 1, 1, 'Friday')
('2021-01-15', 'Makar Sankranti', 1, 0, 0, 1, 0, 'Friday')

t_metrics (date: date, mape: double precision, rmse: double precision, model_id: bigint)
('2024-01-01', 2.21, 33.49, 1)
('2024-01-04', 2.39, 33.19, 1)

t_model_ape (datetime: timestamp without time zone, model_id: integer, block: bigint, ape: double precision)
('2021-01-01 00:00:00', 1, 96, 0.12)
('2021-01-01 00:15:00', 2, 1, 2.1)                                  
 
Your main job is to full give SQL query for User Input. Please provide sql query in single line only and no extra text required.
Please always put schema name in double quotes(i.e. "AEML") while creating queries.
Use CTE for better query generation and give data of each model in different column if applicable.
**Context:** Use previous generated sql queries and user inputs for additional context when applicable.

{user_input}

SQL Query:

Do not put limit in your query no matter what.
"""},
        ],
        "temperature": 0.1
    }

    # Send the request
    response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers)
 
    # Print the response
    if response.status_code == 200:
        store = response.json()["choices"][0]["message"]["content"]
        # print(store)
    else:
        print("Error:", response.text)
 
    sql_query = store.strip("```sql\n").strip("\n```").strip()
    if sql_query:
        sql_query = sql_query
    else:
        sql_query = store
 
    with engine.connect() as connection:
        try:
            restricted_keywords = {"DROP", "DELETE", "UPDATE", "ALTER", "TRUNCATE", "INSERT", "CREATE"}
            if any(keyword in sql_query.upper() for keyword in restricted_keywords):
                return "The requested operation is not allowed."
            result = connection.execute(text(sql_query)).fetchall()
            return sql_query
        except Exception as e:
            error_mes = str(e)
            print(error_mes)
            return f"can't create sql query: {str(e)}"

# This functions is to check if user want to generate graph or not   
def check_question(user_input):
        # URL for the LLaMA 3.3 model (check DeepInfra for the exact model endpoint)
    MODEL_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
 
    # Define the request payload
    payload = {
        "model": model_for_checkQ,  # Change model name if needed
        "messages": [
            {"role": "user",
            "content": f"""Try to understand the User Input
User Input: {user_input}
if User Input need trend, image, graph, chart, report, plot or any diagram output give string output "random =0" otherwise, give string output "random =1".
please do not give extra text."""}
        ],
        "temperature": 0.1
    }
    # Send the request
    # print (payload)
    response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers)
 
    # Print the response
    if response.status_code == 200:
        store = response.json()["choices"][0]["message"]["content"]
        # print(store)
        return store
    else:
        print("Error:", response.text)
        return "Error in LLM"
 
# Craete Naural answer of User's Question and data retrived from DB 
def explanation_text(user_input, results):
        # URL for the LLaMA 3.3 model (check DeepInfra for the exact model endpoint)
    MODEL_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
 
    # Define the request payload
    payload = {
        "model": model_for_explanation,  # Change model name if needed
        "messages": [
            {"role": "user",
            "content": f"""You are a Data analyst and giving answer in the natural language based on user input and SQL Query Results.
The following SQL query was executed based on the user's input, and the results have been retrieved.
always provide answer in order of SQL Query Results

Please provide a human-readable answer in **natural language** in one of the following formats:
### **Instructions:**
- Answer based on the order of SQL Query Results.
- Maintain a **concise and structured** response.
- **Output Format:**
  1. **Single Row Result:** Provide a direct **textual explanation** (avoid table format).
  2. **Multiple Rows Result:** Return a JSON-formatted table inside triple backticks (```json).
  3. **Mixed Format (Text + Table):** Start with a brief explanation, followed by JSON-formatted data.
Ensure that the response is clear and concise without any extraneous phrases like "presented in a JSON format" or "for easy conversion into tables in React". Keep the tabular data format consistent do not add any extra words.   
User Input: {user_input}
SQL Query Results: {results}

Ensure that the response is clear and concise without any extraneous phrases like "presented in a JSON format" or "for easy conversion into tables in React". Keep the tabular data format consistent do not add any extra words.                                                                   
Explanation:
"""}
        ],
        "temperature": 0.1
    }
 
    # Send the request
    response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers)
 
    # Print the response
    if response.status_code == 200:
        store = response.json()["choices"][0]["message"]["content"]
        # print(store)
        return store
    else:
        print("Error:", response.text)
        return "Error in LLM"

# This Function is for getting python code to generate graph
def explanation_graph(user_input, results, data_type):
        # URL for the LLaMA 3.3 model (check DeepInfra for the exact model endpoint)
    MODEL_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
 
    # Define the request payload
    payload = {
        "model": model_for_graph,  # Change model name if needed
        "messages": [
                {"role": "system",
                "content": """You are a python graph creator. Your job is to create graph code based on Sample of data provided by user.
Please keep in mind that provided data is sample and you have to give code which can be implemented on whole dataframe df.
Code provided by you will be executed on my total df in my computer so do not create df in your code.
Please provide generic code which can be implemented on my df.
"""},
            {"role": "user",
            "content": f"""Following is User input(requirement from User) and Sample of data(the sample of total data frame df which I have)
Please provide a python code to create graph or chart as per User Input using columns of these sample data.
sample of data which I am providing to you to know columns and typical entries to create graphs. create graph using these columns only.
 
Rules:                                                
sort data by x-axis column of graph.
if legends are there please put them outside of main graph.
Enhance Grid and Legends: Make the grid subtler and add a shadow effect for the legend.
                 
User Input: {user_input}
Sample of data: {results}
datatype of each column = {data_type}
 
change datatype of column if required.
if required split date and time from datetime column for better graph generation.
Always convert x-axis component to string format before generating graph.
Give code which I can implement on my total dataframe df(it is on my computer).
just give python code no other text is required.
"""}
        ],
        "temperature": 0.1
    }
 
    # Send the request
    response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers)
 
    # Print the response
    if response.status_code == 200:
        store = response.json()["choices"][0]["message"]["content"]
        print(store)
        return store
    else:
        print("Error:", response.text)
        return "Error in LLM"
    

def save_session_history(session_id, user_input, sql_query):
    """Save session history while keeping only the last 7 entries."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
       
        # Insert new session data
        cursor.execute("""
        INSERT INTO session_history (session_id, user_input, sql_query)
        VALUES (?, ?, ?)
        """, (session_id, user_input, sql_query))
       
        # Keep only the last 7 records per session
        cursor.execute("""
        DELETE FROM session_history
        WHERE session_id = ? AND rowid NOT IN (
            SELECT rowid FROM session_history
            WHERE session_id = ? ORDER BY rowid DESC LIMIT 7
        )
        """, (session_id, session_id))
       
        conn.commit()
 
def get_session_history(session_id):
    """Retrieve session history in correct chronological order (oldest first)."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT user_input, sql_query FROM session_history
        WHERE session_id = ? ORDER BY rowid ASC LIMIT 7
        """, (session_id,))
        return cursor.fetchall()  # Returns a list of (user_input, sql_query) tuples
 
def build_prompt(user_input, session_id):
    """Construct the prompt using session queries in correct order."""
    previous_chats = "\n".join(
        f"User Input: {q}\nGenerated SQL: {a}" for q, a in get_session_history(session_id)
    )
    prompt = f"""      
Previous Queries:
{previous_chats}
           
User Input: {user_input}      
"""
    return prompt
 
 
def set_session_context(session_id, context):
    """Set session context (past/future) in SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
       
        # Upsert logic (update if exists, insert otherwise)
        cursor.execute("""
        INSERT INTO session_context (session_id, context)
        VALUES (?, ?)
        ON CONFLICT(session_id) DO UPDATE SET context = excluded.context
        """, (session_id, context))
       
        conn.commit()
 
def get_session_context(session_id):
    """Retrieve session context from SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT context FROM session_context WHERE session_id = ?
        """, (session_id,))
        result = cursor.fetchone()
        return result[0] if result else None
 
def handle_user_input(user_input, session_id):
    """Handle user input while maintaining session consistency."""
    print(f"Session ID: {session_id}\n")
 
    # Handle context switching
    if user_input.lower() in ["past", "future"]:
        set_session_context(session_id, user_input.lower())
        return f"Switching to '{user_input.lower()}' context."
 
    # Retrieve session context
    context = get_session_context(session_id)
 
    if context is None:
        return "Please specify 'past' or 'future' to proceed. Type 'past' for historical data or 'future' for forecasting future demand."
 
    elif context == "past":
        return handle_past(user_input, session_id)
 
    elif context == "future":
        return handle_future_user_input(user_input, session_id)
 

def handle_past(user_input, session_id):
    max_retries = 3
    attempt = 0

    question_prompt = check_question(user_input)
    requires_graph = question_prompt == "random =0"
   
    while attempt < max_retries:
        query_prompt = build_prompt(user_input, session_id)
        print(f"\n\n build_promt {query_prompt} \n\n")
        query_text = query_and_execute(query_prompt)
        if query_text == "The requested operation is not allowed.":
            return query_text
       
        sql_query = query_text.strip("```sql\n").strip("\n```\n").strip()
        if sql_query.endswith(";"):
            sql_query = sql_query[:-1]
        limited_query = sql_query + " LIMIT 10;"
        sql_query = sql_query + ";"
       
        print(f"Attempt {attempt+1}: {sql_query}")
        restricted_keywords = {"DROP", "DELETE", "UPDATE", "ALTER", "TRUNCATE"}
        if any(keyword in sql_query.upper() for keyword in restricted_keywords):
            return "The requested operation is not allowed."
       
        try:
            with engine.connect() as connection:
                result = connection.execute(text(sql_query)).fetchall()
                preview_result = connection.execute(text(limited_query)).fetchall()
           
            full_results = [row._asdict() for row in result] if result else "No data found"
            preview_results = [row._asdict() for row in preview_result] if preview_result else "No data found"
            print(pd.DataFrame(full_results))
           
            save_session_history(session_id, user_input, sql_query)
           
            if requires_graph:
                return generate_graph(user_input, preview_results, full_results)
            else:
                return generate_text_explanation(user_input, preview_results, full_results)
       
        except Exception as e:
            error_message = str(e)
            print(f"Query failed with error: {error_message}")
           
            correction_prompt = f"""
            The SQL query for the previous question failed with an error:
            Error: {error_message}
            SQL Query: {sql_query}
            Please correct the query while following the same structure.
            """
            user_input = correction_prompt
            attempt += 1
 
    return f"Failed to understand question after {max_retries} attempts."  
 
def handle_future_user_input(user_input, session_id):
          # URL for the LLaMA 3.3 model (check DeepInfra for the exact model endpoint)
    MODEL_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
    query_prompt = build_prompt(user_input, session_id)
    # Define the request payload
    payload = {
        "model": model_for_forecast,  # Change model name if needed
        "messages": [
            {"role": "user",
            "content": f"""
    Your task is to understand the user input and extract the required parameters to fill the given function:
    if user ask about getting forecasted weather data on specific date then get_forecasted_weather(input_date, sp_id):
    with sp_id=1
    if user want to get demand data then give below function and if user dont provide start time end time temp or humidity then keep pass None
    direct_forecast_sktime_lgbm_JM( df_train= data_prep_train(input_date=pred_date,lb=1,sp_id=1),
                                    df_test = data_prep_test(input_date, sp_id, start_time, end_time, temp_change=0, hum_change=0),
                                    wl=96,
                                    ll=0.7,
                                    model_id=7)
    If the user asks about the cost of demand then use below function
    def calculate_total_cost(input_date, start_time=None, end_time=None, temp_change=0, hum_change=0)
    for calculate_total_cost function input date is in "DD-MM-YYYY" format
    and for direct_forecast_sktime_lgbm_JM and get_forecasted_weather input date is in "YYYY-MM-DD" format
    start_time and end time should be in "HH:MM:SS" format
    keep sp_id=1 always
    Return the function call with correctly extracted values.
    only return function no other text
    Please differentiate correctly between "forecasted weather" and "forecasted demand" and call the right function, it is really important.
    previous chats contains the questions asked by user before User Input. Please use them for additional context when crafting the response.
 
    {query_prompt}
    """}
        ],
        "temperature": 0.1
    }
 
    # Send the request
    response = requests.post(MODEL_ENDPOINT, json=payload, headers=headers)
 
    # Print the response
    if response.status_code == 200:
        store = response.json()["choices"][0]["message"]["content"]
 
    # Remove markdown code block if present
    response_text = re.sub(r"^```python\n|```$", "", store).strip()
    save_session_history(session_id, user_input, "this was future analysis so no sql query generated. in this case only use User Input as context.")
    print(response_text)
    try:
        result = eval(response_text)
        if isinstance(result, pd.DataFrame):
            if 'datetime' in result.columns:
                result['datetime'] = pd.to_datetime(result['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            if 'date' in result.columns:
                result['date'] = pd.to_datetime(result['datetime']).dt.date.astype(str)
            # Convert DataFrame to JSON with ISO date formatting and parse to Python dict
            json_str = result.to_json(orient='records', date_format=None)
            return {"response": json.loads(json_str)}
        else:
            # Directly return the result assuming it's already serializable
            return {"response": result}
    except Exception as e:
        return {"error": str(e)}
def generate_text_explanation(user_input, preview_results, full_results):
    """
    Generates a natural language explanation based on query results.
    """
    if isinstance(full_results, list) and len(full_results) > 15:
        full_results= pd.DataFrame(full_results)
         # Convert the 'datetime' column to datetime type (if applicable)
        if 'datetime' in full_results.columns:
            full_results['datetime'] = pd.to_datetime(full_results['datetime'])
        # Convert the 'date' column to datetime type (if applicable)
        if 'date' in full_results.columns:
            full_results['date'] = pd.to_datetime(full_results['date']).dt.date
        # Convert the 'time' column to datetime type (if applicable)
        if 'time' in full_results.columns:
            full_results['time'] = pd.to_datetime(full_results['time'], format='%H:%M:%S').dt.time
        # Round any float columns to 2 decimal places (including all floating-point types)
        for column in full_results.select_dtypes(include=np.floating).columns:
            full_results[column] = full_results[column].round(2)
        full_results = full_results.astype(str)
        # Convert DataFrame to JSON with ISO date formatting and parse to Python dict
        json_str = full_results.to_json(orient='records', date_format=None)
        return {"response": json.loads(json_str)}
        # return generate_graph(user_input, preview_results, full_results)
    explanation_response = explanation_text(user_input, full_results)
    return explanation_response

def generate_graph(user_input, preview_results, full_results):
    """
    Generates a graph based on query results using Python code returned by the LLM.
    Returns the graph in Base64 format instead of saving it to a static folder.
    """
    # if isinstance(full_results, list) and len(full_results) == 1:
    #     return generate_text_explanation(user_input, preview_results, full_results)
    
    dataframe = pd.DataFrame(full_results)
    data_type = dataframe.dtypes
    graph_prompt = explanation_graph(user_input, preview_results, data_type)
    
    # Extract Python code from response
    code_match = re.search(r"```python\n(.*?)\n```", graph_prompt, re.DOTALL)
    if not code_match:
        return "No valid Python code found."
    
    code_to_execute = code_match.group(1).strip()
    print(code_to_execute)
    
    # Prepare execution environment
    execution_globals, execution_locals = {}, {"df": dataframe}
    
    try:
        exec(code_to_execute, execution_globals, execution_locals)

        # Check if `plt` is in execution_locals and save the image to memory
        if "plt" in execution_locals:
            buf = io.BytesIO()
            execution_locals["plt"].savefig(buf, format="png")
            execution_locals["plt"].close()
            
            # Encode the image in base64
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            
            return {"response": "Graph generated.", "graph": image_base64}
        
        return "Code executed successfully without generating a graph."
    
    except Exception as e:
        return f"Error during code execution of graph: {str(e)}"

