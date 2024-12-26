from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import os


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    **Example Complex Queries:**
    
    - Question: What is the total sales revenue after discounts for all Adidas T-shirts in size 'XS'?
      SQL Query: 
      SELECT SUM(price * (1 - `pct_discount` / 100)) 
      FROM `t_shirts` 
      JOIN `discounts` ON `t_shirts`.`t_shirt_id` = `discounts`.`t_shirt_id` 
      WHERE `brand` = 'Adidas' AND `size` = 'XS';
    
    - Question: How many units of each product have been sold in the last month?
      SQL Query: 
      SELECT `product_id`, SUM(`quantity_sold`) 
      FROM `sales` 
      WHERE `sale_date` >= '2024-11-01' 
      GROUP BY `product_id`;
    
    - Question: What is the average price of products sold by each vendor in the last quarter?
      SQL Query: 
      SELECT `vendor_id`, AVG(`price`) 
      FROM `products` 
      WHERE `sale_date` BETWEEN '2024-10-01' AND '2024-12-31' 
      GROUP BY `vendor_id`;
    
    - Question: Which employees have the highest sales in the last year and what is the total amount of sales for each?
      SQL Query: 
      SELECT `employee_id`, SUM(`sales_amount`) 
      FROM `sales` 
      WHERE `sale_date` >= '2023-01-01' 
      GROUP BY `employee_id` 
      ORDER BY SUM(`sales_amount`) DESC 
      LIMIT 5;
    
    - Question: What are the top 5 products with the most units sold last quarter and their sales revenue?
      SQL Query: 
      SELECT `product_id`, SUM(`quantity_sold`) AS `units_sold`, SUM(`quantity_sold` * `price`) AS `revenue` 
      FROM `sales` 
      WHERE `sale_date` BETWEEN '2024-07-01' AND '2024-09-30' 
      GROUP BY `product_id` 
      ORDER BY `units_sold` DESC 
      LIMIT 5;
    
    - Question: What is the total sales revenue for each product category, including those with zero sales in the last month?
      SQL Query:
      SELECT `category_id`, COALESCE(SUM(`sales_amount`), 0) AS `total_sales`
      FROM `categories`
      LEFT JOIN `sales` ON `sales`.`category_id` = `categories`.`category_id`
      WHERE `sale_date` >= '2024-11-01'
      GROUP BY `category_id`;
    
    - Question: List the names of customers who made purchases worth more than $500 in total last year.
      SQL Query:
      SELECT `customer_name`
      FROM `customers`
      JOIN `sales` ON `sales`.`customer_id` = `customers`.`customer_id`
      WHERE `sale_date` >= '2023-01-01'
      GROUP BY `customers`.`customer_id`
      HAVING SUM(`sales_amount`) > 500;
    
    **Your Turn:**
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3, api_key=api_key)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )




# def get_sql_chain(db):
#   template = """
#     You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
#     Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

#     <SCHEMA>{schema}</SCHEMA>
    
#     **Guidelines for Complex Queries:**
#     - If the question involves calculations, such as discounts or percentages, ensure proper mathematical operations with parentheses.
#     - Handle any column names that contain special characters (like underscores) by wrapping them in backticks.
#     - For aggregation functions (like `SUM`, `COUNT`, `AVG`), ensure that the grouping and ordering are correct.
#     - Be sure to include any necessary `JOIN` operations and ensure that table relationships are correctly represented.
#     - Avoid unnecessary complexity and make sure the query is clear and efficient.

#     **Example Complex Queries:**
#     - Question: What is the total sales revenue after discounts for all Adidas T-shirts in size 'XS'?
#       SQL Query: SELECT SUM(price * (1 - `pct_discount` / 100)) 
#                  FROM `t_shirts` 
#                  JOIN `discounts` ON `t_shirts`.`t_shirt_id` = `discounts`.`t_shirt_id` 
#                  WHERE `brand` = 'Adidas' AND `size` = 'XS';

#     - Question: How many units of each product have been sold in the last month?
#       SQL Query: SELECT `product_id`, SUM(`quantity_sold`) 
#                  FROM `sales` 
#                  WHERE `sale_date` >= '2024-11-01' 
#                  GROUP BY `product_id`;

#     **Your Turn:**

#     Question: {question}
#     SQL Query:
#   """

#   prompt = ChatPromptTemplate.from_template(template)

#   llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3, api_key=api_key)
  
#   def get_schema(_):
#     return db.get_table_info()

#   return (
#     RunnablePassthrough.assign(schema=get_schema)
#     | prompt
#     | llm
#     | StrOutputParser()
#   )

    
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>
                                                                                  
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL> 
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.3, api_key=api_key)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })
    
  
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
      AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

load_dotenv()

st.set_page_config(page_title="Chat", page_icon=":speech_balloon:")

st.title("Chat with Database")

with st.sidebar:
    st.subheader("Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="admin", key="Password")
    st.text_input("Database", value="Chinook", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")
    
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

user_query = st.chat_input("Type a message...")
if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.markdown(response)
        
    st.session_state.chat_history.append(AIMessage(content=response))