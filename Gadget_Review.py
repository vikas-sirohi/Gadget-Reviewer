from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel

load_dotenv()

GROQ_API_KEY = "<>"
llm = ChatGroq(model = "mixtral-8x7b-32768",
                 temperature = 0.4,
                 max_tokens = 200,
                 max_retries = 2,
                 )

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful technical gadget agent."),
    ("human","Given product is {gadget} you will list the features of it." ),
])

pros_prompt = ChatPromptTemplate([
    ("system", "You a helpful agent, Who is a technical reviewer."),
    ("human", "Given Features : {features}, List the pros of the product."),
])

cons_prompt = ChatPromptTemplate([
    ("system", "You a helpful agent, Who is a technical reviewer."),
    ("human", "Given Features : {features}, List the cons of the product."),
])


Pros_Chain = (
    pros_prompt
    | llm
    | StrOutputParser()
)

Cons_Chain = (
    cons_prompt
    | llm
    | StrOutputParser()
)

def combine_review(pros, cons):
    return f"\nPros:{pros} \n\n Cons:{cons}."


chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableParallel(branches = {"Pros" : Pros_Chain,
                                   "Cons": Cons_Chain})
    | RunnableLambda(lambda x: combine_review(x["branches"]["Pros"], x["branches"]["Cons"]))
) 

while True:
    query = input("\nEnter Your Gadget: ")
    if query == "exit":
        break
    
    result = chain.invoke({"gadget": query})
    print(result)