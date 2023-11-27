from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
load_dotenv()
from langchain.agents import load_tools,initialize_agent,AgentType



def lang_chain_agent():
    llm = OpenAI(temperature=0.5)
    tools = load_tools(['wikipedia','llm-math'],llm=llm)
    agent = initialize_agent(
        tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose= True
    )
    result =agent.run(
        "What is the average age of the dog. multiply it by 3"
    )
    print(result)
def generate(animal_type):
    llm = OpenAI(temperature=0.5)

    prompt_temp = PromptTemplate(
        input_variables=['animal_type'],
        template="Suggest 5 names for my pet {animal_type}"
    )
    name = LLMChain(llm=llm, prompt=prompt_temp)

    return name({'animal_type': animal_type})



lang_chain_agent()