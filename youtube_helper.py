from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
load_dotenv()

embeddings = OpenAIEmbeddings()


def vector_db_youtube(video_url: str)->FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap = 100)
    docs = text.split_documents(transcript)
    
    db = FAISS.from_documents(docs,embedding=embeddings)
    return db

def get_response_from_query(db,query,k=4):
    docs = db.similarity_search(query,k=k)
    docs_page = " ".join([d.page_content for d in docs])
    
    llm = OpenAI(model = "text-davinci-003")
    prompt_temp = PromptTemplate(
        input_variables=['question',"docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed.
        """,
    )
    
    chain = LLMChain(llm=llm, prompt=prompt_temp)
    response = chain.run(question=query,docs = docs_page)
    response = response.replace('\n',"")
    return response

# print(vector_db_youtube("https://www.youtube.com/watch?v=lG7Uxts9SXs")  )  