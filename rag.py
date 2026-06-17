
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from langchain_core.tools import retriever
from tenacity import retry_unless_exception_type
from langchain_community.chat_models.tongyi import ChatTongyi

from file_history_store import get_history
from vector_stores import VectorStoreService
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name),
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            messages=[
                ('system','以我提供的已知参考资料为主'
                 '简洁专业的回答问题，参考资料{context}'),
                ('system','并提供用户提问记录，如下：'),
                MessagesPlaceholder('history'),
                ('user','请回答用户问题:{input}')
            ]
        )
        self.chat_model=ChatTongyi(model=config.chat_model_name)
        self.chain = self._get_chain()

    def _get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(docs:list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str:str = ""
            for doc in docs:
                formatted_str += f'文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n'
            return formatted_str
        def temp(value:dict)->str:
            return value['input']
        def temp2(value):
            new_value={}
            new_value['input']=  value['input']['input']
            new_value['history']=  value['input']['history']
            new_value['context']=  value['context']
            return new_value
        chain = (
            {
                "input":RunnablePassthrough(),
                "context":RunnableLambda(temp)|retriever|format_document
            }| RunnableLambda(temp2)| self.prompt_template|self.chat_model|StrOutputParser()
        )
        conversation_chain=RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain
if __name__=='__main__':
    #session_id 配置
    session_config={
        'configurable':{
            'session_id':'user_001'
        }
    }
    res=RagService().chain.invoke({'input':"我应该穿什么尺码"},session_config)
    print(res)