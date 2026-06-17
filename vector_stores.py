from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.tools import retriever
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-chroma"])

from langchain_chroma import Chroma

import config_data as config
from langchain_chroma import Chroma


class VectorStoreService:
    def __init__(self,embedding):
            self.embedding=embedding
            self.vector_store=Chroma(
                collection_name=config.collection_name,
                embedding_function=self.embedding,
                persist_directory=config.persist_directory,
            )
    def get_retriever(self):
            return self.vector_store.as_retriever(search_kwargs={'k':config.similarity_threshold})

if __name__=='__main__':
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever=VectorStoreService(DashScopeEmbeddings(model='text-embedding-v4')).get_retriever()
    res=retriever.invoke('我的体重120斤，尺码推荐')
    print(res)