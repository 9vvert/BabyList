import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from pydantic import SecretStr
load_dotenv('../../.llm_env')

API_KEY : str= os.getenv("MY_OPENAI_API_KEY") or ""
BASE_URL = os.getenv("MY_OPENAI_API_BASE")

llm = ChatOpenAI(api_key=SecretStr(API_KEY),  base_url=BASE_URL,  model="gpt-4",temperature=0)

## --- 提示词 1：提取信息 ---
prompt_extract = ChatPromptTemplate.from_template(
    "从以下文本中提取技术规格：\n\n{text_input}"
)

## --- 提示词 2：转换为 JSON ---
prompt_transform = ChatPromptTemplate.from_template(
    "将以下规格转换为 JSON 对象，使用 'cpu'、'memory' 和 'storage' 作为键：\n\n{specifications}"
)

## --- 利用 LCEL 构建处理链 ---
## StrOutputParser() 将 LLM 的消息输出转换为简单字符串。
extraction_chain = prompt_extract | llm | StrOutputParser()

## 完整的链将提取链的输出传递到转换提示词的 'specifications' 变量中。
full_chain = (
    {"specifications": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

## --- 运行链 ---
input_text = "新款笔记本电脑型号配备 3.5 GHz 八核处理器、16GB 内存和 1TB NVMe 固态硬盘。"

## 使用输入文本字典执行链。
final_result = full_chain.invoke({"text_input": input_text})

print("\n--- 最终 JSON 输出 ---")
print(final_result)

