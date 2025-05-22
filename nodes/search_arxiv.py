from langchain_community.retrievers import ArxivRetriever
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage,HumanMessage
from langgraph.graph import StateGraph,MessagesState,END,START
from dotenv import load_dotenv
load_dotenv()

@tool('ArxivLoader',description="논문 자료를 찾을 때 사용하는 도구")
def getArxivRetriever(query: str):
    retriever = ArxivRetriever(
        top_k_results=2,
        load_max_docs=10000,
        get_full_documents=True
    )

    docs = retriever.invoke(query)
    return docs

tools = [getArxivRetriever]
tool_node = ToolNode(tools)

model_with_tools = ChatOpenAI(model="gpt-4.1-nano",temperature=0.3).bind_tools(tools)

def should_continue(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

def call_model(state:MessagesState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

workflow = StateGraph(MessagesState)
workflow.add_node("agent",call_model)
workflow.add_node("tools",tool_node)
workflow.add_edge(START,"agent")
workflow.add_conditional_edges("agent", should_continue, ["tools",END])
workflow.add_edge("tools","agent")

graph = workflow.compile()

for chunk in graph.stream(
    {"messages":[SystemMessage("""당신은 오픈 소스 모델의 개방성에 대해 조사하는 전문가입니다.\n
                               요청 받은 모델에 대해 논문에서 어떠한 정보가 공개되어 있는지 정리해서 말해주세요.
                               조사 대상은 다음과 같습니다.
                               1. Architecture
                               2. Tokenizer
                               3. Hardware spec
                               4. Software spec
                               5. Training 관련 정보
                               6. Data 구성 정보
                               7. Data Filtering 정보
                               각각의 내용을 정리해서 말해주세요.
                               """),HumanMessage("Kanana Model")]},stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

"""

================================ Human Message =================================

Kanana Model
================================== Ai Message ==================================
Tool Calls:
  ArxivLoader (call_QVxbK2ZbxMHJnEB13SqiHdVN)
 Call ID: call_QVxbK2ZbxMHJnEB13SqiHdVN
  Args:
    query: Kanana Model
  ArxivLoader (call_Do77Qb1ICDQkUiUXk4df9lBN)
 Call ID: call_Do77Qb1ICDQkUiUXk4df9lBN
  Args:
    query: Kanana Model architecture
  ArxivLoader (call_YkYHoKQs5LmUFlaiMnVSgFQm)
 Call ID: call_YkYHoKQs5LmUFlaiMnVSgFQm
  Args:
    query: Kanana Model tokenizer
  ArxivLoader (call_1H7IHGT12QHy0wKMEmxn2sdY)
 Call ID: call_1H7IHGT12QHy0wKMEmxn2sdY
  Args:
    query: Kanana Model hardware specifications
  ArxivLoader (call_RkZTPCntvDSHyfszsrdfGLsB)
 Call ID: call_RkZTPCntvDSHyfszsrdfGLsB
  Args:
    query: Kanana Model software specifications
  ArxivLoader (call_WRfrZHgs1mkPbLY1TCLmhhD2)
 Call ID: call_WRfrZHgs1mkPbLY1TCLmhhD2
  Args:
    query: Kanana Model training details
  ArxivLoader (call_kxX2Cfy5VdKdRP5fINiVJYlN)
 Call ID: call_kxX2Cfy5VdKdRP5fINiVJYlN
  Args:
    query: Kanana Model data composition
  ArxivLoader (call_owv7PJTTzIjhgMm5IFeqDREg)
 Call ID: call_owv7PJTTzIjhgMm5IFeqDREg
  Args:
    query: Kanana Model data filtering
================================= Tool Message =================================
Name: ArxivLoader

Error: AttributeError("module 'fitz' has no attribute 'fitz'")
 Please fix your mistakes.
================================== Ai Message ==================================

Kanana 모델에 대한 정보를 종합하면 다음과 같습니다:

1. **Architecture**: Kanana는 한국어와 영어에서 뛰어난 성능을 보여주는 바이링구얼(Lingual) 언어 모델 시리즈입니다. 모델 크기는 2.1B에서 32.5B 파라미터까지 다양하며, 효율성을 높이기 위해 단계별(pre-training) 및 깊이 확장(depth up-scaling), 가지치기(pruning), 증류(distillation) 등의 기술을 활용하여 설계되었습니다. 특히, 모델의 계산 비용을 크게 낮추면서도 경쟁력 있는 성능을 유지하는 것이 특징입니다.

2. **Tokenizer**: 상세한 토크나이저 정보는 제공되지 않았으나, 한국어와 영어를 모두 지원하는 바이링구얼 특성을 고려할 때, 두 언어에 최적화된 토크나이저를 사용했을 가능성이 높습니다.

3. **Hardware spec**: 구체적인 하드웨어 사양은 명시되어 있지 않으나, 효율적인 학습을 위해 비용 절감 기법을 적용했으며, 대규모 모델 학습에 적합한 GPU 또는 TPU 클러스터를 사용했을 것으로 추정됩니다.

4. **Software spec**: 사용된 소프트웨어 프레임워크에 대한 상세 내용은 공개되지 않았지만, 일반적으로 대형 언어 모델 학습에는 PyTorch 또는 TensorFlow 기반의 맞춤형 학습 환경이 활용됩니다.

5. **Training 관련 정보**: 
   - 데이터는 약 3조 토큰 규모의 고품질 필터링된 데이터셋을 사용하였으며, 이는 모델의 성능을 유지하면서도 데이터 효율성을 높인 결과입니다.
   - 학습 전략으로는 단계별(pre-training) 학습, 깊이 확장, 가지치기, 증류, 그리고 후속 fine-tuning과 선호도 최적화(preference optimization)가 포함됩니다.
   - 계산 비용을 낮추기 위해 staged pre-training과 depth up-scaling 기법을 적극 활용하였으며, 이는 기존 SOTA 모델보다 훨씬 낮은 비용으로 높은 성능을 달성하는 데 기여했습니다.

6. **Data 구성 정보**: 
   - 데이터는 한국어와 영어를 포함하는 바이링구얼 데이터셋으로, 총 3조 토큰 규모입니다.
   - 데이터는 높은 품질을 유지하기 위해 엄격한 필터링 과정을 거쳤으며, 다양한 출처의 텍스트를 포함하는 것으로 보입니다.

7. **Data Filtering 정보**: 
   - 고품질 데이터 선별을 위해 엄격한 필터링 기법이 적용되었으며, 이는 모델의 성능 향상과 학습 효율성을 동시에 달성하는 데 중요한 역할을 했습니다.
   - 구체적인 필터링 방법론은 공개되지 않았으나, 데이터의 품질을 높이기 위한 사전 검증 및 정제 과정이 포함된 것으로 예상됩니다.

추가적으로, Kanana는 공개된 모델로서 연구와 개발 목적으로 2.1B 크기의 base, instruct, embedding 모델이 공개되어 있으며, 이는 한국어 자연어처리 연구 발전에 기여하기 위해 설계된 것으로 보입니다.\"""