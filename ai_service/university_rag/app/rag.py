from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import pickle
from dotenv import load_dotenv
import os

from app.config import *

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

from groq import Groq
client = Groq()

# -------------------------
# Load FAISS
# -------------------------
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

vectorstore = FAISS.load_local(
    FAISS_PATH,
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 12,
        "lambda_mult": 0.5
    }
)

# -------------------------
# Load BM25 docs
# -------------------------
with open(DOCS_PATH, "rb") as f:
    docs = pickle.load(f)

bm25_docs = [
    Document(page_content=d.page_content.replace("passage: ", ""), metadata=d.metadata)
    for d in docs
]

bm25 = BM25Retriever.from_documents(bm25_docs)
bm25.k = 12

# -------------------------
# Query Expansion 
# -------------------------


def enhance_query(question):
    prompt = prompt = f"""
أنت مساعد متخصص في إعادة صياغة الأسئلة الأكاديمية الخاصة باللوائح الجامعية.

المطلوب:
1) تحويل السؤال إلى العربية الفصحى
2) إنشاء 5 صيغ بحث قوية

قواعد:
- لا تغيّر المعنى
- لا تضف معلومات
- استخدم لغة أكاديمية (يشترط، اجتياز، متطلب، مقرر)
- إذا كان هناك اسم مادة:
  - احتفظ به كما هو
  - وأنشئ بدائل محتملة له (مثل: Math 2 → Mathematics 2 → Calculus 2)
- اجعل بعض queries قصيرة (keywords)
- اجعل بعض queries تحتوي على كلمات السؤال الأصلية

الإخراج يجب أن يكون بهذا الشكل فقط:

<normalized>
...
</normalized>

<queries>
...
...
...
...
...
...
</queries>
السؤال:
{question}
"""

    completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200,
    temperature=0.2,  
    top_p=0.9
)

    content = completion.choices[0].message.content

    return content    


# -------------------------
# Parse_llm_Output
# -------------------------
import re

def parse_llm_output(text, expected_queries=5):
    text = text.strip()

    normalized_match = re.search(
        r"<normalized>(.*?)</normalized>",
        text,
        re.DOTALL
    )

    if normalized_match:
        normalized = normalized_match.group(1).strip()
    else:
        normalized = text.split("\n")[0].strip()


    queries_match = re.search(
        r"<queries>(.*?)</queries>",
        text,
        re.DOTALL
    )

    if queries_match:
        queries_block = queries_match.group(1).strip()
        raw_queries = queries_block.split("\n")
    else:
        raw_queries = text.split("\n")[1:]


    cleaned_queries = []
    for q in raw_queries:
        q = q.strip()

        q = re.sub(r"^[\-\d\.\)\s]+", "", q)

        if q:
            cleaned_queries.append(q)

    cleaned_queries = list(dict.fromkeys(cleaned_queries))


    if len(cleaned_queries) < expected_queries:
        cleaned_queries.append(normalized)

    cleaned_queries = cleaned_queries[:expected_queries]


    final_queries = [normalized] + cleaned_queries

    return normalized, final_queries


# -------------------------
# Get_Queries
# -------------------------
def get_queries(question):
    raw_output = enhance_query(question)
    normalized_q, queries = parse_llm_output(raw_output)
    return normalized_q, queries



# -------------------------
# Hybrid
# -------------------------
def hybrid_retrieve(query, k=7, w_faiss=0.7, w_bm25=0.3):

    query_e5 = "query: " + query

    faiss_docs = retriever.invoke(query_e5)
    bm25_docs = bm25.invoke(query)

    scored_dict = {}


    for rank, doc in enumerate(faiss_docs):
        key = doc.page_content  # already "passage: ..."

        score = w_faiss * (1 / (rank + 1))

        if key not in scored_dict:
            scored_dict[key] = (score, doc)
        else:
            scored_dict[key] = (scored_dict[key][0] + score, doc)


    for rank, doc in enumerate(bm25_docs):
        key = "passage: " + doc.page_content  # عشان يتوافق مع FAISS

        score = w_bm25 * (1 / (rank + 1))

        if key in scored_dict:
            scored_dict[key] = (
                scored_dict[key][0] + score,
                scored_dict[key][1]
            )
        else:
            scored_dict[key] = (
                score,
                Document(page_content=key, metadata=doc.metadata)
            )

    sorted_docs = sorted(
        scored_dict.values(),
        key=lambda x: x[0],
        reverse=True
    )

    return [doc for score, doc in sorted_docs[:k]]



# -------------------------
# Multi_query_retrieve
# -------------------------
def multi_query_retrieve(question, k=10):

    normalized_q, queries = get_queries(question)


    queries = [normalized_q] + queries

    all_docs = []

    for q in queries[:6]:  
        docs = hybrid_retrieve(q, k=5)
        all_docs.extend(docs)

    # remove duplicates
    unique = {}
    for doc in all_docs:
        key = doc.page_content[:200]
        unique[key] = doc

    return list(unique.values())[:k], normalized_q



# -------------------------
# Reranker
# -------------------------
def simple_rerank(query, docs, top_k=5):

    scored = []
    query_words = query.split()

    for doc in docs:
        score = 0

        text = doc.page_content

        for word in query_words:
            if word in text:
                score += 1

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scored[:top_k]]


# -------------------------
# Build_Context
# -------------------------
def build_context(docs, max_chars=5000):
    context = ""

    for d in docs:
        text = d.page_content.replace("passage: ", "")

        if len(context) + len(text) > max_chars:
            break

        context += text.strip() + "\n\n"

    return context


# -------------------------
# RAG main
# -------------------------
def ask(question):

    docs, normalized_q = multi_query_retrieve(question)

    docs = simple_rerank(normalized_q, docs, top_k=5)

    context = build_context(docs, max_chars=5000)


    prompt = f"""
أنت مساعد أكاديمي متخصص في لائحة كلية الحاسبات والمعلومات بجامعة طنطا.

مهمتك:
الإجابة على السؤال باستخدام المعلومات الموجودة فقط في النصوص المرفقة.

قواعد مهمة جدًا:
1. اعتمد فقط على النصوص — لا تضف أي معلومات خارجية.
2. استخرج الإجابة بشكل مباشر أو قريب جدًا من النص.
3. إذا كانت الإجابة موجودة في أكثر من جزء، قم بدمجها في إجابة واحدة واضحة.
4. إذا كانت المعلومات غير كافية، لا تخمّن.

النصوص:
{context}

السؤال:
{question}

طريقة الإجابة:
- ابدأ بالإجابة مباشرة بدون مقدمات.
- اكتب إجابة واضحة ومباشرة.
- استخدم نقاط إذا لزم الأمر.
- لا تذكر أنك تعتمد على "النصوص" أو "السياق".

إذا لم تجد إجابة:
اكتب فقط:
"لا أستطيع إيجاد إجابة واضحة من اللائحة بناءً على النصوص المتاحة".
"""

    completion = client.chat.completions.create(
        model=LLM_MODEL,  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,  
        top_p=0.9
    )

    return completion.choices[0].message.content