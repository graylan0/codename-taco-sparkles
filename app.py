import eel
import asyncio
import re
import speech_recognition as sr
import logging
from concurrent.futures import ThreadPoolExecutor
from duckduckgo_search import ddg
from llama_cpp import Llama
from weaviate.embedded import EmbeddedOptions
from os import path
import weaviate
import nltk
import spacy
from collections import Counter

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

client = weaviate.Client(embedded_options=EmbeddedOptions())
bundle_dir = path.abspath(path.dirname(__file__))
model_path = path.join(bundle_dir, 'llama-2-7b-chat.ggmlv3.q8_0.bin')
llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=3900)
recognizer = sr.Recognizer()
mic = sr.Microphone()

eel.init('web')
logging.basicConfig(level=logging.DEBUG)
executor = ThreadPoolExecutor(max_workers=5)

is_listening = False

def is_code_like(chunk):
    code_patterns = r'\b(def|class|import|if|else|for|while|return|function|var|let|const|print)\b|[\{\}\(\)=><\+\-\*/]'
    return bool(re.search(code_patterns, chunk))

def determine_token(chunk, max_words_to_check=100):
    if not chunk:
        return "[attention]"
    if is_code_like(chunk):
        return "[code]"
    words = nltk.word_tokenize(chunk)[:max_words_to_check]
    tagged_words = nltk.pos_tag(words)
    pos_counts = Counter(tag[:2] for _, tag in tagged_words)
    most_common_pos, _ = pos_counts.most_common(1)[0]
    if most_common_pos == 'VB':
        return "[action]"
    elif most_common_pos == 'NN':
        return "[subject]"
    elif most_common_pos in ['JJ', 'RB']:
        return "[description]"
    else:
        return "[general]"

def advanced_semantic_chunk_text(text, max_chunk_size=100):
    logging.debug("Starting advanced semantic chunking.")
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in doc.sents:
        for token in sentence:
            if token.ent_type_ and token.ent_iob_ == 'B':
                entity = " ".join([ent.text for ent in token.ent])
                current_chunk.append(entity)
                current_length += len(entity)
            else:
                current_chunk.append(token.text)
                current_length += len(token)
            if current_length >= max_chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    logging.debug("Completed advanced semantic chunking.")
    return chunks

async def fetch_data_from_duckduckgo(query, max_results=5):
    logging.debug(f"Fetching data from DuckDuckGo for query: {query}")
    try:
        with ddg() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
        logging.debug("Data fetched successfully from DuckDuckGo.")
        return results
    except Exception as e:
        logging.error(f"Error fetching data from DuckDuckGo: {e}")
        return []

def extract_keywords_with_spacy(text):
    logging.debug("Extracting keywords using SpaCy.")
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_stop == False and token.is_punct == False]

async def update_weaviate_with_ddg_data(query):
    logging.debug(f"Updating Weaviate with data for query: {query}")
    ddg_results = await fetch_data_from_duckduckgo(query)
    ddg_text = ' '.join(ddg_results)
    extracted_keywords = extract_keywords_with_spacy(ddg_text)
    for keyword in extracted_keywords:
        idea_entry = {
            "name": keyword,
            "description": ddg_text
        }
        try:
            client.data_object.create(idea_entry, "ideas")
            logging.debug(f"Weaviate updated with keyword: {keyword}")
        except Exception as e:
            logging.error(f"Error updating Weaviate with DDG data: {e}")

async def query_weaviate_for_ideas(keywords):
    logging.debug(f"Querying Weaviate for ideas with keywords: {keywords}")
    ddg_results = await fetch_data_from_duckduckgo(" ".join(keywords))
    ddg_text = ' '.join(ddg_results)
    additional_keywords = extract_keywords_with_spacy(ddg_text)
    all_keywords = list(set(keywords + additional_keywords))
    try:
        query = {
            "operator": "Or",
            "operands": [
                {
                    "path": ["description"],
                    "operator": "Like",
                    "valueString": keyword
                } for keyword in all_keywords
            ]
        }
        results = (
            client.query
            .get('ideas', ['name', 'description'])
            .with_where(query)
            .do()
        )
        logging.debug("Weaviate query executed successfully.")
        return results['data']['Get']['ideas'] if 'data' in results and 'Get' in results['data'] else []
    except Exception as e:
        logging.error(f"Error querying Weaviate: {e}")
        return []

@eel.expose
async def set_speech_recognition_state(state):
    global is_listening
    is_listening = state
    logging.debug(f"Speech recognition state set to: {state}")

async def continuous_speech_recognition():
    global is_listening
    logging.debug("Starting continuous speech recognition.")
    while True:
        if is_listening:
            try:
                audio_data = await asyncio.to_thread(recognizer.listen, mic, timeout=1)
                text = await asyncio.to_thread(audio_to_text, audio_data)
                if text not in ["Could not understand audio", ""]:
                    response = await run_llm(text)
                    eel.update_chat_box(f"User: {text}")
                    eel.update_chat_box(f"Llama: {response}")
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                eel.update_chat_box(f"An error occurred: {e}")
                logging.error(f"Error in continuous speech recognition: {e}")
        else:
            await asyncio.sleep(1)

def audio_to_text(audio_data):
    try:
        text = recognizer.recognize_google(audio_data)
        logging.debug(f"Converted audio to text: {text}")
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        logging.error(f"Request error in speech recognition: {e}")
        return f"Request error: {e}"

async def run_llm(prompt):
    logging.debug(f"Running Llama model with prompt: {prompt}")
    keywords = extract_keywords_with_spacy(prompt)
    await update_weaviate_with_ddg_data(" ".join(keywords))
    recommended_ideas = await query_weaviate_for_ideas(keywords)
    ideas_recommendations = "\n".join([f"- {idea['name']}: {idea['description']}" for idea in recommended_ideas])
    query_prompt = f"User:'{prompt}'. Ideas: {ideas_recommendations}"
    full_prompt = query_prompt
    response = llm(full_prompt, max_tokens=900)['choices'][0]['text']
    logging.debug(f"Llama model response: {response}")
    return response

@eel.expose
def send_message_to_llama(message):
    logging.debug(f"Received message for Llama: {message}")
    response = asyncio.run(run_llm(message))
    return response

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    logging.debug("Starting main application.")
    asyncio.run(continuous_speech_recognition())
    eel.start('index.html')
