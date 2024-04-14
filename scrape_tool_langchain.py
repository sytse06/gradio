import gradio as gr
import asyncio
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_extraction_chain
import json
import os
from urllib.parse import urlparse
import logging

#Load credentials
with open('credentials.json', 'r', encoding='utf-8') as f:
    credentials = json.load(f)
os.environ['OPENAI_API_KEY'] = credentials['openai_api_key']

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the LLM for extraction
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# Async scraping function adapted for Gradio
async def async_scrape_and_save(urls, schema, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    urls = [url for url in urls.split('\n') if url]
    loader = AsyncChromiumLoader(urls)
    bs_transformer = BeautifulSoupTransformer()
    html_contents = await loader.load()
    transformed_docs = bs_transformer.transform_documents(html_contents, tags_to_extract=["span"])

    for doc in transformed_docs:
        try:
            extracted_content = create_extraction_chain(schema=schema, llm=llm).run(doc.page_content)
            parsed_url = urlparse(doc.url)
            filename = f"{parsed_url.netloc.replace('.', '_')}.json"
            filepath = os.path.join(output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(extracted_content, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved extracted content to {filepath}")
        except Exception as e:
            logger.error(f"Error processing {doc.url}: {str(e)}")

# Gradio interface function that integrates with the async scraping
async def scrape_and_display_results(urls):
    schema = {
        "properties": {
            "news_article_title": {"type": "string"},
            "news_article_summary": {"type": "string"},
        },
        "required": ["news_article_title", "news_article_summary"],
    }
    # Directly await the asynchronous function
    await async_scrape_and_save(urls, schema)
    return f"Processed URLs. Check the 'downloads' folder for results."

# Set up the Gradio interface
iface = gr.Interface(fn=scrape_and_display_results,
                     inputs=gr.Textbox(label="Enter URLs (one per line)"),
                     outputs=gr.Textbox(label="Processing Results"),
                     title="LLM-based web scrape tool",
                     description="Enter the URLs you want to scrape in the textbox below, one URL per line. The results will be saved in the 'downloads' folder.")

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()