import asyncio
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import ChatOpenAI
from langchain.chains import create_extraction_chain
import json
import os
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the LLM for extraction
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

async def scrape_and_save(urls, schema, output_dir="downloads"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the loader and transformer
    loader = AsyncChromiumLoader(urls)
    bs_transformer = BeautifulSoupTransformer()

    # Load the HTML content
    html_contents = await loader.load()
    
    # Transform the HTML to extract relevant tags (e.g., <span> for news articles)
    transformed_docs = bs_transformer.transform_documents(html_contents, tags_to_extract=["span"])

    # Extract content based on the provided schema
    for doc in transformed_docs:
        try:
            extracted_content = create_extraction_chain(schema=schema, llm=llm).run(doc.page_content)
            
            # Generate the filename based on the URL
            parsed_url = urlparse(doc.url)
            filename = f"{parsed_url.netloc.replace('.', '_')}.json"
            filepath = os.path.join(output_dir, filename)
            
            # Save the extracted content to a file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(extracted_content, f, ensure_ascii=False, indent=4)
                
            logger.info(f"Saved extracted content to {filepath}")
        except Exception as e:
            logger.error(f"Error processing {doc.url}: {str(e)}")
            # Save error information
            filepath = os.path.join(output_dir, f"error_{parsed_url.netloc.replace('.', '_')}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({'url': doc.url, 'error': str(e)}, f, ensure_ascii=False, indent=4)

# Example usage
schema = {
    "properties": {
        "news_article_title": {"type": "string"},
        "news_article_summary": {"type": "string"},
    },
    "required": ["news_article_title", "news_article_summary"],
}

urls = ["https://www.example.com"]

# Run the async function
asyncio.run(scrape_and_save(urls, schema))