import re
import os
import json
import time
import string
import requests
import concurrent
import pdfplumber

from tqdm import tqdm
from io import BytesIO
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Optional, Tuple
from nltk.tokenize import sent_tokenize
from requests.exceptions import Timeout
from concurrent.futures import ThreadPoolExecutor

# ----------------------- Custom Headers -----------------------
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    # 'Accept-Language': 'en-US,zh-CN;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)



def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)

def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    Extract the sentence that best matches the snippet and its context from the full text.

    Args:
        full_text (str): The full text extracted from the webpage.
        snippet (str): The snippet to match.
        context_chars (int): Number of characters to include before and after the snippet.

    Returns:
        Tuple[bool, str]: The first element indicates whether extraction was successful, the second element is the extracted context.
    """
    try:
        full_text = full_text[:50000]

        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        # sentences = re.split(r'(?<=[.!?]) +', full_text)  # Split sentences using regex, supporting ., !, ? endings
        sentences = sent_tokenize(full_text)  # Split sentences using nltk's sent_tokenize

        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            # If no matching sentence is found, return the first context_chars*2 characters of the full text
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"

def extract_text_from_url(url, use_jina=False, jina_api_key=None, snippet: Optional[str] = None):
    """
    Extract text from a URL. If a snippet is provided, extract the context related to it.

    Args:
        url (str): URL of a webpage or PDF.
        use_jina (bool): Whether to use Jina for extraction.
        snippet (Optional[str]): The snippet to search for.

    Returns:
        str: Extracted text or context.
    """
    try:
        if use_jina:
            jina_headers = {
                'Authorization': f'Bearer {jina_api_key}',
                'X-Return-Format': 'markdown',
                # 'X-With-Links-Summary': 'true'
            }
            response = requests.get(f'https://r.jina.ai/{url}', headers=jina_headers).text
            # Remove URLs
            pattern = r"\(https?:.*?\)|\[https?:.*?\]"
            text = re.sub(pattern, "", response).replace('---','-').replace('===','=').replace('   ',' ').replace('   ',' ')
        else:
            response = session.get(url, timeout=20)  # Set timeout to 20 seconds
            response.raise_for_status()  # Raise HTTPError if the request failed
            # Determine the content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type:
                # If it's a PDF file, extract PDF text
                return extract_pdf_text(url, 1000)
            # Try using lxml parser, fallback to html.parser if unavailable
            try:
                soup = BeautifulSoup(response.text, 'lxml')
            except Exception:
                print("lxml parser not found or failed, falling back to html.parser")
                soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

        if snippet:
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            # If no snippet is provided, return directly
            return text[:8000]
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_page_content(urls, max_workers=4, use_jina=False, snippets: Optional[dict] = None, jina_api_key=None):
    """
    Concurrently fetch content from multiple URLs.

    Args:
        urls (list): List of URLs to scrape.
        max_workers (int): Maximum number of concurrent threads.
        use_jina (bool): Whether to use Jina for extraction.
        snippets (Optional[dict]): A dictionary mapping URLs to their respective snippets.
        jina_api_key (Optional[str]): Jina API key if using Jina.

    Returns:
        dict: A dictionary mapping URLs to the extracted content or context.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use tqdm to display a progress bar
        futures = {
            executor.submit(extract_text_from_url, url, use_jina, jina_api_key, snippets.get(url) if snippets else None): url
            for url in urls
        }
        for future in tqdm(concurrent.futures.as_completed(futures), desc="Fetching URLs", total=len(urls)):
            url = futures[future]
            try:
                data = future.result()
                results[url] = data
            except Exception as exc:
                results[url] = f"Error fetching {url}: {exc}"
            time.sleep(0.2)  # Simple rate limiting
    return results


def google_web_search(query, api_key, cse_id, endpoint, num_results=10, start=1, timeout=20):
    """
    Perform a search using the Google Custom Search API with a set timeout.

    Args:
        query (str): Search query.
        api_key (str): API key for the Google Custom Search API.
        cse_id (str): Custom Search Engine ID.
        num_results (int): Number of results to return (max 10 per request, use start parameter for more).
        start (int): The index of the first result to return (1-indexed). Use this for pagination.
                     For page 1: start=1, page 2: start=11, page 3: start=21, etc.
        timeout (int or float or tuple): Request timeout in seconds.
                                         Can be a float representing the total timeout,
                                         or a tuple (connect timeout, read timeout).

    Returns:
        dict: JSON response of the search results. Returns empty dict if the request times out or fails.
    """
    
    endpoint = "https://www.googleapis.com/customsearch/v1",
    
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": min(num_results, 10),  # Google API allows max 10 results per request
        "start": start  # Pagination parameter
    }

    try:
        response = requests.get(endpoint, params=params, timeout=timeout)
        response.raise_for_status()  # Raise exception if the request failed
        search_results = response.json()
        return search_results
    except Timeout:
        print(f"Google Custom Search request timed out ({timeout} seconds) for query: {query}")
        return {}  # Or you can choose to raise an exception
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Google Custom Search request: {e}")
        return {}


def google_web_search_multiple_pages(query, api_key, cse_id, num_pages=3, results_per_page=10, timeout=20):
    """
    Perform a Google Custom Search across multiple pages and combine results.

    Args:
        query (str): Search query.
        api_key (str): API key for the Google Custom Search API.
        cse_id (str): Custom Search Engine ID.
        num_pages (int): Number of pages to fetch (each page has up to 10 results).
        results_per_page (int): Number of results per page (max 10).
        timeout (int): Request timeout in seconds.

    Returns:
        dict: Combined JSON response with all results from all pages.
    """
    all_items = []
    total_results = 0
    
    for page in range(num_pages):
        start_index = page * results_per_page + 1  # Google uses 1-indexed start
        print(f"Fetching page {page + 1} (results {start_index} to {start_index + results_per_page - 1})...")
        
        search_results = google_web_search(
            query, api_key, cse_id, 
            num_results=results_per_page, 
            start=start_index, 
            timeout=timeout
        )
        
        if 'items' in search_results:
            all_items.extend(search_results['items'])
            # Update total results from the last page response
            if 'searchInformation' in search_results:
                total_results = int(search_results['searchInformation'].get('totalResults', 0))
        else:
            # No more results available
            print(f"No more results found. Stopped at page {page + 1}.")
            break
        
        # Rate limiting between requests
        if page < num_pages - 1:
            time.sleep(0.5)
    
    # Combine results
    combined_results = {
        'items': all_items,
        'searchInformation': {
            'totalResults': str(total_results),
            'searchTime': 0,
            'formattedTotalResults': str(len(all_items))
        }
    }
    
    return combined_results


def google_search_with_content(query, api_key, cse_id, num_pages=3, extract_content=True, 
                                use_jina=False, jina_api_key=None, timeout=20):
    """
    Perform Google Custom Search across multiple pages and extract content from URLs.
    Returns results in JSON format with full content.

    Args:
        query (str): Search query.
        api_key (str): API key for the Google Custom Search API.
        cse_id (str): Custom Search Engine ID.
        num_pages (int): Number of pages to fetch (each page has up to 10 results).
        extract_content (bool): Whether to extract full content from URLs.
        use_jina (bool): Whether to use Jina for content extraction (better quality).
        jina_api_key (str): Jina API key if using Jina.
        timeout (int): Request timeout in seconds.

    Returns:
        dict: JSON formatted results with extracted content.
    """
    # Get search results from multiple pages
    search_results = google_web_search_multiple_pages(
        query, api_key, cse_id, num_pages=num_pages, timeout=timeout
    )
    
    if 'items' not in search_results or not search_results['items']:
        return {
            'query': query,
            'total_results': 0,
            'results': []
        }
    
    # Extract relevant info
    extracted_info = extract_relevant_info(search_results)
    
    # Extract content from URLs if requested
    if extract_content:
        print(f"\nExtracting content from {len(extracted_info)} URLs...")
        urls = [info['url'] for info in extracted_info]
        snippets_dict = {info['url']: info['snippet'] for info in extracted_info}
        
        # Fetch content concurrently
        url_contents = fetch_page_content(
            urls, 
            max_workers=4, 
            use_jina=use_jina, 
            snippets=snippets_dict,
            jina_api_key=jina_api_key
        )
        
        # Add content to results
        for info in extracted_info:
            url = info['url']
            if url in url_contents:
                content = url_contents[url]
                if content and not content.startswith("Error"):
                    # Try to extract snippet context if we have a snippet
                    if info['snippet']:
                        success, context = extract_snippet_with_context(content, info['snippet'])
                        if success:
                            info['content'] = context
                        else:
                            info['content'] = content[:8000]  # Limit content length
                    else:
                        info['content'] = content[:8000]
                else:
                    info['content'] = content
            else:
                info['content'] = "Failed to fetch content"
    
    # Format as JSON
    json_results = {
        'query': query,
        'total_results': len(extracted_info),
        'search_metadata': {
            'total_available': search_results.get('searchInformation', {}).get('totalResults', '0'),
            'pages_fetched': num_pages
        },
        'results': extracted_info
    }
    
    return json_results


def extract_pdf_text(url, max_length=600):
    """
    Extract text from a PDF.

    Args:
        url (str): URL of the PDF file.
        max_length (int): Maximum length of the extracted text. Default is 600.

    Returns:
        str: Extracted text content or error message.
    """
    try:
        response = session.get(url, timeout=20)  # Set timeout to 20 seconds
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        # Limit the text length
        cleaned_text = ' '.join(full_text.split()[:max_length])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_relevant_info(search_results):
    """
    Extract relevant information from Google search results.

    Args:
        search_results (dict): JSON response from the Google Custom Search API.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    useful_info = []
    
    if 'items' in search_results:
        for id, result in enumerate(search_results['items']):
            # Extract displayLink for site name (domain)
            display_link = result.get('displayLink', '')
            # Extract formattedUrl or link for the URL
            url = result.get('link', result.get('formattedUrl', ''))
            # Extract snippet (description)
            snippet = result.get('snippet', '')
            # Extract title
            title = result.get('title', '')
            # Extract date if available (Google API doesn't always provide this)
            date = result.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time', '')
            if date:
                date = date.split('T')[0] if 'T' in date else date
            
            info = {
                'id': id + 1,  # Increment id for easier subsequent operations
                'title': title,
                'url': url,
                'site_name': display_link,
                'date': date,
                'snippet': snippet,
                # Add context content to the information
                'context': ''  # Reserved field to be filled later
            }
            useful_info.append(info)
    
    return useful_info


# ------------------------------------------------------------

if __name__ == "__main__":
    #------------------ Load Environment Variables -----------------
    env_path = '/Users/shuogudaojin/Course/JHU_601.727_2025Fall/Agentic-Reasoning/.env'
    load_dotenv(env_path)
    
    # Example usage
    # Define the query to search
    query = "Structure of dimethyl fumarate"
    
    # API key and Custom Search Engine ID for Google Custom Search API
    GOOGLE_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    
    if not GOOGLE_API_KEY:
        raise ValueError("Please set the GOOGLE_SEARCH_API_KEY environment variable.")
    if not GOOGLE_CSE_ID:
        raise ValueError("Please set the GOOGLE_CSE_ID environment variable.")
    
    # Perform the search
    print("Performing Google Custom Search...")
    search_results = google_web_search(query, GOOGLE_API_KEY, GOOGLE_CSE_ID)
    
    print("Extracting relevant information from search results...")
    extracted_info = extract_relevant_info(search_results)

    print("Fetching and extracting context for each snippet...")
    for info in tqdm(extracted_info, desc="Processing Snippets"):
        full_text = extract_text_from_url(info['url'], use_jina=False)  # Get full webpage text
        if full_text and not full_text.startswith("Error"):
            success, context = extract_snippet_with_context(full_text, info['snippet'])
            if success:
                info['context'] = context
            else:
                info['context'] = f"Could not extract context. Returning first 8000 chars: {full_text[:8000]}"
        else:
            info['context'] = f"Failed to fetch full text: {full_text}"

    # print("Your Search Query:", query)
    # print("Final extracted information with context:")
    # print(json.dumps(extracted_info, indent=2, ensure_ascii=False))

