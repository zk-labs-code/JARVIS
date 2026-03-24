"""Web and internet module for JARVIS.

Provides web searching, content fetching, and internet-based learning
capabilities using DuckDuckGo and web scraping.
"""

import logging
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("jarvis.actions.web")


class WebAssistant:
    """Handles internet searches, web content fetching, and online learning."""

    def __init__(
        self,
        timeout: int = 15,
        max_results: int = 5,
        user_agent: str = "JARVIS-AI-Assistant/1.0",
    ):
        """Initialize the web assistant.

        Args:
            timeout: Request timeout in seconds.
            max_results: Maximum search results to return.
            user_agent: User-Agent header for requests.
        """
        self.timeout = timeout
        self.max_results = max_results
        self.headers = {"User-Agent": user_agent}

    def search(self, query: str) -> str:
        """Search the web using DuckDuckGo.

        Args:
            query: Search query string.

        Returns:
            Formatted search results.
        """
        logger.info(f"Web search: {query}")

        # Try DuckDuckGo search
        results = self._search_duckduckgo(query)
        if results:
            return self._format_search_results(query, results)

        # Fallback: scrape DuckDuckGo HTML
        results = self._search_duckduckgo_html(query)
        if results:
            return self._format_search_results(query, results)

        return f"No results found for: {query}"

    def _search_duckduckgo(self, query: str) -> list[dict[str, str]]:
        """Search using duckduckgo-search library.

        Args:
            query: Search query.

        Returns:
            List of result dictionaries.
        """
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", r.get("link", "")),
                        "snippet": r.get("body", r.get("snippet", "")),
                    }
                    for r in results
                ]
        except ImportError:
            logger.debug("duckduckgo-search not installed, using HTML fallback")
            return []
        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")
            return []

    def _search_duckduckgo_html(self, query: str) -> list[dict[str, str]]:
        """Fallback search by scraping DuckDuckGo HTML.

        Args:
            query: Search query.

        Returns:
            List of result dictionaries.
        """
        try:
            url = "https://html.duckduckgo.com/html/"
            response = requests.post(
                url,
                data={"q": query},
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for result in soup.select(".result"):
                title_el = result.select_one(".result__title")
                snippet_el = result.select_one(".result__snippet")
                link_el = result.select_one(".result__url")

                if title_el:
                    results.append({
                        "title": title_el.get_text(strip=True),
                        "url": link_el.get_text(strip=True) if link_el else "",
                        "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                    })

                if len(results) >= self.max_results:
                    break

            return results

        except Exception as e:
            logger.warning(f"DuckDuckGo HTML search error: {e}")
            return []

    def _format_search_results(
        self, query: str, results: list[dict[str, str]]
    ) -> str:
        """Format search results into a readable string.

        Args:
            query: Original search query.
            results: List of result dictionaries.

        Returns:
            Formatted results string.
        """
        if not results:
            return f"No results found for: {query}"

        output = f"Search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r['title']}\n"
            if r.get("url"):
                output += f"   URL: {r['url']}\n"
            if r.get("snippet"):
                output += f"   {r['snippet']}\n"
            output += "\n"

        return output.strip()

    def fetch_page(self, url: str, extract_text: bool = True) -> str:
        """Fetch and parse a web page.

        Args:
            url: URL to fetch.
            extract_text: If True, extract text content. Otherwise return raw HTML.

        Returns:
            Page content string.
        """
        logger.info(f"Fetching page: {url}")

        try:
            response = requests.get(
                url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()

            if not extract_text:
                return response.text

            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            # Truncate if too long
            if len(clean_text) > 5000:
                clean_text = clean_text[:5000] + "\n\n[Content truncated...]"

            return clean_text

        except requests.Timeout:
            return f"Request timed out for: {url}"
        except requests.RequestException as e:
            return f"Failed to fetch {url}: {str(e)}"

    def learn_from_web(self, topic: str) -> dict[str, Any]:
        """Search and learn about a topic from the web.

        Args:
            topic: Topic to learn about.

        Returns:
            Dictionary with learning results.
        """
        logger.info(f"Learning about: {topic}")

        # Search for the topic
        results = self._search_duckduckgo(topic)
        if not results:
            results = self._search_duckduckgo_html(topic)

        if not results:
            return {
                "topic": topic,
                "success": False,
                "content": f"Could not find information about: {topic}",
            }

        # Fetch content from top results
        learned_content = []
        for result in results[:3]:
            if result.get("url"):
                content = self.fetch_page(result["url"])
                if content and not content.startswith("Failed"):
                    learned_content.append({
                        "title": result.get("title", ""),
                        "url": result["url"],
                        "content": content[:2000],  # Limit per source
                    })

        # Compile summary
        summary = f"Learned about '{topic}':\n\n"
        for i, item in enumerate(learned_content, 1):
            summary += f"Source {i}: {item['title']}\n"
            summary += f"URL: {item['url']}\n"
            summary += f"Content: {item['content'][:500]}...\n\n"

        if not learned_content:
            # Use search snippets as fallback
            summary = f"Brief information about '{topic}':\n\n"
            for r in results:
                summary += f"- {r.get('snippet', '')}\n"

        return {
            "topic": topic,
            "success": True,
            "content": summary,
            "sources": [r.get("url", "") for r in results if r.get("url")],
        }

    def download_file(self, url: str, save_path: str) -> str:
        """Download a file from URL.

        Args:
            url: URL to download from.
            save_path: Local path to save the file.

        Returns:
            Result message.
        """
        try:
            response = requests.get(
                url, headers=self.headers, timeout=self.timeout, stream=True
            )
            response.raise_for_status()

            from pathlib import Path
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded: {url} -> {save_path}")
            return f"Downloaded to: {save_path}"

        except Exception as e:
            return f"Download failed: {str(e)}"
