"""Link metadata extraction."""

from typing import Dict, Optional
import aiohttp
from bs4 import BeautifulSoup
import asyncio

from bufferiq.ml.multimodal.types import LinkMetadata
from bufferiq.ml.multimodal.exceptions import MediaProcessingError


class LinkMetadataExtractor:
    """Extract metadata from web links."""
    
    def __init__(self, timeout: int = 10):
        """
        Initialize metadata extractor.
        
        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
    
    async def extract(self, url: str) -> LinkMetadata:
        """
        Extract metadata from URL.
        
        Args:
            url: URL to extract metadata from
            
        Returns:
            Link metadata
            
        Raises:
            MediaProcessingError: If extraction fails
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract Open Graph tags
            og_tags = self._extract_og_tags(soup)
            
            # Extract Twitter Card tags
            twitter_tags = self._extract_twitter_tags(soup)
            
            # Extract basic metadata
            title = self._extract_title(soup, og_tags, twitter_tags)
            description = self._extract_description(soup, og_tags, twitter_tags)
            image_url = self._extract_image(soup, og_tags, twitter_tags)
            site_name = self._extract_site_name(soup, og_tags)
            
            return LinkMetadata(
                title=title,
                description=description,
                image_url=image_url,
                site_name=site_name,
                url=url,
                og_tags=og_tags,
                twitter_tags=twitter_tags,
            )
            
        except asyncio.TimeoutError:
            raise MediaProcessingError(f"Request timeout for URL: {url}")
        except Exception as e:
            raise MediaProcessingError(f"Metadata extraction failed: {str(e)}")
    
    def _extract_og_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Open Graph tags."""
        og_tags = {}
        
        for tag in soup.find_all('meta', property=lambda x: x and x.startswith('og:')):
            property_name = tag.get('property', '')
            content = tag.get('content', '')
            if property_name and content:
                og_tags[property_name] = content
        
        return og_tags
    
    def _extract_twitter_tags(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract Twitter Card tags."""
        twitter_tags = {}
        
        for tag in soup.find_all('meta', attrs={'name': lambda x: x and x.startswith('twitter:')}):
            name = tag.get('name', '')
            content = tag.get('content', '')
            if name and content:
                twitter_tags[name] = content
        
        return twitter_tags
    
    def _extract_title(
        self,
        soup: BeautifulSoup,
        og_tags: Dict[str, str],
        twitter_tags: Dict[str, str]
    ) -> Optional[str]:
        """Extract title with fallbacks."""
        # Priority: og:title > twitter:title > <title> tag
        if 'og:title' in og_tags:
            return og_tags['og:title']
        if 'twitter:title' in twitter_tags:
            return twitter_tags['twitter:title']
        
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        return None
    
    def _extract_description(
        self,
        soup: BeautifulSoup,
        og_tags: Dict[str, str],
        twitter_tags: Dict[str, str]
    ) -> Optional[str]:
        """Extract description with fallbacks."""
        # Priority: og:description > twitter:description > meta description
        if 'og:description' in og_tags:
            return og_tags['og:description']
        if 'twitter:description' in twitter_tags:
            return twitter_tags['twitter:description']
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            return meta_desc.get('content', '').strip()
        
        return None
    
    def _extract_image(
        self,
        soup: BeautifulSoup,
        og_tags: Dict[str, str],
        twitter_tags: Dict[str, str]
    ) -> Optional[str]:
        """Extract image URL with fallbacks."""
        # Priority: og:image > twitter:image > first <img> tag
        if 'og:image' in og_tags:
            return og_tags['og:image']
        if 'twitter:image' in twitter_tags:
            return twitter_tags['twitter:image']
        
        img_tag = soup.find('img')
        if img_tag:
            return img_tag.get('src', '').strip()
        
        return None
    
    def _extract_site_name(
        self,
        soup: BeautifulSoup,
        og_tags: Dict[str, str]
    ) -> Optional[str]:
        """Extract site name."""
        if 'og:site_name' in og_tags:
            return og_tags['og:site_name']
        
        return None