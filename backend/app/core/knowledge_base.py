import re
import markdown
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class KnowledgeBaseProcessor:
    def __init__(self, kb_path: str):
        self.kb_path = Path(kb_path)
        self.documents = []
        
    def parse_markdown_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse a single markdown file and extract documents"""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract metadata from YAML front matter
        metadata = self._extract_metadata(content)
        
        # Remove YAML front matter
        content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)
        
        # Convert markdown to HTML
        html = markdown.markdown(content, extensions=['extra'])
        soup = BeautifulSoup(html, 'html.parser')
        
        # Process each section
        for section in soup.find_all(['h2', 'h3']):
            section_title = section.get_text().strip()
            section_content = []
            
            # Get all content until next heading
            for sibling in section.find_next_siblings():
                if sibling.name in ['h2', 'h3']:
                    break
                text = sibling.get_text().strip()
                if text:
                    section_content.append(text)
                    
            if section_content:
                full_content = ' '.join(section_content)
                full_content = self._add_source_citation(full_content, metadata)
                
                doc = {
                    'content': full_content,
                    'title': section_title,
                    'category': metadata.get('category', 'general'),
                    'source': metadata.get('source', ''),
                    'last_updated': metadata.get('last_updated', '')
                }
                documents.append(doc)
                
        return documents
        
    def _extract_metadata(self, content: str) -> Dict:
        """Extract metadata from YAML front matter"""
        metadata = {}
        yaml_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
        
        if yaml_match:
            yaml_content = yaml_match.group(1)
            for line in yaml_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
                    
        return metadata
        
    def _add_source_citation(self, content: str, metadata: Dict) -> str:
        """Add source citation to content if not present"""
        if not re.search(r'\[https?://[^\]]+\]', content):
            source = metadata.get('source', '')
            if source:
                content = f"{content} [{source}]"
        return content
        
    def process_all_files(self) -> List[Dict[str, Any]]:
        """Process all markdown files in knowledge base"""
        logger.info(f"Processing knowledge base from: {self.kb_path}")
        
        for file_path in self.kb_path.glob("**/*.md"):
            logger.info(f"Processing: {file_path.name}")
            try:
                docs = self.parse_markdown_file(file_path)
                self.documents.extend(docs)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                
        logger.info(f"Total documents processed: {len(self.documents)}")
        return self.documents