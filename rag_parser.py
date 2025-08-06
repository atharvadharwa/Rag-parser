#!/usr/bin/env python3
"""
RAG Parser using Unstructured Library

This script parses multiple file types and converts them to markdown format
for LLM understanding, with special handling for tables that may be split
across pages.
"""

import os
from typing import List, Union, Dict, Any
from ragflow.parser import RAGFlowParser
from ragflow.table_recognizer import TableStructureRecognizer
import pandas as pd
from pathlib import Path
import json
import html


class RAGParser:
    """
    A parser for RAG systems using RAGFlow with OCR, TSR, and DLR capabilities.
    Converts documents to LLM-consumable format with HTML tables.
    """
    
    def __init__(self):
        """Initialize the RAGFlow parser with supported file types."""
        self.parser = RAGFlowParser(enable_ocr=True, enable_tsr=True, enable_dlr=True)
        self.table_recognizer = TableStructureRecognizer()
        self.supported_types = {
            '.pdf': self._parse_generic,
            '.docx': self._parse_generic,
            '.pptx': self._parse_generic,
            '.html': self._parse_generic,
            '.htm': self._parse_generic,
            '.txt': self._parse_generic,
            '.md': self._parse_generic,
            '.jpg': self._parse_generic,
            '.jpeg': self._parse_generic,
            '.png': self._parse_generic,
        }
    
    def parse_file(self, file_path: Union[str, Path]) -> str:
        """
        Parse a file and convert it to markdown format.
        
        Args:
            file_path: Path to the file to parse
            
        Returns:
            str: Markdown formatted content
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_types:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Parse the file using the appropriate method
        elements = self.supported_types[file_extension](file_path)
        
        # Convert to LLM-consumable format
        llm_content = self._convert_to_llm_format(elements)
        
        return llm_content
    
    def _parse_generic(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse file using RAGFlow's advanced parsing capabilities.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict: Parsed content with text and tables
        """
        return self.parser.parse(str(file_path))
    
    
    def _convert_to_llm_format(self, parsed_content: Dict[str, Any]) -> str:
        """
        Convert parsed content to LLM-consumable format with HTML tables.
        
        Args:
            parsed_content: Parsed content from RAGFlow
            
        Returns:
            str: Formatted content for LLM consumption
        """
        output_lines = []
        
        # Process text content
        if 'text' in parsed_content:
            output_lines.append(parsed_content['text'])
        
        # Process tables
        if 'tables' in parsed_content:
            for table in parsed_content['tables']:
                table_html = self.table_recognizer.construct_table(table)
                output_lines.append(f"\n<!-- TABLE START -->\n{table_html}\n<!-- TABLE END -->\n")
        
        # Process images (if any)
        if 'images' in parsed_content:
            for image in parsed_content['images']:
                output_lines.append(f"\n<!-- IMAGE: {image['description']} -->\n")
        
        return "\n".join(output_lines)
    


def main():
    """Main function to demonstrate the parser usage."""
    parser = RAGParser()
    
    # Example usage
    sample_dir = Path("sample")
    parent_dir = sample_dir.parent
    
    if sample_dir.exists():
        for file_path in sample_dir.iterdir():
            # Skip hidden files
            if file_path.name.startswith('.'):
                continue
            try:
                print(f"Parsing {file_path.name}...")
                # Parse file using RAGFlow
                parsed_content = parser.parse_file(file_path)
                # Convert to LLM format
                llm_content = parser._convert_to_llm_format(parsed_content)
                
                # Save to parent folder
                output_path = parent_dir / f"{file_path.stem}_parsed.txt"
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(llm_content)
                print(f"Saved parsed content to {output_path}")
            except Exception as e:
                print(f"Error parsing {file_path.name}: {e}")

if __name__ == "__main__":
    main()