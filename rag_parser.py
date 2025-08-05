#!/usr/bin/env python3
"""
RAG Parser using Unstructured Library

This script parses multiple file types and converts them to markdown format
for LLM understanding, with special handling for tables that may be split
across pages.
"""

import os
from typing import List, Union
from unstructured.partition.auto import partition
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.staging.base import elements_to_json
import pandas as pd
from pathlib import Path
import json


class RAGParser:
    """
    A parser for RAG systems that converts various document formats to markdown
    with special handling for tables that span multiple pages.
    """
    
    def __init__(self):
        """Initialize the parser with supported file types."""
        self.supported_types = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.pptx': self._parse_pptx,
            '.html': self._parse_html,
            '.htm': self._parse_html,
            '.txt': self._parse_text,
            '.md': self._parse_text,
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
        
        # Convert elements to markdown with table merging
        markdown_content = self._elements_to_markdown(elements)
        
        return markdown_content
    
    def _parse_pdf(self, file_path: Path) -> List:
        """
        Parse PDF file with table extraction.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List: List of unstructured elements
        """
        # Use fast strategy to avoid downloading models
        elements = partition_pdf(
            filename=str(file_path),
            strategy="fast",
            infer_table_structure=True,
            extract_images_in_pdf=False
        )
        return elements
    
    def _parse_docx(self, file_path: Path) -> List:
        """
        Parse DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List: List of unstructured elements
        """
        elements = partition_docx(filename=str(file_path))
        return elements
    
    def _parse_pptx(self, file_path: Path) -> List:
        """
        Parse PPTX file.
        
        Args:
            file_path: Path to the PPTX file
            
        Returns:
            List: List of unstructured elements
        """
        elements = partition_pptx(filename=str(file_path))
        return elements
    
    def _parse_html(self, file_path: Path) -> List:
        """
        Parse HTML file.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            List: List of unstructured elements
        """
        elements = partition_html(filename=str(file_path))
        return elements
    
    def _parse_text(self, file_path: Path) -> List:
        """
        Parse text-based files (txt, md).
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List: List of unstructured elements
        """
        elements = partition_text(filename=str(file_path))
        return elements
    
    def _elements_to_markdown(self, elements: List) -> str:
        """
        Convert unstructured elements to markdown with table merging.
        
        Args:
            elements: List of unstructured elements
            
        Returns:
            str: Markdown formatted content
        """
        markdown_lines = []
        i = 0
        
        while i < len(elements):
            element = elements[i]
            element_type = element.to_dict()['type']
            
            if element_type == 'Table':
                # Handle table merging across pages
                table_content = self._process_table(elements, i)
                if table_content:
                    markdown_lines.append(table_content)
                    # Skip processed table elements
                    i += self._count_merged_table_elements(elements, i)
                else:
                    i += 1
            else:
                # Handle other element types
                text_content = str(element)
                if element_type == 'Title':
                    markdown_lines.append(f"# {text_content}")
                elif element_type == 'Header':
                    markdown_lines.append(f"## {text_content}")
                elif element_type == 'NarrativeText':
                    markdown_lines.append(text_content)
                elif element_type == 'ListItem':
                    markdown_lines.append(f"- {text_content}")
                else:
                    markdown_lines.append(text_content)
                i += 1
        
        return "\n\n".join(markdown_lines)
    
    def _process_table(self, elements: List, start_index: int) -> str:
        """
        Process table elements and merge tables that span multiple pages.
        
        Args:
            elements: List of unstructured elements
            start_index: Index where the table starts
            
        Returns:
            str: Markdown formatted table or empty string if processing fails
        """
        table_elements = []
        i = start_index
        
        # Collect consecutive table elements that might be parts of the same table
        while i < len(elements) and elements[i].to_dict()['type'] == 'Table':
            table_elements.append(elements[i])
            i += 1
        
        if not table_elements:
            return ""
        
        # Debug output
        print(f"Found {len(table_elements)} consecutive table elements")
        
        # Try to merge tables with similar structures
        try:
            if len(table_elements) == 1:
                # Single table, just convert to markdown
                print("Processing single table")
                result = self._table_to_markdown(table_elements[0])
                print(f"Single table result: {result[:100]}...")
                return result
            else:
                # Multiple tables, attempt to merge them
                print("Processing multiple tables")
                merged_table = self._merge_tables(table_elements)
                if merged_table:
                    print(f"Merged table result: {merged_table[:100]}...")
                    return merged_table
                else:
                    # If merging failed, try to concatenate with page indicators
                    print("Merging failed, concatenating tables")
                    markdown_tables = []
                    for j, table_element in enumerate(table_elements):
                        table_md = self._table_to_markdown(table_element)
                        if j > 0:
                            table_md = f"\n\n<!-- Continuation from previous page -->\n{table_md}"
                        markdown_tables.append(table_md)
                    result = "\n\n".join(markdown_tables)
                    print(f"Concatenated table result: {result[:100]}...")
                    return result
        except Exception as e:
            print(f"Exception in table processing: {e}")
            # Fallback: return concatenated tables with notes
            try:
                markdown_table = self._table_to_markdown(table_elements[0])
                if len(table_elements) > 1:
                    markdown_table += f"\n\n<!-- Note: This table continues on subsequent pages -->\n"
                    # Add content from subsequent table elements
                    for j in range(1, len(table_elements)):
                        markdown_table += f"<!-- Page {j+1} -->\n"
                        markdown_table += self._table_to_markdown(table_elements[j])
                        if j < len(table_elements) - 1:
                            markdown_table += "\n"
                print(f"Fallback table result: {markdown_table[:100]}...")
                return markdown_table
            except Exception as e2:
                print(f"Exception in fallback table processing: {e2}")
                # Last resort: return raw table text
                result = "\n\n".join([str(table) for table in table_elements])
                print(f"Raw table result: {result[:100]}...")
                return result
    
    def _merge_tables(self, table_elements: List) -> str:
        """
        Merge multiple table elements that are parts of the same table.
        
        Args:
            table_elements: List of table elements to merge
            
        Returns:
            str: Merged markdown table
        """
        if not table_elements:
            return ""
        
        if len(table_elements) == 1:
            return self._table_to_markdown(table_elements[0])
        
        # Extract table data from each element
        all_table_data = []
        for table_element in table_elements:
            try:
                # Try to get table data from metadata
                metadata = table_element.to_dict().get('metadata', {})
                table_data = metadata.get('table_data', None)
                if table_data:
                    all_table_data.append(table_data)
                else:
                    # Try to parse from text_as_html
                    text_as_html = metadata.get('text_as_html', None)
                    if text_as_html:
                        # Parse HTML table using BeautifulSoup
                        rows = self._parse_html_table_rows(text_as_html)
                        if rows:
                            all_table_data.append(rows)
                    else:
                        # Try to parse from text representation
                        text = str(table_element)
                        if text:
                            # Try to convert text table to structured data
                            rows = self._parse_text_table(text)
                            if rows:
                                all_table_data.append(rows)
            except Exception as e:
                continue
        
        # If we have structured table data, merge them
        if all_table_data:
            try:
                merged_data = self._merge_table_data(all_table_data)
                return self._data_to_markdown_table(merged_data)
            except Exception:
                # If merging fails, fall back to concatenating tables
                pass
        
        # Fallback: concatenate table representations
        markdown_tables = []
        for table_element in table_elements:
            markdown_tables.append(self._table_to_markdown(table_element))
        return "\n\n".join(markdown_tables)
    
    def _parse_html_table_rows(self, html_table: str) -> List[List[str]]:
        """
        Parse HTML table into rows of cells.
        
        Args:
            html_table: HTML table string
            
        Returns:
            List[List[str]]: Table data as list of rows
        """
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')
            if not table:
                return []
            
            rows = []
            for tr in table.find_all('tr'):
                row = []
                for cell in tr.find_all(['td', 'th']):
                    row.append(cell.get_text().strip())
                if row:
                    rows.append(row)
            return rows
        except Exception:
            return []
    
    def _parse_text_table(self, text: str) -> List[List[str]]:
        """
        Parse text table into rows of cells.
        
        Args:
            text: Text table string
            
        Returns:
            List[List[str]]: Table data as list of rows
        """
        try:
            lines = text.strip().split('\n')
            if not lines:
                return []
            
            # Remove empty lines
            lines = [line for line in lines if line.strip()]
            
            if not lines:
                return []
            
            # Split each line into columns based on multiple spaces or tabs
            table_rows = []
            for line in lines:
                # Split on multiple spaces or tabs
                columns = line.split()
                if columns:
                    table_rows.append(columns)
            
            return table_rows
        except Exception:
            return []
     
    def _merge_table_data(self, table_data_list: List[List[List[str]]]) -> List[List[str]]:
        """
        Merge multiple table data sets.
        
        Args:
            table_data_list: List of table data sets
            
        Returns:
            List[List[str]]: Merged table data
        """
        if not table_data_list:
            return []
        
        # Start with the first table
        merged_table = table_data_list[0].copy()
        
        # Add rows from subsequent tables, excluding header rows if they match
        for i in range(1, len(table_data_list)):
            table_data = table_data_list[i]
            if not table_data:
                continue
            
            # Check if first row matches the header of the merged table
            if (len(table_data) > 0 and len(merged_table) > 0 and
                table_data[0] == merged_table[0]):
                # Skip the header row and add the rest
                merged_table.extend(table_data[1:])
            else:
                # Add all rows
                merged_table.extend(table_data)
        
        return merged_table
    
    def _data_to_markdown_table(self, table_data: List[List[str]]) -> str:
        """
        Convert table data to markdown format.
        
        Args:
            table_data: Table data as list of rows
            
        Returns:
            str: Markdown formatted table
        """
        if not table_data:
            return ""
        
        # Determine the maximum number of columns
        max_cols = max(len(row) for row in table_data) if table_data else 0
        if max_cols == 0:
            return ""
        
        # Pad all rows to have the same number of columns
        padded_table_data = []
        for row in table_data:
            padded_row = row[:]
            while len(padded_row) < max_cols:
                padded_row.append('')
            padded_table_data.append(padded_row)
        
        # Create markdown table
        markdown_lines = []
        
        # Add header row
        if padded_table_data:
            header = "| " + " | ".join(padded_table_data[0]) + " |"
            markdown_lines.append(header)
            
            # Add separator row
            if len(padded_table_data) > 1:
                separator = "| " + " | ".join(['---'] * max_cols) + " |"
                markdown_lines.append(separator)
                
                # Add data rows
                for row in padded_table_data[1:]:
                    data_row = "| " + " | ".join(row) + " |"
                    markdown_lines.append(data_row)
        
        return "\n".join(markdown_lines)
    
    def _table_to_markdown(self, table_element) -> str:
        """
        Convert a table element to markdown format.
        
        Args:
            table_element: Unstructured table element
            
        Returns:
            str: Markdown formatted table
        """
        try:
            # Try to get the table metadata
            metadata = table_element.to_dict().get('metadata', {})
            text_as_html = metadata.get('text_as_html', None)
            
            if text_as_html:
                # Convert HTML table to markdown
                return self._html_table_to_markdown(text_as_html)
            else:
                # Fallback to basic table representation
                table_text = str(table_element)
                # Try to format as a simple markdown table
                return self._format_simple_table(table_text)
        except Exception:
            # If all else fails, return the raw table text
            return str(table_element)
    
    def _html_table_to_markdown(self, html_table: str) -> str:
        """
        Convert HTML table to markdown format using BeautifulSoup.
        
        Args:
            html_table: HTML table string
            
        Returns:
            str: Markdown formatted table
        """
        try:
            from bs4 import BeautifulSoup
            import pandas as pd
            from io import StringIO
            
            # Parse the HTML table
            soup = BeautifulSoup(html_table, 'html.parser')
            table = soup.find('table')
            
            if not table:
                return html_table
            
            # Convert to pandas DataFrame first
            # Find all rows
            rows = []
            for tr in table.find_all('tr'):
                row = []
                # Find all cells (both th and td)
                for cell in tr.find_all(['th', 'td']):
                    # Get text content and strip whitespace
                    text = cell.get_text(strip=True)
                    row.append(text)
                if row:  # Only add non-empty rows
                    rows.append(row)
            
            if not rows:
                return html_table
            
            # Convert to markdown table
            if len(rows) > 0:
                # Create header row
                header = "| " + " | ".join(rows[0]) + " |"
                
                # Create separator row
                separator = "| " + " | ".join(['---'] * len(rows[0])) + " |"
                
                # Create data rows
                data_rows = []
                for row in rows[1:]:
                    # Pad or truncate row to match header length
                    if len(row) < len(rows[0]):
                        row.extend([''] * (len(rows[0]) - len(row)))
                    elif len(row) > len(rows[0]):
                        row = row[:len(rows[0])]
                    data_rows.append("| " + " | ".join(row) + " |")
                
                # Combine all parts
                markdown_table = [header, separator] + data_rows
                return "\n".join(markdown_table)
            
            return html_table
        except Exception as e:
            # Fallback to raw HTML
            return html_table
    
    def _format_simple_table(self, table_text: str) -> str:
        """
        Format simple table text as markdown.
        
        Args:
            table_text: Raw table text
            
        Returns:
            str: Markdown formatted table
        """
        # Split into lines and try to format as a markdown table
        lines = table_text.strip().split('\n')
        if len(lines) < 1:
            return table_text
        
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        if len(lines) < 1:
            return table_text
        
        # Try to detect column separators (multiple spaces or tabs)
        # Split each line into columns
        table_rows = []
        for line in lines:
            # Split on multiple spaces or tabs
            columns = line.split()
            if columns:
                table_rows.append(columns)
        
        if not table_rows:
            return table_text
        
        # Determine the number of columns (maximum columns in any row)
        max_cols = max(len(row) for row in table_rows)
        
        # Pad all rows to have the same number of columns
        for row in table_rows:
            while len(row) < max_cols:
                row.append('')
        
        # Create markdown table
        markdown_lines = []
        
        # Header row
        header = "| " + " | ".join(table_rows[0]) + " |"
        markdown_lines.append(header)
        
        # Separator row
        separator = "| " + " | ".join(['---'] * max_cols) + " |"
        markdown_lines.append(separator)
        
        # Data rows
        for row in table_rows[1:]:
            data_row = "| " + " | ".join(row) + " |"
            markdown_lines.append(data_row)
        
        return '\n'.join(markdown_lines)
    
    def _count_merged_table_elements(self, elements: List, start_index: int) -> int:
        """
        Count how many consecutive table elements have been merged.
        
        Args:
            elements: List of unstructured elements
            start_index: Starting index
            
        Returns:
            int: Number of table elements processed
        """
        count = 0
        i = start_index
        
        while i < len(elements) and elements[i].to_dict()['type'] == 'Table':
            count += 1
            i += 1
        
        return count


def main():
    """Main function to demonstrate the parser usage."""
    parser = RAGParser()
    
    # Example usage
    sample_dir = Path("sample")
    if sample_dir.exists():
        for file_path in sample_dir.iterdir():
            # Skip hidden files
            if file_path.name.startswith('.'):
                continue
            if file_path.suffix.lower() == '.pdf':
                try:
                    print(f"Parsing {file_path.name}...")
                    markdown_content = parser.parse_file(file_path)
                    output_path = file_path.with_suffix('.md')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    print(f"Saved markdown to {output_path}")
                except Exception as e:
                    print(f"Error parsing {file_path.name}: {e}")


if __name__ == "__main__":
    main()