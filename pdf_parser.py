from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions

from docling.chunking import HierarchicalChunker
from langchain.schema import Document
import re
import os
from collections import OrderedDict

try:
    from src.summaries_images import summaries
    summaries = OrderedDict(summaries)
except ImportError:
    summaries = OrderedDict()


class DoclingParser:
    def __init__(self):
        self.summaries = summaries.copy()
        self.pipeline_options = PdfPipelineOptions(
            do_ocr=False,
            do_table_structure=True,
            generate_picture_images=True,
            generate_page_images=True,
            do_formula_enrichment=True,
            images_scale=2,
            table_structure_options={"do_cell_matching": True}
            # accelerator_options=AcceleratorOptions(num_threads=4, device=AcceleratorDevice.CPU),
        )
        self.format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)}
        self.converter = DocumentConverter(format_options=self.format_options)

    def replace_base64_images(self, md_text, summary_dict):
        pattern = r'!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=\n]+\)'

        def replacement(match):
            if summary_dict:
                key, value = summary_dict.popitem(last=False)
                return f"\n\n{value}\n\n"
            else:
                return "\n\n[Image removed - no summary available]\n\n"

        return re.sub(pattern, replacement, md_text)

    def convert_pdf_to_markdown(self, pdf_path: str, output_path: str = None) -> str:
        if output_path is None:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}.md")
        
        result = self.converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown(image_mode="embedded")
        new_markdown = self.replace_base64_images(markdown_text, self.summaries.copy())
        markdown_text = new_markdown
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        return markdown_text

    def convert_pdf_to_chunks_with_metadata(self, pdf_path: str, chunk_size: int = 3000, chunk_overlap: int = 200) -> list[Document]:
        """
        Convert PDF to chunks with metadata including document name and page numbers.
        Returns a list of LangChain Document objects with metadata.
        """
        print(f"Processing PDF with metadata extraction: {pdf_path}")
        
        # Convert PDF using docling
        result = self.converter.convert(pdf_path)
        doc = result.document
        
        # Initialize chunker
        chunker = HierarchicalChunker()
        
        # Get document name
        document_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        # Process chunks and extract metadata
        langchain_docs = []
        chunk_id = 0
        
        for chunk in chunker.chunk(doc):
            try:
                # Get chunk content
                chunk_text = chunk.text
                
                # Extract metadata from chunk
                chunk_dict = chunk.model_dump()
                
                # Extract filename
                filename = document_name
                if 'meta' in chunk_dict and 'origin' in chunk_dict['meta']:
                    origin_filename = chunk_dict['meta']['origin'].get('filename')
                    if origin_filename:
                        filename = os.path.splitext(os.path.basename(origin_filename))[0]
                
                # Extract page number
                page_num = None
                if ('meta' in chunk_dict and 
                    'doc_items' in chunk_dict['meta'] and 
                    chunk_dict['meta']['doc_items'] and
                    'prov' in chunk_dict['meta']['doc_items'][0] and
                    chunk_dict['meta']['doc_items'][0]['prov']):
                    page_num = chunk_dict['meta']['doc_items'][0]['prov'][0].get('page_no')
                
                # Extract heading information
                heading = None
                if ('meta' in chunk_dict and 
                    'headings' in chunk_dict['meta'] and 
                    chunk_dict['meta']['headings']):
                    heading = chunk_dict['meta']['headings'][0]
                
                # Create metadata dict
                metadata = {
                    'source': pdf_path,
                    'document_name': filename,
                    'page_number': page_num if page_num is not None else 'Unknown',
                    'heading': heading,
                    'chunk_id': chunk_id
                }
                
                # Create LangChain Document with metadata
                langchain_doc = Document(
                    page_content=chunk_text,
                    metadata=metadata
                )
                
                langchain_docs.append(langchain_doc)
                chunk_id += 1
                
                # Debug print for first few chunks
                if chunk_id <= 3:
                    print(f"Chunk {chunk_id}: document='{filename}', page={page_num}, heading='{heading}'")
                    
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}")
                # Create basic document without detailed metadata
                metadata = {
                    'source': pdf_path,
                    'document_name': document_name,
                    'page_number': 'Unknown',
                    'heading': None,
                    'chunk_id': chunk_id
                }
                langchain_doc = Document(
                    page_content=chunk.text,
                    metadata=metadata
                )
                langchain_docs.append(langchain_doc)
                chunk_id += 1
        
        print(f"Created {len(langchain_docs)} chunks with metadata from {document_name}")
        return langchain_docs