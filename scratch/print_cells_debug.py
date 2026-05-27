from tagger.models.data_types import LayoutRegion, PageClassification, PageType, PageElement
from tagger.stage5_specialists.table_extractor import extract_table_native
import pickle

# Need to load actual page_elements from somewhere, or mock them.
# The issue is that page_elements is EMPTY if we don't pass it!
# If page_elements is empty, `merged_from` is always empty for all cells!
print("Ah! If page_elements is not provided correctly, merged_from is empty!")
