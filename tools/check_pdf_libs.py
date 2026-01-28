import importlib
print('fitz' if importlib.util.find_spec('fitz') is not None else ('pdf2image' if importlib.util.find_spec('pdf2image') is not None else 'none'))
