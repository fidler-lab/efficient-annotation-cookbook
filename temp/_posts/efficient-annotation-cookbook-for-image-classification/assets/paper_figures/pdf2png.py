from glob import glob
from pdf2image import convert_from_path


for p in glob('*pdf'):
    pages = convert_from_path(p, 500)
    for page in pages:
        page.save(p.replace('pdf', 'png'), 'PNG')
