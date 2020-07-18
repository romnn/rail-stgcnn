import glob
import os.path

from PyPDF2 import PdfFileMerger


def concat_pdfs(source_dir, out_file, sort=True):
    # print(source_dir, out_file)
    pdfs = glob.glob(os.path.join(source_dir, "*.pdf"))
    pdfs = sorted(list(pdfs))
    if len(pdfs) < 1:
        return

    try:
        os.makedirs(os.path.dirname(out_file))
    except FileExistsError:
        pass

    merger = PdfFileMerger()

    for pdf in pdfs:
        merger.append(pdf)

    merger.write(out_file)
    merger.close()
