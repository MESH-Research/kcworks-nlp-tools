## Collection of Materials Used
This document includes materials used during the learning and testing process.

### Stage 1 Data Acquisition and Stage 2 Text Extraction
Resources for both steps are listed together as step 1 was done to prepare for step 2.

1. Extracting text from a pdf using Fitz: https://neurondai.medium.com/how-to-extract-text-from-a-pdf-using-pymupdf-and-python-caa8487cf9d
2. Extract text with OCR, Pytesseract: https://medium.com/@pawan329/ocr-extract-text-from-image-in-8-easy-steps-3113a1141c34
3. When the pdf is image-based: https://stackoverflow.com/questions/69415164/pymupdf-how-to-convert-pdf-to-image-using-the-original-document-settings-for 
4. Tesseract and Pytesseract in practice: https://nanonets.com/blog/ocr-with-tesseract/
5. Stackoverflow on extracting text from .doc files: https://stackoverflow.com/questions/25228106/how-to-extract-text-from-an-existing-docx-file-using-python-docx
6. From ByteScrum Tech on extracting info from .docx: https://blog.bytescrum.com/extracting-information-from-a-docx-file-using-python
7. How to work with Apache tika: https://medium.com/@masreis/text-extraction-and-ocr-with-apache-tika-302464895e5f
8. Great examples from Apache tika's website: https://tika.apache.org/1.8/examples.html
9. Learn how to work with Tika: https://tika.apache.org/3.0.0/gettingstarted.html
10. PPTX: https://rutuparnavk.medium.com/parse-powerpoint-documents-using-python-the-easy-way-89e6829495ee
11. Extract text from pptx: https://medium.com/@juanraful/extracting-text-from-powerpoint-slides-using-python-c07b1aad3ec5

12. fitz/PyMuPDF documentation: https://pymupdf.readthedocs.io/en/latest/
13. Pytesseract documentation: https://pypi.org/project/pytesseract/
14. Image Module from Pillow (PIL): https://pillow.readthedocs.io/en/stable/reference/Image.html
15. Python-docx documentation: https://python-docx.readthedocs.io/en/latest/
16. Python-pptx documentation: https://python-pptx.readthedocs.io/en/latest/

### Stage 3. Text Preprocessing
This step, I mainly learned how much preprocessing for models like BERT requires. Here are some fundational materials.
1. The BERT paper: https://arxiv.org/abs/1810.04805
2. BERT research series from ChrisMcCormick (here is the youtube series, in step 4 I also included his tutorial on BERT word embeddings): https://youtu.be/FKlPCK1uFrc?si=f7j4TCeF8zMf7Y21
3. What we know about how BERT works: https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00349/96482/A-Primer-in-BERTology-What-We-Know-About-How-BERT
4. Why does BERT work so well: https://www.reddit.com/r/MachineLearning/comments/k4saj5/d_why_does_bert_perform_so_well/
5. Using unlabeled data is part of pre-training: https://medium.com/dair-ai/papers-explained-02-bert-31e59abc0615
6. Again, the two-step process where using unlabeled text to learn contextual embeddings is step 1: https://www.geeksforgeeks.org/explanation-of-bert-model-nlp/
7. Some BERT basics: https://towardsdatascience.com/minimal-requirements-to-pretend-you-are-familiar-with-bert-3889023e4aa9
8. Google AI forum on this: https://discuss.ai.google.dev/t/does-text-preprocessing-or-cleaning-required-for-bert-model-or-others-one/29416
9. A comparative study on transformers and other classifiers, "Is text preprocessing still worth the time? A comparative survey on the influence of popular preprocessing methods on Transformers and traditional classifiers": https://www-sciencedirect-com.proxy1.cl.msu.edu/science/article/pii/S0306437923001783
10. Again, transformer models do not need much preprocessing: https://www.reddit.com/r/MachineLearning/comments/wa1rt0/d_how_important_is_text_preprocessing_nowadays/
11. Similar answers: https://stackoverflow.com/questions/63979544/using-trained-bert-model-and-data-preprocessing/63986348#63986348

### Stage 4. Embeddings
I first read the "What are embeddings" during my theoretical learning phase and found it very useful. I also listed tutorials with codes for generating embeddings using BERT here.
1. What are embeddings? By Vicki Boykis: https://vickiboykis.com/what_are_embeddings/
2. Word embeddings tutorials (by Chris McCormick and Nnick Ryan): https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
3. Generate word embedding using BERT: https://www.geeksforgeeks.org/how-to-generate-word-embedding-using-bert/
4. BERT on huggingface: https://huggingface.co/docs/transformers/model_doc/bert#bertforpretraining 
5. More on BERT-base-uncased: https://huggingface.co/google-bert/bert-base-uncased
6. Understanding BERT Embeddings and Tokenization from Rohan-Paul: https://youtu.be/30zPz5Xz-8g?si=aJGv2Y6_vDPurozp

#### Explorartory data analysis (EDA) for our data stored in bert_embeddings_subset.csv
In addition, I incorporated some resources into `examine_bert_embedding_test.py` to focus on evaluating embedding results. While I have conducted initial testing and evaluations in stage 4, there is still much more to explore and learn.

1. Diving into word embeddings with EDA: https://towardsdatascience.com/eda-for-word-embeddings-224c524b5769
2. Cosine similarity between two arrays for word embeddings: https://medium.com/@techclaw/cosine-similarity-between-two-arrays-for-word-embeddings-c8c1c98811b
3. Why cosine similarity for transformer text embeddings: https://www.reddit.com/r/learnmachinelearning/comments/12cp2cg/why_cosine_similarity_for_transformer_text/
4. A critical voice of using cosine similarity--> Is cosine similarity of embeddings really about similarity? https://arxiv.org/abs/2403.05440
5. Another key field is reducing dimensions (e.g. via Principal Component Analysis, aka PCA or TSNE) before doing any visualization: https://medium.com/@sachinsoni600517/mastering-t-sne-t-distributed-stochastic-neighbor-embedding-0e365ee898ea 
I have not looked into this too much. Dimension reduction is its own field.
6. Nearest Neighbor Search(NNS),  k-Nearest Neighbor (KNN) search, Approximate Nearest Neighbor (ANN) search:
- See this complex guide for TensorFlow, especially under "Build the ANN Index for the Embeddings": https://www.tensorflow.org/hub/tutorials/semantic_approximate_nearest_neighbors
- Google's documentations on finding ann: https://cloud.google.com/spanner/docs/find-approximate-nearest-neighbors

