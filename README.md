# Improving vector space representations using semantic resources
### Authors
  * **Shan-ning PHILIPPOT**
  * **Yingqin HU**
  * **Xirui ZHANG**

This project aims to implement and evaluate the retrofitting algorithm proposed by Faruqui et al (2015) on French and English. This algorithm is designed to use the knowledge from semantic resources (synonymy, hypernymy, hyponymy) to improve a lexicon of distributional vector representations of words. 

You can retrofit a skip-gram model using the semantic resources of your choices (e.g. WOLF for French, PPDB / wordnet / framenet for English). After retrofitting, you can also test its performance by the instrinsic evaluation (word similarity) and the extrinsic evaluation (sentiment analysis).

### Requirement
  Python 3.9.6
 
### Data you need
  1. Word vector file 
  2. Lexicon file 
  3. Benchmarks file for instrinsic evaluation (word similarity)
  4. Test and Train file for extrinsic evaluation (sentiment analysis)

  #### Note:   
  1. The vector file should have one word vector per line as follows (space delimited):  
    ```and 0.098221 -0.050079 0.076295 -0.054293 -0.107857...```
  2. The lexicon file should have one word and one or more of its synonyms per line as follows (space delimited):  
    ```numeral figure digit...```
  3. The benchmarks file (word similarity) should have a pair of words and a score per line as follows (space delimited):  
    ```tiger	cat	7.35```
  4. The test and train files (sentiment analysis) should have a class (-1/1) and a phrase or a paragraph (space delimited):  
     ```1 A masterpiece four years in the making .```

### Running the program
```
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")

  parser.add_argument("-e", "--evaluation", type=str, default=None, help="Type of evaluation (i=intrinsic evaluation, s=extrinsic evaluation)")
  parser.add_argument("-b", "--benchmarks", type=str, default=None, help="File of benchmarks")
  parser.add_argument("-a", "--train", type=str, default=None, help="Corpus of train")
  parser.add_argument("-t", "--test", type=str, default=None, help="Corpus of test")
```

  For intrinsic evaluation:  
```python3 main.py -i word_vec_file -l lexicon_file -o out_vec_file -e i -b benchmarks_file```  
  For extrinsic evaluation:  
```python3 main.py -i word_vec_file -l lexicon_file -o out_vec_file -e s -a train_file -t test_file```  

### Reference 
Faruqui, M.; Dodge, J.; Jauhar, S. K.; Dyer, C.; Hovy, E. H. & Smith, N. A. (2015), Retrofitting Word Vectors to Semantic Lexicons., in Rada Mihalcea; Joyce Yue Chai & Anoop Sarkar, ed., 'HLT-NAACL' , The Association for Computational Linguistics, pp. 1606-1615.
