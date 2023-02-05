import argparse
from retrofitting import *
from intrinsic_evaluation import *
from extrinsic_evaluation import *

from halo import Halo
import time


if __name__=='__main__':
  
  start = time.time()
  
  parser = argparse.ArgumentParser()
  parser.add_argument("-i", "--input", type=str, default=None, help="Input word vecs")
  parser.add_argument("-l", "--lexicon", type=str, default=None, help="Lexicon file name")
  parser.add_argument("-o", "--output", type=str, help="Output word vecs")

  parser.add_argument("-e", "--evaluation", type=str, default=None, help="Type of evaluation (i=intrinsic evaluation, s=extrinsic evaluation)")
  parser.add_argument("-b", "--benchmarks", type=str, default=None, help="File of benchmarks")
  parser.add_argument("-a", "--train", type=str, default=None, help="Corpus of train")
  parser.add_argument("-t", "--test", type=str, default=None, help="Corpus of test")

  
  args = parser.parse_args()

  outFileName = args.output

  
  with Halo(text="Retrofitting", spinner='dots') as spinner:
    # Enrich the word vectors using lexicon and print the enriched vectors 
    retrofit(args.input, args.lexicon, outFileName)
    spinner.succeed("Retrofitting")
  
  # count the time of retrofitting
  end = time.time()
  print("Execution time:", round(end-start,2),"s")
  
  # store the path of file  
  path_embedding_retro = "./" + outFileName

  with Halo(text="Evaluation", spinner='dots') as spinner:
    # Type of evaluation = i (intrinsic) / s (extrinsic)
    if (args.evaluation == "i"):
      # calculate the score of spearman 
      distr_spearman_r, retro_spearman_r = word_similarity(args.benchmarks, args.input, path_embedding_retro)
      print("before retrofitting:", distr_spearman_r)
      print("after retrofitting:", retro_spearman_r)

    elif (args.evaluation == "s"):
      # calculate the accuracy of the task sentiment analysis
      print(" before retrofitting:", sentiment_analysis(args.input, args.train, args.test))
      print(" after retrofitting:", sentiment_analysis(path_embedding_retro, args.train, args.test))
    spinner.succeed("Evaluation")
    

