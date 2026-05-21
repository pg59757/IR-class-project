import sys
import os

# Adiciona a raiz do projeto ao Python path
# Isto permite imports do tipo "from src.search.preprocessor import ..."
sys.path.insert(0, os.path.dirname(__file__))