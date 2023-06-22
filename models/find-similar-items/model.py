# Import dependencies
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib

# Load data set
df = pd.read_csv('../../raw_data/2022-12-07.csv', index_col=0)

# Data Preprocessing
# Create list of cleaned, unique items
df['item_clean']=df['item'].str.lower().str.split('-').str[0] \
                           .str.replace('.', '', regex=False) \
                           .str.strip() \
                           .str.replace(r'\s+', ' ', regex=True)
item_list = df['item_clean'].drop_duplicates().to_list()

# Model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(item_list, convert_to_tensor=True)

# Save Model
joblib.dump(model, 'model.pkl')
print('Model dumped!')

# Save other artifacts
joblib.dump(item_list, 'item_list.pkl')
joblib.dump(embeddings, 'embeddings.pkl')
