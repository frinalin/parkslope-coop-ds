# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
from sentence_transformers import util
import torch

# API definition
app = Flask(__name__)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if nlp:
        try:
            json_ = request.json
            print(json_)
            query = json_['item']
            query_embedding = nlp.encode(query, convert_to_tensor=True)

            top_k = min(5, len(item_list))
            cos_scores = util.cos_sim(query_embedding, embeddings)[0]
            filtered_item_list = [(idx, item) for idx, item in enumerate(item_list) if query.lower() not in item]
            filtered_cos_scores = torch.index_select(cos_scores,
                                                    0,
                                                    torch.tensor([i[0] for i in filtered_item_list]))
            top_results = torch.topk(filtered_cos_scores, k=top_k)
            results_df = pd.DataFrame()
            results_df['value'] = top_results[0]
            results_df['idx'] = top_results[1]
            results_df = results_df.merge(pd.DataFrame(filtered_item_list, columns=['idx', 'item']),
                        how='left',
                        left_on='idx',
                        right_index=True)
            return results_df[['item', 'value']].to_json(orient='records')

        except:

            return jsonify({'trace': traceback.format_exc()})
        
    else:
        print('Train the model first')
        return('No model here to use')
    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    nlp = joblib.load('model.pkl')
    print('Model loaded')
    item_list = joblib.load('item_list.pkl')
    embeddings = joblib.load('embeddings.pkl')
    print('Item list and embeddings loaded')

    app.run(port=port, debug=True)
