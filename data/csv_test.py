import csv
import numpy as np

# myDic = {1: ['hello world', 'ppp dada'], 2: ['heat slave', 'stop crying'], 3: ['!']}
#
# header = ["item_id", "reviews"]
# with open('mycsvfile.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     for k, v in myDic.items():
#         writer.writerow([k, ' '.join(v)])
# f.close()


reviews_emb = np.load('reviews_emb.npy')
print(reviews_emb.shape)
