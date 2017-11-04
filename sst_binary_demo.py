from encoder import Model
from matplotlib import pyplot as plt
from utils import sst_binary, train_with_reg_cv
import pickle as pk

model = Model()

try:
    ##transformed_data = pk.load(open('data/labelled_transformed_data.pkl'))
    transformed_data = pk.load(open('data/labelled_transformed_data.pkl', 'rb'), encoding="utf-8", errors="strict")
    trXt = transformed_data['trXt']
    trY = transformed_data['trY']
    vaXt = transformed_data['vaXt']
    vaY = transformed_data['vaY']
    teXt = transformed_data['vaXt']
    teY = transformed_data['vaY']
except Exception as e:
    print ('Transformed data pickle file not found, creating one ...  ||', e)
    trX, vaX, teX, trY, vaY, teY = sst_binary()
    trXt = model.transform(trX)
    vaXt = model.transform(vaX)
    teXt = model.transform(teX)
    pk.dump({'trXt':trXt, 'trY':trY, 'vaXt':vaXt, 'vaY':vaY}, open('data/labelled_transformed_data.pkl', 'wb')) 

# classification results
#full_rep_acc, c, nnotzero = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)
full_rep_acc, c, nnotzero = train_with_reg_cv(trXt[:-1000], trY[:-1000], trXt[-1000:], trY[-1000:], trXt[-1000:], trY[-1000:])
print('%05.2f test accuracy'%full_rep_acc)
print('%05.2f regularization coef'%c)
print('%05d features used'%nnotzero)

## visualize sentiment unit
#sentiment_unit = trXt[:, 2388]
#plt.hist(sentiment_unit[trY==0], bins=25, alpha=0.5, label='neg')
#plt.hist(sentiment_unit[trY==1], bins=25, alpha=0.5, label='pos')
#plt.legend()
#plt.show()
