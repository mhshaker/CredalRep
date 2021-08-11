
# load data
import Data.data_provider as dp

data_name = "Jdata/parkinsons"
features, target = dp.load_data(data_name)
print(features.shape)

# hyper opt setup
import sklearn.gqussian_process as gp
def optimize(prams, x, y):
    # prams = dict(zip)

# train model
# show results