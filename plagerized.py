import numpy as np #(működik a Moodle-ben is)


######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    entropy = 0
    if n_cat1 == 0 or n_cat2 == 0:
        return entropy
    pCat1 = n_cat1/(n_cat1+n_cat2)
    pCat2 = n_cat2/(n_cat1+n_cat2)
    entropy = - pCat1*np.log2(pCat1) - pCat2*np.log2(pCat2)
    return entropy

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(featuresL: list,
                        labelsL: list) -> (int, int):
    best_separation_feature, best_separation_value = 0, 0
    
    features = np.array(featuresL)
    labels = np.array(labelsL)
    infoProfit = 0
    #getting each column in features  
    for column in range(features.shape[1]):
        for val in features[:,column]:
            #getting the entropy of the overall data
            H = get_entropy(np.sum(labels == 0), np.sum(labels == 1))
            #getting the number of elements in the overall data
            nH = len(labels)

            #getting the entropy of the left side of the separation
            He = get_entropy(np.sum(labels[features[:,column] <= val] == 0), np.sum(labels[features[:,column] <= val] == 1))
            #getting the number of elements lesser than the val
            nHe = np.sum(features[:,column] <= val)


            #getting the entropy of the right side of the separation
            Hf = get_entropy(np.sum(labels[features[:,column] > val] == 0), np.sum(labels[features[:,column] > val] == 1))
            #getting the number of elements greater than the val
            nHf = np.sum(features[:,column] > val)

            #getting the entropy of the separation
            currInfoProfit = H - (nHe/nH *He + nHf/nH * Hf)
            #if the entropy of the separation is lower than the previous one, we save the column and the value of the separation
            if currInfoProfit > infoProfit:
                infoProfit = currInfoProfit
                best_separation_feature = column
                best_separation_value = val

    return best_separation_feature, best_separation_value

def build_tree(train_data: np.ndarray):
    best_separation_feature, best_separation_value = get_best_separation(train_data[:,:-1], train_data[:,-1])
    root=dict()
    #root['feature'] = best_separation_feature
    #root['value'] = best_separation_value
    split(root, train_data,0)
    return root
    

def split(node, train_data, depth):
    #return if labels are the same or entropy = 0
    if get_entropy(np.sum(train_data[:,-1] == 0), np.sum(train_data[:,-1] == 1)) == 0:
        node['projection'] = train_data[:,-1][0]
        node['leaf'] = True
        return node

    node['leaf'] = False
    node['feature'], node['value'] = get_best_separation(train_data[:,:-1], train_data[:,-1])
    

    leftData = train_data[train_data[:,node['feature']] <= node['value']]
    rightData = train_data[train_data[:,node['feature']] > node['value']]

    leftNode = dict()
    leftNode['feature'], leftNode['value'] = get_best_separation(leftData[:,:-1], leftData[:,-1])
    rightNode = dict()
    rightNode['feature'], rightNode['value'] = get_best_separation(rightData[:,:-1], rightData[:,-1])
    split(leftNode, leftData, depth+1)
    split(rightNode, rightData, depth+1)

    node['left'] = leftNode
    node['right'] = rightNode


def predict(node, row):
    #if node is null or it is leaf

    if not node:
        return 1
    if node['leaf']:
        #print("hehe")
        return node['projection']
    
    if row[node['feature']] <= node['value']:
        return predict(node['left'], row)
    else:
        return predict(node['right'], row)



        

################### 3. feladat, döntési fa implementációja ####################
def main():
    #TODO: implementing a decision tree that first trains from a given dataset, then predicts the labels of a given test dataset (the train dataset has n+1 columns, the last one is the label)
    # Load the training dataset
    train_data = np.genfromtxt('train.csv', delimiter=',')

    # Train the decision tree
    tree = build_tree(train_data)

    #print("tree built")
    # Load the test dataset
    test_data = np.genfromtxt('test.csv', delimiter=',')
    #print("test data loaded")
    # Make predictions
    predictions = [predict(tree, row) for row in test_data]
    #print("predictions made")

    # Write predictions to a CSV file
    np.savetxt('results.csv', predictions, fmt='%d', delimiter=',')


    return 0

if __name__ == "__main__":
    main()