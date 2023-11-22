import numpy as np #(működik a Moodle-ben is)
import csv



class Node:
    separation_idx = None
    separation_value = None
    more_tree = None
    less_tree = None
    chosen_label = None
    isLeaf = False


def zero_one_in_labels(labels:list) -> (int, int):
    zeroOut = labels.count(0)
    oneOut = labels.count(1)
    return zeroOut,oneOut


######################## 1. feladat, entrópiaszámítás #########################
def get_entropy(n_cat1: int, n_cat2: int) -> float:
    if (n_cat1 == 0) or (n_cat2 == 0):
        return 0
    p1x = n_cat1/(n_cat1+n_cat2)
    p2x = n_cat2/(n_cat1+n_cat2)
    entropy = -1*((p1x * np.log2(p1x)) + (p2x * np.log2(p2x)) )
    return entropy

###################### 2. feladat, optimális szeparáció #######################
def get_best_separation(featuresL: list, labelsL: list) -> (int, int):
    #zerosones = zero_one_in_labels(labels)
    
    lh_felosztas = 0
    lh_tul_index = 0
    lh_tul_hatar = 0
    
    features = np.array(featuresL)
    labels = np.array(labelsL)
    
    ''' 
    no_feature_rows = features.shape[0]
    no_feature_cols = features.shape[1]
    H_L = get_entropy(np.count_nonzero(labels == 0), np.count_nonzero(labels == 1))
    L = len(labels)

    for i in range(no_feature_rows):
        for j in range(no_feature_cols):
            for k in range(features[i][j]):
                e = 0
                e_cat1 = 0
                e_cat2 = 0
                f = 0
                f_cat1 = 0
                f_cat2 = 0
                for l in range(no_feature_rows):
                    if features[l][j] <= k:
                        e +=1
                        if labels[l] == 1:
                            e_cat1 += 1
                        else:
                            e_cat2 += 1
                    else:
                        f +=1
                        if labels[l] == 1:
                            f_cat1 += 1
                        else:
                            f_cat2 += 1
                curr_informacinyereseg = H_L - ( ((e/L)*get_entropy(e_cat1, e_cat2)) + ((f/L)*get_entropy(f_cat1, f_cat2) ) )
                if curr_informacinyereseg >= lh_felosztas:
                    lh_felosztas = curr_informacinyereseg
                    lh_tul_index = j
                    lh_tul_hatar = k
    '''
    for column in range(features.shape[1]):
        for val in features[:,column]:
            #getting the entropy of the overall data
            H = get_entropy(np.count_nonzero(labels == 0), np.count_nonzero(labels == 1))
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
            curr_informacinyereseg = H - (nHe/nH *He + nHf/nH * Hf)
            #if the entropy of the separation is lower than the previous one, we save the column and the value of the separation
            if curr_informacinyereseg > lh_felosztas:
                lh_felosztas = curr_informacinyereseg
                lh_tul_index = column
                lh_tul_hatar = val                    
         
    return lh_tul_index, lh_tul_hatar

################### 3. feladat, döntési fa implementációja ####################
def treebuilder(features:list, labels:list, tree:Node):

    
    if get_entropy(labels.count(0), labels.count(1)) == 0:
        #leaf
        if labels.count(0) > labels.count(1):
            tree.chosen_label = 0
            tree.isLeaf = True
            return
        else:
            tree.chosen_label = 1
            tree.isLeaf = True
            return
    else:
        #Not a leaf
        (tree.separation_idx, tree.separation_value) = get_best_separation(features, labels)
        
        top_features_list = []
        top_labels_list = []
        bottom_features_list = []
        bottom_labels_list = []
        
        for i in range(len(features)):
            if features[i][tree.separation_idx] <= tree.separation_value:
                bottom_features_list.append(features[i])
                bottom_labels_list.append(labels[i])
            else:
                top_features_list.append(features[i])
                top_labels_list.append(labels[i])
    
        tree.less_tree = Node()
        tree.more_tree = Node()
        
        treebuilder(bottom_features_list, bottom_labels_list, tree.less_tree)
        treebuilder(top_features_list, top_labels_list, tree.more_tree)



def decide(features_row, tree):
    if tree.less_tree == None and tree.more_tree == None:
        return tree.chosen_label
    else:
        if  int(features_row[tree.separation_idx ]) <= int(tree.separation_value):
            return decide(features_row, tree.less_tree)
        else:
            return decide(features_row, tree.more_tree)
    

def printTree(node:Node, level=0):
    if node != None:
        printTree(node.more_tree, level + 1)
        print(' ' * 4 * level + '-> ' + 'id: ' + str(node.separation_idx) + ' value: ' + str(node.separation_value) + ' label: ' + str(node.chosen_label))
        printTree(node.less_tree, level + 1)


def main():
    #get train data
    train_features = []
    train_values = []
    with open('train.csv', newline='') as traincsv:
        reader = csv.reader(traincsv, delimiter=',')
        for row in reader:
            row_as_int = [int(value) for value in row]
            train_features.append(row_as_int[:-1]) 
            train_values.append(int(row[-1]))
           
    #Define and build the decision tree, based on train data
    tree = Node()
    
    #z_o =zero_one_in_labels(train_values)
    #print("Test data labels: ")
    #print(z_o)
    #print(train_values.count(0))
    
    #print(get_best_separation(train_features, train_values))
    
    
    treebuilder(train_features, train_values, tree)
    #print("The Tree is: ")
    #printTree(tree)
    #Get test features
    test_features = []
    
    with open('test.csv', newline='') as testcsv:
        reader = csv.reader(testcsv, delimiter=',')
        for row in reader:
            test_features.append(row) 
    
    #Make decision based on the tree and the test-features
    #Create and open output file, don't forget to close it
    file_name = 'results.csv'

    # Open the file in write mode and create a CSV writer
    with open(file_name, mode='w', newline='') as outcsv:
        csv_writer = csv.writer(outcsv)
        for row in test_features:
            #decision = randint(0,1)
            decision = decide(row, tree)
            csv_writer.writerow(str(decision))
        
    return 0

if __name__ == "__main__":
    main()
