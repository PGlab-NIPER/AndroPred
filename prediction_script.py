#!/usr/bin/env python3

""" 
AndroPred is a Deep Neural Network based tool to predict Androgen Receptor (AR) inhibitor.

It calculates descriptors using PaDEL, and uses them to
predict inhibitors for AR with a trained DNN model. 

Compounds' SMILES must be in .smi file in the same folder.
Train data (data.csv), model (DeePredmodel.h5), and PaDEL
folder must be in the same folder, too. 

Prediction results (class and probability) will be saved in AR-inhibitor_predictions.csv

Edited by Anju Sharma
"""

import os
import sys
import pandas as pd  
import numpy as np
import subprocess
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder



# Create function to get predictions using trained model
def ar(input_ar: pd.DataFrame) -> np.ndarray:
    """Function to predict activity for a set
    set of samples with PaDEL features.

    Args:
        input_ar (pd.DataFrame): PaDEL Features for compounds

    Returns:
        np.ndarray: AR class (1 inhibitor; 0 non-inhibitor)
    """
    # Encoding class labels
    A = ['Inhibitor', 'Noninhibitor']
    encoder = LabelEncoder()
    encoder.fit(A)
    encoded_Y = encoder.transform(A)
    
    # Load training data and scale it 
    ar_data = pd.read_csv(os.path.join(path, "input-dnn.csv"), header=None)
    scaler = StandardScaler()  
    ar_data = scaler.fit_transform(ar_data)  
   
   # Transform user data to numpy to avoid conflict with names
    ar_user_input = scaler.transform(input_ar.to_numpy())
    
    # Load model
    loaded_model = load_model(os.path.join(path, "deepSSLmodel.h5")) 
    print("Model loaded")
   
   # Get predictions for user input
    prediction = loaded_model.predict(ar_user_input)

    a = prediction[:,1]
    b=prediction[:,0]
    c=[]
    for i in range(len(a)):
        if a[i] >= b[i]:
            c.append(a[i])
        else:
            c.append(b[i])
    
    prediction = encoder.inverse_transform(prediction.argmax(axis=1))
   
    return prediction, c
    

# Create main function to run descriptor calculation and predictions 
def run_prediction(folder: str) -> None:
    """Function to calculate descriptors (using PaDEL) and to generate
    predictions of AR inhibitors/ noninhibitors for a set of compounds (SMILES).

    Args:
        folder (str): Folder to search for ".smi" file (multiple structures)

    Returns:
        CSV file with resulting AR activity class (1 inhibitor; 0 noninhibitor)
    """
    # Define command for PaDEL
    padel_cmd = [
        'java', '-jar', 
        os.path.join(path, 'PaDEL-Descriptor/PaDEL-Descriptor.jar'),
        '-descriptortypes', 
        os.path.join(path, 'PaDEL-Descriptor/descriptors.xml'), 
        '-dir', folder, '-file', folder + '/PaDEL_features.csv', 
        '-2d', '-fingerprints', '-removesalt', '-retainorder', '-detectaromaticity', 
        '-standardizenitro']
    # Calculate features
    subprocess.call(padel_cmd)
    print("Features calculated")
    # Create Dataframe for calculated features
    input_ar =pd.read_csv(folder + "/PaDEL_features.csv")
    input_ar.fillna(0, inplace=True)
    # Store name of each sample
    names = input_ar['Name'].copy()
    # Remove names
    input_ar= input_ar.drop(['Name', 'nB', 'nP', 'nBondsQ', 'nHsSH', 'nHsNH3p', 'nHssNH2p', 'nHsssNHp', 'nHmisc', 'nsLi', 'nssBe', 'nssssBem', 'nsBH2', 'nssBH', 'nsssB', 'nssssBm', 'nsNH3p', 'nssNH2p', 'nsssNHp', 'nddsN', 'naOm', 'nsSiH3', 'nssSiH2', 'nsssSiH', 'nsPH2', 'nssPH', 'nsssP', 'ndsssP', 'nddsP', 'nsssssP', 'nsSH', 'ndssS', 'nssssssS', 'nSm', 'nsGeH3', 'nssGeH2', 'nsssGeH', 'nssssGe', 'nsAsH2', 'nssAsH', 'nsssAs', 'ndsssAs', 'nddsAs', 'nsssssAs', 'nsSeH', 'ndSe', 'nssSe', 'naaSe', 'ndssSe', 'nssssssSe', 'nddssSe', 'nsSnH3', 'nssSnH2', 'nsssSnH', 'nssssSn', 'nsPbH3', 'nssPbH2', 'nsssPbH', 'nssssPb', 'SHsSH', 'SHsNH3p', 'SHssNH2p', 'SHsssNHp', 'SHmisc', 'SsLi', 'SssBe', 'SssssBem', 'SsBH2', 'SssBH', 'SsssB', 'SssssBm', 'SsNH3p', 'SssNH2p', 'SsssNHp', 'SddsN', 'SaOm', 'SsSiH3', 'SssSiH2', 'SsssSiH', 'SsPH2', 'SssPH', 'SsssP', 'SdsssP', 'SddsP', 'SsssssP', 'SsSH', 'SdssS', 'SssssssS', 'SSm', 'SsGeH3', 'SssGeH2', 'SsssGeH', 'SssssGe', 'SsAsH2', 'SssAsH', 'SsssAs', 'SdsssAs', 'SddsAs', 'SsssssAs', 'SsSeH', 'SdSe', 'SssSe', 'SaaSe', 'SdssSe', 'SssssssSe', 'SddssSe', 'SsSnH3', 'SssSnH2', 'SsssSnH', 'SssssSn', 'SsPbH3', 'SssPbH2', 'SsssPbH', 'SssssPb', 'minHsSH', 'minHsNH3p', 'minHssNH2p', 'minHsssNHp', 'minHmisc', 'minsLi', 'minssBe', 'minssssBem', 'minsBH2', 'minssBH', 'minsssB', 'minssssBm', 'minsNH3p', 'minssNH2p', 'minsssNHp', 'minddsN', 'minaOm', 'minsSiH3', 'minssSiH2', 'minsssSiH', 'minsPH2', 'minssPH', 'minsssP', 'mindsssP', 'minddsP', 'minsssssP', 'minsSH', 'mindssS', 'minssssssS', 'minSm', 'minsGeH3', 'minssGeH2', 'minsssGeH', 'minssssGe', 'minsAsH2', 'minssAsH', 'minsssAs', 'mindsssAs', 'minddsAs', 'minsssssAs', 'minsSeH', 'mindSe', 'minssSe', 'minaaSe', 'mindssSe', 'minssssssSe', 'minddssSe', 'minsSnH3', 'minssSnH2', 'minsssSnH', 'minssssSn', 'minsPbH3', 'minssPbH2', 'minsssPbH', 'minssssPb', 'maxHsSH', 'maxHsNH3p', 'maxHssNH2p', 'maxHsssNHp', 'maxHmisc', 'maxsLi', 'maxssBe', 'maxssssBem', 'maxsBH2', 'maxssBH', 'maxsssB', 'maxssssBm', 'maxsNH3p', 'maxssNH2p', 'maxsssNHp', 'maxddsN', 'maxssssNp', 'maxaOm', 'maxsSiH3', 'maxssSiH2', 'maxsssSiH', 'maxssssSi', 'maxsPH2', 'maxssPH', 'maxsssP', 'maxdsssP', 'maxddsP', 'maxsssssP', 'maxsSH', 'maxssS', 'maxaaS', 'maxdssS', 'maxddssS', 'maxssssssS', 'maxSm', 'maxsGeH3', 'maxssGeH2', 'maxsssGeH', 'maxssssGe', 'maxsAsH2', 'maxssAsH', 'maxsssAs', 'maxdsssAs', 'maxddsAs', 'maxsssssAs', 'maxsSeH', 'maxdSe', 'maxssSe', 'maxaaSe', 'maxdssSe', 'maxssssssSe', 'maxddssSe', 'maxsSnH3', 'maxssSnH2', 'maxsssSnH', 'maxssssSn', 'maxsPbH3', 'maxssPbH2', 'maxsssPbH', 'maxssssPb', 'n9Ring', 'n10Ring', 'n11Ring', 'n12Ring', 'nG12Ring', 'nF4Ring', 'nF5Ring', 'n8HeteroRing', 'n9HeteroRing', 'n10HeteroRing', 'n11HeteroRing', 'n12HeteroRing', 'nG12HeteroRing', 'nFHeteroRing', 'nF4HeteroRing', 'nF5HeteroRing', 'nTHeteroRing', 'MACCSFP1', 'MACCSFP2', 'MACCSFP3', 'MACCSFP4', 'MACCSFP5', 'MACCSFP6', 'MACCSFP7', 'MACCSFP9', 'MACCSFP10', 'MACCSFP12', 'MACCSFP14', 'MACCSFP15', 'MACCSFP18', 'MACCSFP21', 'MACCSFP29', 'MACCSFP31', 'MACCSFP35', 'MACCSFP44', 'MACCSFP68', 'MACCSFP101', 'MACCSFP166', 'PubchemFP4', 'PubchemFP5', 'PubchemFP6', 'PubchemFP7', 'PubchemFP8', 'PubchemFP17', 'PubchemFP26', 'PubchemFP27', 'PubchemFP29', 'PubchemFP30', 'PubchemFP31', 'PubchemFP32', 'PubchemFP35', 'PubchemFP36', 'PubchemFP39', 'PubchemFP40', 'PubchemFP41', 'PubchemFP42', 'PubchemFP48', 'PubchemFP49', 'PubchemFP50', 'PubchemFP51', 'PubchemFP52', 'PubchemFP53', 'PubchemFP54', 'PubchemFP55', 'PubchemFP56', 'PubchemFP57', 'PubchemFP58', 'PubchemFP59', 'PubchemFP60', 'PubchemFP61', 'PubchemFP62', 'PubchemFP63', 'PubchemFP64', 'PubchemFP65', 'PubchemFP66', 'PubchemFP67', 'PubchemFP68', 'PubchemFP69', 'PubchemFP70', 'PubchemFP71', 'PubchemFP72', 'PubchemFP73', 'PubchemFP74', 'PubchemFP75', 'PubchemFP76', 'PubchemFP77', 'PubchemFP78', 'PubchemFP79', 'PubchemFP80', 'PubchemFP81', 'PubchemFP82', 'PubchemFP83', 'PubchemFP84', 'PubchemFP85', 'PubchemFP86', 'PubchemFP87', 'PubchemFP88', 'PubchemFP89', 'PubchemFP90', 'PubchemFP91', 'PubchemFP92', 'PubchemFP93', 'PubchemFP94', 'PubchemFP95', 'PubchemFP96', 'PubchemFP97', 'PubchemFP98', 'PubchemFP99', 'PubchemFP100', 'PubchemFP101', 'PubchemFP102', 'PubchemFP103', 'PubchemFP104', 'PubchemFP105', 'PubchemFP106', 'PubchemFP107', 'PubchemFP108', 'PubchemFP109', 'PubchemFP110', 'PubchemFP111', 'PubchemFP112', 'PubchemFP113', 'PubchemFP114', 'PubchemFP117', 'PubchemFP119', 'PubchemFP120', 'PubchemFP121', 'PubchemFP124', 'PubchemFP125', 'PubchemFP126', 'PubchemFP127', 'PubchemFP128', 'PubchemFP131', 'PubchemFP133', 'PubchemFP134', 'PubchemFP135', 'PubchemFP136', 'PubchemFP137', 'PubchemFP138', 'PubchemFP139', 'PubchemFP140', 'PubchemFP141', 'PubchemFP142', 'PubchemFP154', 'PubchemFP155', 'PubchemFP156', 'PubchemFP158', 'PubchemFP161', 'PubchemFP162', 'PubchemFP163', 'PubchemFP165', 'PubchemFP166', 'PubchemFP168', 'PubchemFP169', 'PubchemFP170', 'PubchemFP171', 'PubchemFP172', 'PubchemFP173', 'PubchemFP174', 'PubchemFP175', 'PubchemFP176', 'PubchemFP177', 'PubchemFP197', 'PubchemFP198', 'PubchemFP201', 'PubchemFP202', 'PubchemFP203', 'PubchemFP204', 'PubchemFP205', 'PubchemFP208', 'PubchemFP209', 'PubchemFP210', 'PubchemFP211', 'PubchemFP212', 'PubchemFP217', 'PubchemFP220', 'PubchemFP221', 'PubchemFP222', 'PubchemFP223', 'PubchemFP224', 'PubchemFP225', 'PubchemFP226', 'PubchemFP229', 'PubchemFP230', 'PubchemFP231', 'PubchemFP232', 'PubchemFP233', 'PubchemFP234', 'PubchemFP235', 'PubchemFP236', 'PubchemFP237', 'PubchemFP238', 'PubchemFP239', 'PubchemFP240', 'PubchemFP241', 'PubchemFP242', 'PubchemFP243', 'PubchemFP244', 'PubchemFP245', 'PubchemFP246', 'PubchemFP247', 'PubchemFP248', 'PubchemFP249', 'PubchemFP250', 'PubchemFP251', 'PubchemFP252', 'PubchemFP253', 'PubchemFP254', 'PubchemFP262', 'PubchemFP263', 'PubchemFP264', 'PubchemFP265', 'PubchemFP266', 'PubchemFP267', 'PubchemFP268', 'PubchemFP269', 'PubchemFP270', 'PubchemFP271', 'PubchemFP272', 'PubchemFP273', 'PubchemFP274', 'PubchemFP275', 'PubchemFP276', 'PubchemFP277', 'PubchemFP278', 'PubchemFP279', 'PubchemFP280', 'PubchemFP281', 'PubchemFP282', 'PubchemFP288', 'PubchemFP289', 'PubchemFP290', 'PubchemFP292', 'PubchemFP295', 'PubchemFP296', 'PubchemFP302', 'PubchemFP303', 'PubchemFP304', 'PubchemFP306', 'PubchemFP307', 'PubchemFP309', 'PubchemFP310', 'PubchemFP311', 'PubchemFP312', 'PubchemFP313', 'PubchemFP314', 'PubchemFP315', 'PubchemFP316', 'PubchemFP317', 'PubchemFP318', 'PubchemFP319', 'PubchemFP320', 'PubchemFP321', 'PubchemFP322', 'PubchemFP323', 'PubchemFP324', 'PubchemFP325', 'PubchemFP326', 'PubchemFP348', 'PubchemFP360', 'PubchemFP402', 'PubchemFP407', 'PubchemFP410', 'PubchemFP411', 'PubchemFP413', 'PubchemFP424', 'PubchemFP425', 'PubchemFP426', 'PubchemFP433', 'PubchemFP456', 'PubchemFP463', 'PubchemFP469', 'PubchemFP510', 'PubchemFP512', 'PubchemFP525', 'PubchemFP562', 'PubchemFP627', 'PubchemFP649', 'PubchemFP727', 'PubchemFP732', 'PubchemFP733', 'PubchemFP744', 'PubchemFP745', 'PubchemFP753', 'PubchemFP766', 'PubchemFP769', 'PubchemFP774', 'PubchemFP775', 'PubchemFP787', 'PubchemFP795', 'PubchemFP796', 'PubchemFP807', 'PubchemFP808', 'PubchemFP810', 'PubchemFP829', 'PubchemFP832', 'PubchemFP837', 'PubchemFP843', 'PubchemFP844', 'PubchemFP847', 'PubchemFP848', 'PubchemFP849', 'PubchemFP850', 'PubchemFP851', 'PubchemFP852', 'PubchemFP853', 'PubchemFP854', 'PubchemFP855', 'PubchemFP856', 'PubchemFP857', 'PubchemFP858', 'PubchemFP859', 'PubchemFP864', 'PubchemFP865', 'PubchemFP867', 'PubchemFP868', 'PubchemFP869', 'PubchemFP870', 'PubchemFP871', 'PubchemFP872', 'PubchemFP873', 'PubchemFP874', 'PubchemFP875', 'PubchemFP876', 'PubchemFP877', 'PubchemFP878', 'PubchemFP879', 'PubchemFP880'], axis=1)
    # Run predictions
    pred,c = ar(input_ar)    
    
    # Create Dataframe with results
    res = pd.DataFrame(names)
    res['Predicted_class'] = pred
    res['Probability'] = c
    # Save results to csv
    res.to_csv('AR-inhibitor_predictions.csv', index=False)
    
    return None
    

# Run script
if __name__ == "__main__":
    # Define current directory
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = os.getcwd()
    # Verify existence of file with SMILES
    exists = [fname for fname in os.listdir(path) if fname.endswith(".smi")]
    if exists:
        # Get predictions
        run_prediction(path)        
    else:
        raise FileNotFoundError("Input File NOT found")
