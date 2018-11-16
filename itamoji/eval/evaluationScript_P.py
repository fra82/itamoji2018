'''
Created on Sep 12, 2018

@author: Francesco Ronzano
'''

import sys

import json
import codecs
import ntpath
import time

from sklearn.metrics import f1_score, confusion_matrix, classification_report, coverage_error

# Labels of 25 ITAmoji emojis
emojiLabels = []
emojiLabels.append("red_heart")
emojiLabels.append("face_with_tears_of_joy")
emojiLabels.append("smiling_face_with_heart_eyes")
emojiLabels.append("winking_face")
emojiLabels.append("smiling_face_with_smiling_eyes")
emojiLabels.append("beaming_face_with_smiling_eyes")
emojiLabels.append("grinning_face")
emojiLabels.append("face_blowing_a_kiss")
emojiLabels.append("smiling_face_with_sunglasses")
emojiLabels.append("thumbs_up")
emojiLabels.append("rolling_on_the_floor_laughing")
emojiLabels.append("thinking_face")
emojiLabels.append("blue_heart")
emojiLabels.append("winking_face_with_tongue")
emojiLabels.append("face_screaming_in_fear")
emojiLabels.append("flexed_biceps")
emojiLabels.append("face_savoring_food")
emojiLabels.append("grinning_face_with_sweat")
emojiLabels.append("loudly_crying_face")
emojiLabels.append("top_arrow")
emojiLabels.append("two_hearts")
emojiLabels.append("sun")
emojiLabels.append("kiss_mark")
emojiLabels.append("sparkles")
emojiLabels.append("rose")

# Global variables
groundTruthTraining = {}


def loadGroundTruthTrainig(groundTrugthFileFullPath):
    '''
    Load ground-truth data from ITAmoji 2018 test set with ground truth downloadable at: https://drive.google.com/file/d/1kbPGdyI3fg6oSQIjx5-91t7ZXfbyxtFu/view?usp=sharing
    '''
     
    global groundTruthTraining
    
    print("Loading ground-truth items / tweets...")
    with open(groundTrugthFileFullPath, "r") as ins:
        for line in ins:
            jsonLine = json.loads(line)
            if "tid" in jsonLine:
                groundTruthTraining[jsonLine["tid"]] = jsonLine
                
            else:
                print("ERROR WHILE LOADING LINE (GROUND-TRUTH): " + str(line))
    
    print("Loaded " + str(len(groundTruthTraining)) + " ground-truth items / tweets.")
    

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """
    pretty print for confusion matrixes
    """
    prettyPrintedCm = ""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    prettyPrintedCm += "    " + empty_cell + " "
    for label in labels:
        prettyPrintedCm += "%{0}s".format(columnwidth) % label + " "
    prettyPrintedCm += "\n"
    # Print rows
    for i, label1 in enumerate(labels):
        prettyPrintedCm += "    %{0}s".format(columnwidth) % label1 + " "
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            prettyPrintedCm += cell + " "
        prettyPrintedCm += "\n"
    
    return prettyPrintedCm


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def accuracyAtN(n, runResult):
    
    global groundTruthTraining
    
    totalPredictionCount = 0
    correctPredictionInTopNCount = 0
    
    for indexLab in range(1, 25001):
        tid = "ITAMOJI_test_" + str(indexLab)
        if tid in runResult and tid in groundTruthTraining:
            groundTruthLabel = groundTruthTraining[tid]["ground_truth_label"]
            
            totalPredictionCount = totalPredictionCount + 1
            
            predictedInTopN = False
            for indexTopN in range(1, n + 1):
                predictedTopLabel = runResult[tid]["label_" + str(indexTopN)]
                if predictedTopLabel.strip().lower() == groundTruthLabel.strip().lower():
                    predictedInTopN = True
                    break
            
            if predictedInTopN:
                correctPredictionInTopNCount = correctPredictionInTopNCount + 1
    
    return (totalPredictionCount, correctPredictionInTopNCount, float(correctPredictionInTopNCount) / float(totalPredictionCount))


def evaluateRun(fullPathToRunOutput, teamName):
    '''
    Evaluate the predictions file accessible at the path 'fullPathToRunOutput' and from the team with name 'teamName' 
    '''
    
    global groundTruthTraining
    
    runFileName = path_leaf(fullPathToRunOutput)
    
    
    # Create evaluation output file
    dateStr = str(time.strftime("%Y_%m_%d_%H_%M"))
    
    evalResultFullPath = fullPathToRunOutput.replace(".txt", "")
    evalResultFullPath = evalResultFullPath.replace(".list", "") 
    evalResultFullPath = evalResultFullPath.replace(".json", "")
    evalResultFullPath = evalResultFullPath + "_" + dateStr + "_EVALUATION_RESULTS.txt"
    resultFile = codecs.open(evalResultFullPath, "w", "utf-8")
    
    resultFile.write("#########################################################\n")
    resultFile.write("#####   ITAmoji 2018 TASK\n")
    resultFile.write("#####   @ EvalITA 2018\n")
    resultFile.write("#####   Web Site: https://sites.google.com/view/itamoji/\n")
    resultFile.write("#####       RUN EVALUATION RESULTS OF:\n")
    resultFile.write("#####       - TEAM: '" + str(teamName) + "'\n")
    resultFile.write("#####       - RUN FILE NAME: '" + str(runFileName) + "'\n")
    resultFile.write("#########################################################\n")
    resultFile.write("\n\n")
    print("Start evaluation of team: " + str(teamName) + "' run: '" + str(runFileName) + "'")

    """
    Evaluate the run...
    """
    # Load run results
    runResult = {}
    
    dataLoadingErrors = []   
    with open(fullPathToRunOutput, "r") as ins:
        for line in ins:
            jsonLine = json.loads(line)
            if "tid" in jsonLine:
                tidValue = jsonLine["tid"]
                if tidValue.startswith("b'"):
                    tidValue = tidValue.replace("b'", "")
                tidValue = tidValue.replace("'", "").strip()
                
                if tidValue in groundTruthTraining:
                    runResult[tidValue] = jsonLine
                    
                    completeLabelTOP = "label_1"
                    if not completeLabelTOP in jsonLine:
                        dataLoadingErrors.append(str(teamName) + " / " + str(runFileName) + " # ERROR: DATA LOADING FOR ITEM '" + str(tidValue) + "' > missing value for label_1\n   JSON: " + str(jsonLine))
                    
                    for indexLab in range(2, 26):
                        completeLabel = "label_" + str(indexLab)
                        if not completeLabel in jsonLine:
                            dataLoadingErrors.append(str(teamName) + " / " + str(runFileName) + " # WARNING: DATA LOADING FOR ITEM '" + str(tidValue) + "' > missing value for " + completeLabel + "\n   JSON: " + str(jsonLine))
                    
                    for indexLab in range(1, 26):
                        if jsonLine[completeLabel] is None or jsonLine[completeLabel].lower() not in emojiLabels:
                            dataLoadingErrors.append(str(teamName) + " / " + str(runFileName) + " # WARNING: DATA LOADING FOR ITEM '" + str(tidValue) + "' > not existing value for " + completeLabel + "\n   JSON: " + str(jsonLine))
                else:
                    resultFile.write(str(teamName) + " / " + str(runFileName) + " # ERROR: DATA LOADING FOR ITEM > 'tid' value not contained in ground truth.\n   JSON: " + str(jsonLine))
            else:
                resultFile.write(str(teamName) + " / " + str(runFileName) + " # ERROR: DATA LOADING FOR ITEM > 'tid' field not specified\n   JSON: " + str(jsonLine))
                
    if len(dataLoadingErrors) > 0:
        resultFile.write("\n")
        resultFile.write("The following set of ERRORS / WARNINGS OCCURRED WHILE LOADING RUN RESULTS FROM FILE:\n")
        for error in dataLoadingErrors:
            resultFile.write("   - " + error + "\n")
        resultFile.write("\n")
    resultFile.write(" > " + str(teamName) + " / " + str(runFileName) + " # Correctly loaded predictions of " + str(len(runResult)) + " training set tweets.\n") 
    
    # Create data evaluation arrays
    y_true = []
    y_pred = []
    y_true_pos = []
    y_pred_pos = []
    for indexLab in range(1, 25001):
        tid = "ITAMOJI_test_" + str(indexLab)
        if tid in runResult and tid in groundTruthTraining:
            groundTruthLabel = groundTruthTraining[tid]["ground_truth_label"]
            y_true.append(groundTruthLabel.strip().lower())
            
            predictedTopLabel = runResult[tid]["label_1"]
            y_pred.append(predictedTopLabel.strip().lower())
            
            # Add _pos vectors
            y_true_vect = [0] * 25
            groundTruthLabel_index = emojiLabels.index(groundTruthLabel.strip().lower())
            y_true_vect[groundTruthLabel_index] = float(1)
            
            y_pred_vect = [0] * 25
            for indexLab in range(0, 25):
                if "label_" + str(indexLab + 1) in runResult[tid]:
                    predictedLabel = runResult[tid]["label_" + str(indexLab + 1)]
                    predictedLabel_index = emojiLabels.index(predictedLabel.strip().lower())
                    y_pred_vect[predictedLabel_index] = float(1 / float(indexLab + 1))
            
            y_true_pos.append(y_true_vect)
            y_pred_pos.append(y_pred_vect)
            
            # print(tid + " > PREDICTED TOP LABEL (label_1): " + str(predictedTopLabel) + " > GROUND TRUTH: " + str(groundTruthLabel))
        else:
            resultFile.write("   > TID " + str(tid) + " - invalid label.\n")
    
    # Compute F-1 scores 
    resultFile.write(" > \n")
    # resultFile.write(" > RUN EVALUATION RESULTS OF TEAM: '" + str(teamName) + "' / RUN: '" + str(runFileName) + "':\n")
    
    '''
    'micro':
    Calculate metrics globally by counting the total true positives, false negatives and false positives.
    'macro':
    Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    'weighted':
    Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    '''
    macroF1 = f1_score(y_true, y_pred, labels=emojiLabels, average='macro')
    resultFile.write(" > *********************************************\n")
    resultFile.write(" > MACRO F1: " + str(macroF1) + "\n")
    resultFile.write(" >       Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.\n")
    resultFile.write(" >       IMPROTANT: MACRO F1 score is used to generate the final ranking of runs in the ITAmoji 2018 Task.\n")
    resultFile.write(" > \n")
    
    microF1 = f1_score(y_true, y_pred, labels=emojiLabels, average='micro')
    resultFile.write(" > *********************************************\n")
    resultFile.write(" > MICRO F1: " + str(microF1) + "\n")
    resultFile.write(" >       Calculate metrics globally by counting the total true positives, false negatives and false positives.\n")
    resultFile.write(" > \n")
    
    weightedF1 = f1_score(y_true, y_pred, labels=emojiLabels, average='weighted')
    resultFile.write(" > *********************************************\n")
    resultFile.write(" > WEIGHTED F1: " + str(weightedF1) + "\n")
    resultFile.write(" >       Calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.\n")
    resultFile.write(" > \n")
    
    
    # Compute coverage error
    coverageError = coverage_error(y_true_pos, y_pred_pos)
    resultFile.write(" > *********************************************\n")
    resultFile.write(" > COVERAGE ERROR: " + str(coverageError) + "\n")
    resultFile.write(" >       Compute how far we need to go through the ranked scores to cover all true labels. The best value is equal to the average number of labels in y_true per sample.\n")
    resultFile.write(" > \n")
    
    
    # Compute accuracy at n
    resultFile.write(" > *********************************************\n")
    resultFile.write(" > ACCURACY AT TOP-N\n")
    
    
    accuracyAt5 = 0
    accuracyAt10 = 0
    accuracyAt15 = 0
    accuracyAt20 = 0
    for n in range(1, 26):
        accurracyAtNresult = accuracyAtN(n, runResult)
        spaceChars = "\t\t" if n < 10 else "\t"
        resultFile.write(" > N: " + str(n) + spaceChars + str(accurracyAtNresult[2]) + " (correct prediction in top-" + str(n) + ": " + str(accurracyAtNresult[1]) + " over " + str(accurracyAtNresult[0]) + ")\n")
                
        if n == 5: accuracyAt5 = accurracyAtNresult[2]
        if n == 10: accuracyAt10 = accurracyAtNresult[2]
        if n == 15: accuracyAt15 = accurracyAtNresult[2]
        if n == 20: accuracyAt20 = accurracyAtNresult[2]
        
        
    resultFile.write(" > \n")
    
    
    # Print classification report
    resultFile.write(" > \n")
    resultFile.write(" > *********************************************\n")
    resultFile.write(" > CLASSIFICATION REPORT: \n")
    resultFile.write(classification_report(y_true, y_pred, digits=4))
    resultFile.write(" > \n")
    resultFile.write(" > \n")
    
    
    # Print confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    emojiLabelsNum = []
    resultFile.write(" > LABELS: ")
    for indx, emojiLabel in enumerate(emojiLabels):
        resultFile.write(" \n")
        resultFile.write(" >   L_" + str(indx + 1) + ": '" + emojiLabel + "'")
        emojiLabelsNum.append("L_" + str(indx + 1))
    resultFile.write(" \n")
    cmStr = print_cm(cnf_matrix, emojiLabelsNum, hide_zeroes=False, hide_diagonal=False, hide_threshold=None)
    resultFile.write(" > \n")
    resultFile.write(" > *********************************************\n")
    resultFile.write(" > CONFUSION MATRIX:\n")
    resultFile.write(cmStr)
    resultFile.write(" > \n")
    resultFile.write(" > \n")
    
    resultFile.close()
       
    return [teamName, runFileName, path_leaf(evalResultFullPath), macroF1, microF1, weightedF1, coverageError, accuracyAt5, accuracyAt10, accuracyAt15, accuracyAt20]



if __name__== "__main__":
    '''
    The script has been developed with Python 3.6 and needs scikit-learn >=0.19, since it exploits some methods provided by scikit-learn to compute
    evaluation metrics.
     
    As you can see from the last lines of the script, to run the script you have to provide two command line arguments:
    - the full local path of the test dataset with ground-truth file (downloadable at: https://drive.google.com/file/d/1kbPGdyI3fg6oSQIjx5-91t7ZXfbyxtFu/view?usp=sharing)
    - the full local path of your run results file
    
    '''
    
    global numTweetInTrainSetByEmoji
    
    pathToGroundTruthFile = sys.argv[1] # "/full/local/path/to/ITAmoji_2018_TESTdataset_v1_withGroundTruth.list"
    pathToSystemRunResultsFile = sys.argv[2] # "/full/local/path/to/run/results/run1.txt"
    
    
    # STEP 1 / 2 >>> Load ground truth from file <<<
    # One parameter equal to the absolute local path of the ITAmoji 2018 ground-truth with labels file
    # that is available for download at: https://drive.google.com/file/d/1kbPGdyI3fg6oSQIjx5-91t7ZXfbyxtFu/view?usp=sharing
    loadGroundTruthTrainig(pathToGroundTruthFile)
    
    # STEP 2 / 2 >>> Evaluate run rpediction <<<
    # The first agument is the absolutelocal path of the run prediction file while the second argument is a run/team string identifier (any non empty string).
    # This command will generate, in the same folder of the run prediction file, a file with name ending in '_EVALUATION_RESULTS.txt' containing the evaluation of the predictions 
    evaluateRun(pathToSystemRunResultsFile, "TEAM_RUN_NAME")
    
    
    