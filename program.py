# Name: Shiva Kumar
# Net id: sak220007

# Description of program: The program takes a training corpus (.txt file) and a smoothing factor (0 or 1) for its input.
# Then, the program computes a bigram count, bigram probabilty, and computes the probabilty of the test sentence as the output.
import re
import pandas as pd
import argparse


# singleSentenceArray: takes the trainCorpus and turns it into an array of sentences that are separated by spaces
def singleSentenceArray(trainCorpus):
    arraySentences = []

    for sentence in trainCorpus:

        # Remove Punctuation using regex expression
        sentence = re.sub('[^\w\s]', '', sentence)

        # Makes every word lower case
        sentence = sentence.lower()

        # Splits the sentence into array
        sentence = sentence.split()

        # Add <s> at the beginning of the array
        sentence.insert(0, "<s>")

        # Add </s> at the end of the array
        sentence.append("</s>")

        arraySentences.append(sentence)
    return arraySentences


# uniGram: Calculates the frequency of each word in the train Corpus.
# These numbers are considered unigrams and are the denominators for the probabilites.
def uniGram(corpusArray):
    # Create hashMap that counts each the frequency of each word
    uniGram = dict()
    # For every sentence: check if a word is in unigram: If it is, update count
    # if a word is not in unigram add it to unigram with count of 1
    for sentence in corpusArray:

        for word in sentence:
            if word in uniGram:
                uniGram[word] += 1
            else:
                uniGram[word] = 1
    # Returns uniGram
    return uniGram


# noSmoothBigramCount: Calculates the bigram count for the no smooth model
def noSmoothBigramCount(corpusArray, uniGram):
    # Creates all the bigram counts
    bigramCounter = dict()
    for sentence in corpusArray:
        # Start at 1 since there is no word before the start of sentence.
        for letter in range(1, len(sentence)):
            if sentence[letter-1] + " " + sentence[letter] in bigramCounter:
                bigramCounter[sentence[letter-1] +
                              " " + sentence[letter]] += 1
            else:
                bigramCounter[sentence[letter-1] + " " + sentence[letter]] = 1

    # Full BigramCount for train Corpus
    # For each word combination in the train corpus input the bigram count
    # There are a total of 3021 words in the unigram. So each word will have 3021 bigram counts.
    fullBiGramCount = dict()
    for word in uniGram:
        for nextWord in uniGram:
            if word + " " + nextWord in bigramCounter:
                fullBiGramCount[word + " " +
                                nextWord] = bigramCounter[word + " " + nextWord]
            else:
                fullBiGramCount[word + " " + nextWord] = 0
    return fullBiGramCount


# calculateNoSmoothBigramProbability :Calculates bigram Probabilites (no smoothing)
def calculateNoSmoothBigramProbability(biGramCount, uniGramCount):
    bigramProbability = dict()
    for count in biGramCount:
        # numerator is bigram count
        numerator = biGramCount[count]
        # Split the string into an array
        arrayCount = count.split()
        # Get frequency of previous word
        denominator = uniGramCount[arrayCount[0]]
        bigramProbability[count] = numerator/denominator
    return bigramProbability


# smoothingBigramCount: Creates the bigram count for smoothing elements
def smoothingBigramCount(corpusArray, uniGram):
    # Bigram Count for no smoothing
    noSmoothingBigramCount = noSmoothBigramCount(corpusArray, uniGram)

    # Add-on smoothing adds 1 to every count
    Add1SmoothingBigramCount = dict()
    for count in noSmoothingBigramCount:
        Add1SmoothingBigramCount[count] = noSmoothingBigramCount[count] + 1
    return Add1SmoothingBigramCount


# calculateSmoothBigramProbability: Calculates bigram Probabilites for Smoothing
def calculateSmoothBigramProbability(SmoothingBigramCount, uniGramCount):
    SmoothBigramProbability = dict()
    for count in SmoothingBigramCount:

        numerator = SmoothingBigramCount[count]

        # count is w[i-1] + " " + w[i]
        # arrayCount is now [w[i-1],w[i]]
        arrayCount = count.split()

        # C(w[i-1] + number of tokens)
        denominator = uniGramCount[arrayCount[0]] + len(uniGramCount)

        # Calculates Probability
        SmoothBigramProbability[count] = numerator/denominator
    return SmoothBigramProbability


# Test Sentences to test Bigram Model.
testSentence1 = "mark antony shall say i am not well , and for thy humor , i will stay at home ."
testSentence2 = "talke not of standing . publius good cheere , there is no harme intended to your person , nor to no roman else : so tell them publius"
testSentenceArray = [testSentence1, testSentence2]

# Takes the input from the Terminal.
# The input is the corpus file "name".txt followed by the smoothing factor.
# The smoothing factor can be either 0 or 1
parser = argparse.ArgumentParser()
parser.add_argument("CorpusFile", type=argparse.FileType('r'))
parser.add_argument("userInput", type=int)
Input = parser.parse_args()

# trainCorpus has the name of the Corpus file and the userInput has the smoothing factor(0,1)
trainCorpus = Input.CorpusFile
userInput = Input.userInput

# Converts the corpus dataset into an array of individual sentences.
corpusArray = singleSentenceArray(trainCorpus)

# Makes the Trained Model Unigram
uniGramCount = uniGram(corpusArray)
# If userInput ==0: Then find the no smooth trained model bigram Count
# If userInput ==1: Then find the smooth trained model bigram Count
# If userInput is anything else, quit the program.
if userInput == 0:
    trainedbiGramCount = noSmoothBigramCount(corpusArray, uniGramCount)
elif userInput == 1:
    trainedbiGramCount = smoothingBigramCount(corpusArray, uniGramCount)
else:
    print("Invalid Smoothing Value!!")
    print("Enter a 0 or 1")
    quit()


# Name the files to write the data to
texttoWrite = ''
if userInput == 0:
    texttoWrite = 'nosmooth.txt'
if userInput == 1:
    texttoWrite = 'smooth.txt'

# Open the correctly named file based on the smoothing value
with open(texttoWrite, mode='w') as writeOutput:
    for testSentence in testSentenceArray:
        # currentOuput array will contain the countTable, probabilityTable, probability of sentence.
        currentOutput = []

        testSentence = re.sub('[^\w\s]', '', testSentence)

        # Splits the test sentence into an array
        testSentence = testSentence.split()

        # probability variable will be used to calculate the probability of test sentence
        probability = 1

        # Insert <s> to the beginning of test sentence array
        testSentence.insert(0, "<s>")

        # Insert </s> to the end of test sentence array
        testSentence.append("</s>")

        # testBigramCount will contain the the bigram count for the test sentence
        testBigramCount = dict()

        # testBigramProbability will contain the bigram probabilities for each test sentence
        testBigramProbability = dict()

        # setofCurrentWords will be the column labels
        setofCurrentWords = []

        # Will be the rows labels
        setofPreviousWords = []

        # This will contain all the counts data
        allData = []

        for previousWord in range(0, len(testSentence)-1):

            # There is no word after the end of the sentence.
            if (previousWord != len(testSentence)-1):
                setofPreviousWords.append(testSentence[previousWord])
            currentData = []

            # There is no word before the start of the sentence.
            # So we start at 1
            for nextWord in range(1, len(testSentence)):
                # Only need the setofCurrentWords one time
                if previousWord == 0:
                    setofCurrentWords.append(testSentence[nextWord])
                testBigramCount[testSentence[previousWord] + " " + testSentence[nextWord]
                                ] = trainedbiGramCount[testSentence[previousWord] + " " + testSentence[nextWord]]
                currentData.append(
                    testBigramCount[testSentence[previousWord] + " " + testSentence[nextWord]])
            allData.append(currentData)

        # The max number of columns that can be printed on the terminal is 25. The second sentence is longer than 25 (including <s>)

        if (len(setofPreviousWords) > 24):
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', 500)
        # Create bigram count table
        countTable = pd.DataFrame(
            allData, setofPreviousWords, setofCurrentWords)

        # Prints countTable
        print(countTable)
        print("\n")

        # Calculates Bigram Probability for test sentence
        # No smooth Probability
        if (userInput == 0):
            testBigramProbability = calculateNoSmoothBigramProbability(
                testBigramCount, uniGramCount)
        # Smooth Probability
        if userInput == 1:
            testBigramProbability = calculateSmoothBigramProbability(
                testBigramCount, uniGramCount)

        # Create bigram Probability Table data
        allProbabilityData = []
        for previousWord in range(0, len(testSentence)-1):
            currentData = []
            for nextWord in range(1, len(testSentence)):

                currentData.append(
                    testBigramProbability[testSentence[previousWord] +
                                          " " + testSentence[nextWord]])
            allProbabilityData.append(currentData)

        # Creates bigram dataframe that will be used to write the table in .txt file
        ProbabilityTable = pd.DataFrame(
            allProbabilityData, setofPreviousWords, setofCurrentWords)

        # Prints ProbabilityTable
        print(ProbabilityTable)
        print("\n")

        # Computes Probability of test Sentence
        for currentWord in range(1, len(testSentence)):
            probability *= testBigramProbability[testSentence[currentWord -
                                                              1] + " " + testSentence[currentWord]]
        print("Probability of this sentence is ", probability)
        print("\n")

        # Append countTable,probabilityTable,probabilty to currentOutput
        currentOutput.append(countTable)
        currentOutput.append(ProbabilityTable)
        currentOutput.append(pd.DataFrame(
            ["Probability of this sentence is " + str(probability)]))

    # Print bigram counts, bigram probability, and probability of the test Sentence
        for i in range(0, len(currentOutput)):

            # The dataframe is the probability of the sentence
            if (i == 2):
                dataFrame = currentOutput[i]
                dataFrame = dataFrame.to_string(
                    header=False, index=False)
            else:
                dataFrame = currentOutput[i]
                dataFrame = dataFrame.to_string()
            writeOutput.write(dataFrame)
            writeOutput.write("\n")
writeOutput.close()
