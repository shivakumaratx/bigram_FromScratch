# bigram_FromScratch
Assumptions Made:

Assumptions 1: Tokens are white space separated. So, to get a list of all the tokens 
I can split the Corpus by white space. 


Assumption 2: Punctation are not words. So, the unigram will compose of all the 
tokens without punctuation. 


How to run my program:
Set up: Put the train corpus in a .txt file named train
Note: I am using python3 to run my code. 


No-smoothing:
In the cmd/terminal window enter:
python program.py train.txt 0
![Screenshot 2023-02-27 143011](https://user-images.githubusercontent.com/118697629/221678551-c5e5170e-ee4e-4b53-944a-0bff4ee6aac7.png)


Smoothing:
In the cmd/terminal window enter:

python program.py train.txt 1

![Screenshot 2023-02-27 142917](https://user-images.githubusercontent.com/118697629/221678369-7a5a4da9-47a0-4f02-83db-75ce129cb4b0.png)

Output:
The program will display the count table, probability table, and 
probability for each test sentence in the terminal and will put them in 
nosmooth.txt or smooth .txt depending on user input.
