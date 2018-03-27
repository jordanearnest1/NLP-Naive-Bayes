import re
import csv
from sklearn.metrics import f1_score

y_true = []
y_pred = []

training_data_lst= []
training_data_token_lst = []

def tokenize(a_string):
	x = a_string.split()
	return x

with open('train.tsv','r', encoding="utf-8") as training_data_file: #dev.tsv train.tsv
	read_tsv = [line.strip().split("\t") for line in training_data_file]
	for line in read_tsv[1:]:
		training_data_lst.append(line)

total_word_lst = []
total_dict = {}
for line in training_data_lst:
	tweet_words = line[1].split()
	for word in tweet_words:
		total_word_lst.append(word)
		if word not in total_dict:
			total_dict[word] = 1
		else:
			total_dict[word] += 1
# print(total_dict)

#################################################################################  
####   Separating Data into 2 classes and counting number in each class        ##
#################################################################################

def train(training_data_lst, smoothing_alpha = 8):
	num_docs = 0
	num_hate_docs = 0
	num_non_hate_docs = 0
	hate_class_lst = []
	non_hate_class_lst = []

	for row in training_data_lst:   ##will need to exclude first row if using
		num_docs += 1
		if row[-1] is "0":
			y_true.append(int(row[-1]))  
			non_hate_class_lst.append(row[1])   ## a list with a list of strings
			num_non_hate_docs += 1
		else:
			y_true.append(int(row[-1]))  
			hate_class_lst.append(row[1])
			num_hate_docs += 1

#####################################################################
##                     Probability of each class                   ##
#####################################################################

	p_hate_class = num_hate_docs / num_docs
	p_non_hate_class =  num_non_hate_docs / num_docs

######################################################################
#    Making lists of each word and dictionaries of occurances       ##
######################################################################


# def better_tokenize(my_string):
# 	my_string = re.sub(r'[^\w\s]','', my_string)
# 	token= my_string.split()
# 	return(token)





	non_hate_word_lst = []
	nh_dict = {}
	for line in non_hate_class_lst:
		my_string = re.sub(r'[^\w\s]','', line)   ##getting rid of tokenizer
		my_string = line.lower()
		my_string = my_string.split()
		#print(string_whole_tweet)
		for word in my_string:
			if "http" not in word:   ###trying this
				non_hate_word_lst.append(word)
				if word not in nh_dict:
					nh_dict[word] = 1
				else:
					nh_dict[word] += 1
	#print(nh_dict)
	hate_word_lst = []
	hate_dict = {}

	for line in hate_class_lst:
		my_string = re.sub(r'[^\w\s]','', line)    ##getting rid of tokenizer
		my_string = line.lower()
		my_string = my_string.split()
		for word in my_string:
			if "http" not in word:   ###trying this
				hate_word_lst.append(word)
				if word not in hate_dict:
					hate_dict[word] = 1
				else:
					hate_dict[word] += 1
# so = hate_dict.items()
# sorted_hate_dict = sorted(so, key = lambda x:x[1], reverse=True)

#######################################################################
#                   Number words in a category                       ##
# ######################################################################

	num_nh_words = 0
	num_h_words = 0

	for i in nh_dict:
		num_nh_words += 1
	for i in hate_dict:
		num_h_words += 1

#######################################################################
##             Number words occurences per category                  ##
#######################################################################

	num_nh_occur = 0
	num_hate_occur = 0

	for i in nh_dict:
		num_occurences = nh_dict[i]
		num_nh_occur += num_occurences

	for i in hate_dict:
		num_occurences = hate_dict[i]
		num_hate_occur += num_occurences

######################################################################
##           Total number of occurences of every word               ##
######################################################################

	running_total_words = 0

	total_occurences_each_word ={}  ## an important dict!!!
	total_occurences = num_nh_occur + num_hate_occur

	for i in (nh_dict):
		num_occurences = nh_dict[i]
		if i in total_occurences_each_word:
			total_occurences_each_word[i] += nh_dict[i]
			running_total_words += 1
		else:
			total_occurences_each_word[i] = nh_dict[i]
			running_total_words += 1

	for i in (hate_dict):
		num_occurences = hate_dict[i]
		if i in total_occurences_each_word:
			total_occurences_each_word[i] += hate_dict[i]
			running_total_words += 1
		else:
			total_occurences_each_word[i] = hate_dict[i]
			running_total_words += 1

###################################################################
##         P of times a word occurs in a given category          ##
###################################################################

	nh_p_dict ={}
	total_occurences = num_nh_occur + num_hate_occur
	for i in (nh_dict):
		num_occurences = nh_dict[i]
		prob_occurence = ((num_occurences + smoothing_alpha)/ (int(len(non_hate_word_lst)) + smoothing_alpha * len(total_dict)))    ## or does it need to be all occurences?! 
		nh_p_dict[i] = prob_occurence
	    
	hate_p_dict ={}
	for i in (hate_dict):
		occurence_num = hate_dict[i]
		prob_occurence = ((occurence_num + smoothing_alpha)/ (int(len(hate_word_lst)) + smoothing_alpha * len(total_dict))) 
		hate_p_dict[i] = prob_occurence

	return nh_p_dict, hate_p_dict, p_non_hate_class, p_hate_class, running_total_words

trained = train(training_data_lst)
nh_p_dict = trained[0]
hate_p_dict = trained[1]
p_non_hate_class = trained[2]
p_hate_class = trained[3]
running_total_words = trained[4]

######################################################################
##          A Function to read test data, line by line              ##
######################################################################

def classify(training_data_lst, smoothing_alpha = 8): 
	num_lines = 0
	# print(training_data_lst)
	for word in training_data_lst:
		# print(word)  
		p_line_nh = 1 
		p_line_hate = 1
		if word not in nh_p_dict:
			p_line_nh *= (smoothing_alpha / (smoothing_alpha * len(total_dict)))		
		else:
			p_line_nh *= nh_p_dict[word]	##do i need smoothing here!?
			# p_line_nh *= (smoothing_alpha / (smoothing_alpha * len(total_dict)))
		if word not in hate_p_dict:
			p_line_hate *= (smoothing_alpha / (smoothing_alpha * len(total_dict)))
		else:
			p_line_hate *= hate_p_dict[word]

		posterior_p_line_nh = p_line_nh * p_non_hate_class
		posterior_p_line_hate= p_line_hate * p_hate_class

		if posterior_p_line_nh > posterior_p_line_hate:
			return  "0"
		else:
			return "1"

def better_tokenize(my_string):
	my_string = re.sub(r'[^\w\s]','', my_string)
	my_string = my_string.lower()
	token= my_string.split()
	return(token)
 
fhand = open("test_data_output.csv", 'w')
with open("test.unlabeled.tsv", encoding = "utf-8") as test_file:   ##dev.tsv # test.unlabeled.tsv
	read_tsv = [line.strip().split("\t") for line in test_file]
	# print(line[1])
	fhand.write("instance_id,class\n")
	for line in read_tsv[1:]:
		# print(line[1])
		fhand.write(line[0] + "," + classify(better_tokenize(line[1])) + "\n")
		y_pred.append(int(classify(better_tokenize(line[1]))))
fhand.close()



# score = f1_score(y_true, y_pred, average="micro")
# print(score)

# 0.538856632425 alpha 1
# 0.538856632425 alpha 2

