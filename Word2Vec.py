"""
A wrapper around a word2vec implementation for experimental usage only, just playing
around with different implementations for evaluation.

The code libraries used are the property of their respective authors.


***GenSim implements a bunch of nice training data stream objects, for training from files of line-based
training sequences (sentences), and so forth. Moving stuff into ABLE/Sentinel could be as easy as
inheriting from those classes to implement ones that would read and preprocess headline objects
and so on. Use good practices and recognize that most of the etl boilerplate can probably be covered
via stream classes that perform all that internal logic.
"""

#from util.ascii_text_normalizer import AsciiTextNormalizer
import sys
import gensim

#Small class for memory-efficient line-based sequence generation from potentially large files.
#You could modify this to implement another scheme, such as sentence based streaming by splitting on periods instead of lines, etc.
#Might even be able to inherit from class File to implement a pure pythonic stream.
class FileSequenceStream(object):
	def __init__(self, fname, limit):
		"""
		@fname: The path to some file containing text which will be haphazardly broken into training sequences per-line.
		@limit: The number of sequences to generate before terminating the stream. If -1, then no limit (entire file).
		"""
		self._fname = fname
		self._limit = limit
		#self._textNormalizer = AsciiTextNormalizer()

	def __iter__(self):
		with open(self._fname, "r") as sequenceFile:
			ct = 0
			for line in sequenceFile:
				if self._limit < 0 or ct < self._limit:
					yield line.lower().strip().split()
					#yield self._textNormalizer.NormalizeText(line).split()

class FbDataFileStream(object):
	def __init__(self, fname, limit):
		"""
		This class is completely ad hoc for experimenting with how quickly gensim can build a model from a large ABLE-based fbData.py file.
		Note this is very dirty anyway and does involve any text cleaning or normalization.

		@fname: The path to some file containing text which will be haphazardly broken into training sequences per-line.
		@limit: The number of sequences to generate before terminating the stream. If -1, then no limit (entire file).
		"""
		self._fname = fname
		self._limit = limit

	def __iter__(self):
		with open(self._fname, "r") as sequenceFile:
			ct = 0
			for line in sequenceFile:
				try:
					#print("line: {}".format(line))
					fbDict = eval(line)[1]["og_object"]
					#print(str(fbDict))
					#exit()
					if self._limit < 0 or ct < self._limit:
						seq =  (fbDict["title"]+" "+fbDict["description"]).lower().split()
						#print(str(seq))
						ct += 1
						#exit()
						yield seq
					else:
						break
				except:
					pass


def isValidCmdLine():
	isValid = False

	if len(sys.argv) < 3:
		print("Insuffiient cmd line parameters")	
	elif not any("-fname=" in arg for arg in sys.argv):
		print("No fname passed")
	elif not any("-trainLimit=" in arg for arg in sys.argv):
		print("No training-example limit passed")
	elif not any("-iter=" in arg for arg in sys.argv):
		print("No n-iterations params passed")
	else:
		isValid = True

	return isValid

def usage():
	print("Usage: python3 ./Word2Vec.py\n\t-fname=[path to line-based training txt file]\n\t-trainLimit=[num training sequences to extract from file; pass -1 for no limit]")
	print("\t-iter=[num iterations]")

def main():
	if not isValidCmdLine():
		print("Insufficient/incorrect args passed, see usage.")
		usage()
		return -1

	fname = ""
	limit = -1
	numIterations = 10
	minTermFrequency = 5
	for arg in sys.argv:
		if "-fname=" in arg:
			fname = arg.split("=")[-1]
		if "-trainLimit=" in arg:
			limit = int(arg.split("=")[-1])
		if "-iter=" in arg:
			numIterations = int(arg.split("=")[-1])
		if "-minFreq=" in arg:
			minTermFrequency = int(arg.split("=")[-1])

	if "fbData.py" in fname: #this is just a hack to see how quickly large amounts of fb content can be trained
		stream = FbDataFileStream(fname, limit)	
	else:
		stream = FileSequenceStream(fname, limit)

	model = gensim.models.Word2Vec(stream, iter=numIterations, min_count=minTermFrequency)
	print("Training completed")

	if "treasureisland" in fname.lower():
		simTerms = ['hawkins', 'silver']
		mostSim = model.most_similar(positive=simTerms, topn=10)
		print(str(mostSim))
		#model.doesnt_match("breakfast cereal dinner lunch";.split())
		sim = model.similarity('hawkins', 'money')
		print("Sim, hawkins/money: {}".format(sim))
	elif "fbdata.py" in fname.lower():
		"""
		Gensim trained a 200k file of cbs fb content in about 6 minutes with minFreq=5, -iter=10
		"""

		simTerms = ['republican', 'trump']
		mostSim = model.most_similar(positive=simTerms, topn=10)
		print("Repulican/trump most similar: "+str(mostSim))
		#model.doesnt_match("breakfast cereal dinner lunch";.split())


		sim = model.similarity('trump', 'racist')
		print("Sim, trump/racist: {}".format(sim))
		sim = model.similarity('trump', 'offensive')
		print("Sim, trump/offensive: {}".format(sim))
		sim = model.similarity('clinton', 'racist')
		print("Sim, clinton/racist: {}".format(sim))
		sim = model.similarity('clinton', 'offensive')
		print("Sim, trump/offensive: {}".format(sim))

	return 0

if __name__ == "__main__":
	main()


