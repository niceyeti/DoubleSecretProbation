"""
>>This class is lazily copied from another closed project; don't update it here<<

Keep this clean and modular, as its very useful and also quite critical for sound textual
analyses. The web is full of encodings--utf8, latin-1, ascii--and often times they get mangled,
or the code used to read them handles/stores them inappropriately. Thus input text is expected
to have escape codes for unicode of latin-1 character sets, or may even contain escaped sequences
representing those characters.

The intent of this objct is to have the cleanest object that makes a best-effort to convert unicode/latin-1
characters to their proximate ascii equivalents (if any). For instance, many websites use non-breaking space &nbsp; instead
of space char 0x20. Obviously this is not the case for all characters, eg cyrillics have no ascii equivalents,
but a best effort can be made. Also, per the cyrillics, this object is clearly english-centric, and assumes 
most input data represents english speaking content. This is not always the case, since ABLE/Sentinel
are intended for global analyses.

As far as global locales, testing on unidecode shows it works very well for approximating english phonetic
equivalents to russian, and hence may still be feasible for foregin language analyses:

	this --> "Четыре года назад, летом 2014-го, безымянные могилы на кладбищах Пскова"
	becomes --> "Chetyre goda nazad, letom 2014-go, bezymiannye mogily na kladbishchakh Pskova"

	... a near-perfect mapping of cyrillic to phonetic english characters.

"""

import sys
if sys.version_info.major < 3:
	print("ERROR use python3 instead of python 2.7")
	exit()

import util.parse_util as parse_util
from unidecode import unidecode
from html import unescape
from lxml import etree


class AsciiTextNormalizer(object):
	"""
	An object for making a best-effort attempt to clean english/western input text containing escaped/unescaped unicode/latin-1 and
	other characters to plain ascii. Its is not true that this can be done in general since these encodings encompass far more
	characters than ascii, hence its just 'best-effort'.
	"""
	def __init__(self):
		self._stripCharDict = dict([(ord(c),u" ") for c in u"!@#$%^&*()[]{};:'\",<.>?/~`-_=+|\\/"])
		self._stripChars = "!@#$%^&*()[]{};:'\",<.>?/~`-_=+|\\/"
		self._stripCharTable_BlankReplacement = str.maketrans(self._stripChars, "".join([" " for i in range(len(self._stripChars))]))
		self._stripCharTable_Deletion = str.maketrans({ord(c):None for c in self._stripChars})

	"""
	For some space-delimited string, filters out all words containing any member of junkSubstrSet as a substr.
	
	def FilterJunkStrings(self, s, junkSubstrSet):
		ws = []
		for w in s.split():
			for junk in junkSubstrSet:
				if junk not in w:
					ws.append(w)
		return " ".join(ws)
	"""

	"""
	Given an Headline object and a junk list, greedily removes words containing junk substrings (links, unicode flags, etc)

	@headline: an Headline object
	@junkSubstrSet: A set of substrs. Any word in which these substrs are detected will be removed
	
	def FilterJunkStrings(self, text, junkSubstrSet):
		hasJunk = False
	
		#outer check on the raw, unsplit strings: this averts the n^2 word iteration for most (valid) strings
		for junk in junkSubstrSet:
			if junk in text:
				hasJunk = True
	
		if hasJunk:
			text = self.FilterJunk(text, junkSubstrSet)
	
		if hasDescJunk:
			headline.Description = self.FilterJunk(headline.Description, junkSubstrSet)
		
		return headline
	"""

	def CompressWhitespace(self, s):
		s = s.replace("\n"," ").replace("\t", " ").strip()

		#logarithmically reduces the number of whitespaces, 1/2 reduction per iteration
		for i in range(4):
			s = s.replace("  "," ")

		return s

	"""
	Filters non-alphanumeric characters.
	@deleteMode: If true, then non-alphanumeric chars will be deleted rather than replaced.
	"""
	def FilterNonAlphaNum(self, s, deleteMode=True):
		if deleteMode:
			#deletes filtered characters
			return s.translate(self._stripCharTable_Deletion)
		else:
			#replaces filtered characters with blank spaces
			return s.translate(self._stripCharTable_BlankReplacement)


	def DecodeInTextLinks(self, s):
		"""
		Text from social media or other sites often contains <a> tags or other tags (<strong>, etc) within the text itself
		as a literal, especially when the text was derived from within javascript sections of sites.
		This method parses these links, removing the html tags and any of their content.
		"""

		if "<" in s and ">" in s:
			try:
				#A slick hack: it would be difficult and error-prone to try to extract the tag text content from the string. Instead,
				#enclose the string as if it were html to begin with, then use existing parse utils to remove its text content
				elementText = "<html>"+s+"</html>" #add dummy tags, purely to make the string parseable
				tree = etree.HTML(elementText)
				s = parse_util.getAllElementText(tree)
			except:
				traceback.print_exc()
				print("There was an error in DecodeHyperlinks, see previous output. Decoding skipped, string preserved as-is: "+s)

		return s

	def UnescapeUnicode(self, s):
		"""
		Resolves a specific/common nuisance with escaped unicode in python3, converting strings like '\\u003c' (literally a backslash followed by u and a literal numeral
		unicode code) back into unescaped unicode characters. So this unescapes unicode; it does not remove it.
		"""
		try:
			if "\\u" in s or "\\x" in s:
				s = bytes(s,"utf8").decode("unicode-escape")
		except:
			traceback.print_exc()

		return s

	def EncodeAscii(self, s):
		"""
		Amazingly useful method of this class, takes a string and some encoding parameters, and attempts to convert the text
		as much as possible. Only do encoding conversions here, not other text normalization (lowering, etc.).

		Sequence:
			0) unescape double escaped chars: '\\u00E0' -> '\u00E0'
			1) unescape any html/other escaped characters in string
			2) unidecode to map unicode chars to the ascii approximations
		"""
		#unescape unicode escaped chars
		s = self.UnescapeUnicode(s)
		#unescape html escaped chars
		s = unescape(s)
		try:
			s = unidecode(s)
		except:
			traceback.print_exc()
			print("Unidecode failed for: "+s)

		return s

	def NormalizeText(self, text, filterNonAlphaNum=True, deleteFiltered=False, lowercase=True):
		"""
		Primary class method, makes a best-effort attempt to convert input text to standard characters.
		Of course, note that many of the methods in this class are order-dependent; can't remove non-alpha characters, then
		decode html from within strings, for instance.

		@lowercase: If true, return text in lowercase as last step of normalization
		@filterNonAlphaNum: If true, filter all non-alphanumeric characters. This is done last, after any parsing of those characters. Note
							that this will destroy in text links: http://cnn.it/dksdjkj becomes "http cnn it dksdkjk"
		@deleteFiltered: This param only makes sense in the context of @filterNonAlphaNum = True. If both are true, then filtering will delete
					the filtered characters, rather than replacing them with spaces.
		"""

		text = self.EncodeAscii(text) # must be done before DecodeInTextLinks, to ensure \\u003c becomes '<' for tags for instance.
		text = self.DecodeInTextLinks(text)

		if filterNonAlphaNum:
			text = self.FilterNonAlphaNum(text, deleteMode=deleteFiltered)

		if lowercase:
			text = text.lower()

		text = self.CompressWhitespace(text)

		return text



