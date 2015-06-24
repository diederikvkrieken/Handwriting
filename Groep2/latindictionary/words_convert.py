import re
import string

# Generates a HTML MOBI source file from
# William Whitaker's 'Words' source file
# http://www.erols.com/whitaker/dictpage.zip
# Note that a few lines of the source dictionary
# file do not end with ';' if you modify regex below

dictionary = open('DICTPAGE.txt')
print '<html><body>'
for line in dictionary:
	
		# strip multiple spaces
		line = re.sub('\s+', ' ', line)
	
		entry = re.match('^#(?P<headword>.*?)[, ](?P<parts>.*?)::(?P<definition>.*?)$', line)
		print '<idx:entry>'		
		print '<idx:orth><b>' + entry.group('headword') + '</b></idx:orth>'
		print '<i>' + entry.group('parts') + '</i>'
		print entry.group('definition')
		print '<br><br>'
		print '</idx:entry>'
print '</body></html>'


dictionary.close()
		

