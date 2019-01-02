import os,sys
print('data ro be fana nadi ha!)
sys.exit(1)
f = open('types-sig17.txt','r')
types = f.readlines()
types = [mtype.replace('\n','') for mtype in types]
# print(len(lines))
types_list = list(set(types))
# print(lines)
# print(len(lines))



output_str = ''
continue_falg = False
for ln in open('types-depth-2018-temp.txt','r'):
	continue_falg = False
	for mtype in types_list:
		if ln.strip() == mtype.strip():
			output_str += ln
			continue_falg = True
			break

	if continue_falg==True:
		continue
	ln = ln.replace('\n','') + '(r)' + '\n'
	output_str += ln
	# print(ln)

f3 = open('types-depth-2015-10.txt','w+')
f3.write(output_str)
