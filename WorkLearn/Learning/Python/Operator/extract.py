import thread
import os
import os.path
import time
threadnum=5
path = 'F:/ILSVRC2012_dataset/image_train'
dir=[]
files=[]
dirs = os.listdir(path)
for p in dirs:
	if os.path.isfile(os.path.join(path,p)):
		files.append(p)
	else:
		dir.append(p)
for p in dir:
	files.remove(p+'.tar')
print(len(files))
	




def thread_winrar(paras,root):
	for dir in paras:
		extent=os.path.splitext(dir)[1].replace('.','');
		file=os.path.splitext(dir)[0];
		print(extent)
		if extent=='zip'or extent=='rar'or extent=='7z'or extent=='tar':
			os.makedirs(root+"/"+file) 
			os.system('WinRAR e '+root+"/"+"\""+dir+"\" "+root+"/"+"\""+file+"\"")
			
unit=len(files)/threadnum
for i in range(threadnum):
	if i<(threadnum-1):
		thread.start_new_thread(thread_winrar,(files[i*unit:unit*(i+1)],path))
	else:
		thread.start_new_thread(thread_winrar,(files[i*unit:],path))			
time.sleep(50000)