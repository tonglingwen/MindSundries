
import os
import os.path
path = 'E:/model'
dirs = os.listdir(path)
for dir in dirs:
 extent=os.path.splitext(dir)[1].replace('.','');
 file=os.path.splitext(dir)[0];
 print(extent)
 if extent=='zip'or extent=='rar'or extent=='7z':
  os.makedirs(path+"/"+file+extent) 
  os.system('WinRAR e '+path+"/"+"\""+dir+"\" "+path+"/"+"\""+file+extent+"\"")