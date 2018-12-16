#将指定文件夹path下面的所有zip、rar、7z通过WinRAR解压到指定目录下
import os
import os.path
path = 'E:/model'              #要进行解压缩的目录
dirs = os.listdir(path)
for dir in dirs:
 extent=os.path.splitext(dir)[1].replace('.','');
 file=os.path.splitext(dir)[0];
 print(extent)
 if extent=='zip'or extent=='rar'or extent=='7z': #判断后缀名是否是这些格式       
  os.makedirs(path+"/"+file+extent)               #创建文件夹 
  os.system('WinRAR e '+path+"/"+"\""+dir+"\" "+path+"/"+"\""+file+extent+"\"") #启动WinRAR程序进行解压缩