import xml.dom.minidom as xmldom
import os

def GetBBoxByPath(path):
	xmlfilepath = os.path.abspath(path)
	domobj = xmldom.parse(xmlfilepath)
	elementobj = domobj.documentElement
	bndbox=elementobj.getElementsByTagName("object")[0].getElementsByTagName("bndbox")[0]
	name=elementobj.getElementsByTagName("object")[0].getElementsByTagName("name")[0]
	size=len(bndbox.childNodes)
	for node in bndbox.childNodes:
		print(node)
	print("name",name.firstChild.data)
	return GetBndbox(bndbox)

def GetBndbox(bndbox):
	xmin=int(bndbox.getElementsByTagName("xmin")[0].firstChild.data)
	ymin=int(bndbox.getElementsByTagName("ymin")[0].firstChild.data)
	xmax=int(bndbox.getElementsByTagName("xmax")[0].firstChild.data)
	ymax=int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)
	return [xmin,ymin,xmax,ymax]



print(GetBBoxByPath("C:\\Users\\25285\\Desktop\\voc\\VOCtrainval_11-May-2012\\VOCdevkit\\VOC2012\\Annotations\\2007_000027.xml"))