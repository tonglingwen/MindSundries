import xml.dom.minidom as xmldom
import os

def GetBBoxByPath(path):
	xmlfilepath = os.path.abspath(path)
	domobj = xmldom.parse(xmlfilepath)
	elementobj = domobj.documentElement
	bndbox=elementobj.getElementsByTagName("object")[0].getElementsByTagName("bndbox")[0]
	xmin=int(bndbox.getElementsByTagName("xmin")[0].firstChild.data)
	ymin=int(bndbox.getElementsByTagName("ymin")[0].firstChild.data)
	xmax=int(bndbox.getElementsByTagName("xmax")[0].firstChild.data)
	ymax=int(bndbox.getElementsByTagName("ymax")[0].firstChild.data)
	return [xmin,ymin,xmax,ymax]


print(GetBBoxByPath(r'F:\ILSVRC2012_dataset\image_other\ILSVRC_train_v2\n01440764\n01440764_18.xml'))
