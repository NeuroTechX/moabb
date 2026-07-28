from .NDFSysParser import FolderNHF,FolderNSF,FileNEF
from .NDFSysUtility import FolderCatergory, NDFUtility
import os

def ReadNDFChannels(fodername):
    dataFolder = fodername
    flCat = NDFUtility.GetFolderCategory(dataFolder)
    events = []
    if flCat == FolderCatergory.NDFFileTypeNHF:
        FolderType = FolderCatergory.NDFFileTypeNHF
        nhfObj = FolderNHF(dataFolder)
        flObj = nhfObj
    elif flCat == FolderCatergory.NDFFileTypeNSF:
        FolderType = FolderCatergory.NDFFileTypeNSF
        flNsfObj = FolderNSF(dataFolder)
        flObj = flNsfObj
    elif flCat == FolderCatergory.NDFFileTypeNTP:
        print('Please select a *.NHF or *.NSF folder')
        return
    else:
        print('Please select a *.NHF or *.NSF folder')
        return

    return flObj.ChannelsDic


def ReadOneChannel(foderPath, patient,channelName,start,end):
    flCat = NDFUtility.GetFolderCategory(foderPath)
    if flCat == FolderCatergory.NDFFileTypeNHF:
        dataFolder = foderPath
    elif flCat == FolderCatergory.NDFFileTypeNTP:
        dataFolder = os.path.join(foderPath, patient)
    else:
        print('Please select a *.NHF or *.NSF folder')
        return
    nhfObj = FolderNHF(dataFolder)
    ret = nhfObj.readOneChannelTimeRange(channelName,start,end)

    return ret

