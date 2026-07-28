from enum import Enum
import os

class FolderCatergory(Enum):
        NDFFileTypeInvalid = 0
        NDFFileTypeNTP = 1
        NDFFileTypeNHF = 2
        NDFFileTypeNSF = 3
        NDFFileTypeNVF = 4
        NDFFileTypeNDF = 5
        NDFFileTypeNEF = 6

class Subject:
    def __init__(self):
        self.participant=[]
        self.subjectID=[]
        self.folderName=[]
        self.isSensor=[]


class Host:
    def __init__(self):
        self.name = []
        self.folder = []

class Event:
    def __init__(self):
        self.Annotation = []
        self.Remark=[]
        self.EventIndex=0
        self.Onset= 0
        self.Duration = 0
        self.Color=0

class Channel:
    def __init__(self):
        self.ChannelName = []
        self.ChannelType = []
        self.TransducerType = []
        self.PreFilter = []
        self.Unit = []
        self.BytesOneSample = []
        self.SampleRate = []
        self.Bytes = []
        self.MaxDigital = []
        self.MinDigital = []
        self.MaxPhysical = []
        self.MinPhysical = []
        self.ChannelOffset = []
        self.D2PCoef = []
        self.P2DCoef = []
        self.ChannelID = -1

class Segment:
    def __init__(self):
        self.Index = -1
        self.FolderName = []
        self.IsContinous = []
        self.Start = []
        self.Duration = []
        self.Stop = []


class NDFSysCfg:
    EncodeFormat = 'gb2312'
    #EncodeFormat = 'cp437'

class NDFPageInfo:
    def __init__(self):
        self.PageIndex = 0
        self.PagePosition = 0
        self.PageRecordOffsetAr = []
        self.RecordValidLastIndex = -1

class NDFUtility:
    @staticmethod
    def GetFolderCategory(folder):
        try:
            fAr = []
            fAr += [each for each in os.listdir(folder) if each.endswith('.nfi')]
            fAr += [each for each in os.listdir(folder) if each.endswith('.NFI')]
            fArLen = len(fAr)
            if fArLen<1:
                return FolderCatergory.NDFFileTypeInvalid
            fNFI = fAr[0]
            fPath = os.path.join(folder, fNFI)
            with open(fPath,'rb') as fID:
                lenRead = 10
                rawAr = fID.read(lenRead)
                ret = FolderCatergory(rawAr[lenRead-1])
                return ret
        except:
            print("Failed to open folder %s"%folder)

    @staticmethod
    def Decode(bytesList):
        try:
            strRes = bytesList.decode(NDFSysCfg.EncodeFormat,'strict')
            strRes=strRes.rstrip('\x00')
            return strRes
        except:
            print('Failed to decode bytes!')



