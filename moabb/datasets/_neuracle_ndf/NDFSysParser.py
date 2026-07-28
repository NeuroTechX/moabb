import struct,os
from .NDFSysUtility import FolderCatergory, Subject,Host,Segment,Channel,Event,NDFUtility,NDFPageInfo
import numpy as np
from .NDFDecompression import NDFDecoder


class FileNTP:

    def __init__(self,fPath):
        self.filePath = fPath
        self.parse()


    def parse(self):
        try:
            with open(self.filePath,'rb') as fPt:
                hdLen = 64
                rawAr = fPt.read(hdLen)
                isLEAr = struct.unpack('BB',rawAr[0:2])
                if isLEAr[0] == 0xfe and isLEAr[1] == 0xff:
                    isLE = True
                elif isLEAr[1] == 0xfe and isLEAr[0] == 0xff:
                    isLE = False
                else:
                    print('Failed to parse %s'%self.filePath)
                    return

                recordDate = rawAr[11:21]
                recordTime = rawAr[21:29]
                subCount = struct.unpack('B',rawAr[29:30])[0]
                isSensor = struct.unpack('B',rawAr[30:31])[0]
                examID = struct.unpack('B'*16,rawAr[31:47])

                fPt.seek(hdLen,0)
                subLen = 112
                rawArLen = subCount*subLen
                rawAr = fPt.read(rawArLen)
                subInfoAr = []
                for i in range(subCount):
                    posStart = i*subLen
                    participantInfo = rawAr[posStart:posStart+40]
                    posStart+=40
                    subjectID  = rawAr[posStart:posStart+16]
                    posStart+=16
                    vfName = rawAr[posStart:posStart+40]
                    posStart+=40
                    isSensor = rawAr[posStart:posStart+1]
                    subObj = Subject()
                    subObj.participant = participantInfo
                    subObj.subjectID = subjectID
                    subObj.folderName = vfName
                    subObj.isSensor = bool(isSensor)
                    subInfoAr[i]=subObj
                self.IsLE = isLE
                self.RecordDate = recordDate
                self.RecordTime = recordTime
                self.SubCount = subCount
                self.IsSensor = isSensor
                self.ExamID = examID
                self.SubAr = subInfoAr
                fPt.close()
        except:
            print("Failed to open file %s"%self.filePath)


class FileNHF:
    def __init__(self,fPath):
        self.filePath = fPath
        self.parse()

    def parse(self):
        try:
            with open(self.filePath,'rb') as fPt:
                hdLen = 80
                rawAr = fPt.read(hdLen)
                isLEAr = struct.unpack('BB',rawAr[0:2])
                if isLEAr[0] == 0xfe and isLEAr[1] == 0xff:
                    isLE = True
                elif isLEAr[1] == 0xfe and isLEAr[0] == 0xff:
                    isLE = False
                else:
                    print('Failed to parse %s'%self.filePath)
                    return
                recordDate = rawAr[11:21]
                recordTime = rawAr[21:29]
                hostCount =rawAr[29]
                recordCount =rawAr[30]
                subID = rawAr[31:47]
                examID = rawAr[47:63]
                hosts = []
                fPt.seek(80,0)
                hFrmLen = 96
                for i in range(hostCount):
                    rawAr = fPt.read(hFrmLen)
                    hName = rawAr[0:40]
                    hFolder = rawAr[40:80]
                    hObj = Host()
                    hObj.name = hName
                    hObj.folder = hFolder
                    hosts.append(hObj)

                self.IsLE =isLE
                self.HostCount = hostCount
                self.SubID = subID
                self.ExamID = examID
                self.RecordCount = recordCount
                self.Hosts = hosts
                self.RecordDate = recordDate
                self.RecordTime = recordTime

                fPt.close()
        except:
            print("Failed to open file %s"%self.filePath)


class FileNSF:
    def __init__(self,fPath):
        self.filePath = fPath
        self.parse()

    def parse(self):
        try:
            with open(self.filePath,'rb') as fPt:
                hdLen = 128
                rawAr = fPt.read(hdLen)
                isLEAr = struct.unpack('BB',rawAr[0:2])
                if isLEAr[0] == 0xfe and isLEAr[1] == 0xff:
                    isLE = True
                elif isLEAr[1] == 0xfe and isLEAr[0] == 0xff:
                    isLE = False
                else:
                    print('Failed to parse %s'%self.filePath)
                    return

                gender = chr(rawAr[11])
                if gender == 'M':
                    isMale = True
                elif gender =='F':
                    isMale = False


                nameInfo = rawAr[13:33]
                birthday = rawAr[33:43]
                recordDate = rawAr[43:53]
                recordTime = rawAr[53:61]

                if isLE:
                    recordDuration = struct.unpack('<I',rawAr[61:65])[0]
                    channelCount =  struct.unpack('<H',rawAr[65:67])[0]
                    segCount =  struct.unpack('<H',rawAr[67:69])[0]
                else:
                    recordDuration = struct.unpack('>I',rawAr[61:65])[0]
                    channelCount =  struct.unpack('>H',rawAr[65:67])[0]
                    segCount =  struct.unpack('>H',rawAr[67:69])[0]


                self.IsLE =isLE
                self.Name = nameInfo
                self.Birthday = birthday
                self.RecordDate = recordDate
                self.RecordTime = recordTime
                self.RecordDuration = recordDuration
                self.ChannelCount = channelCount
                self.SegCount = segCount

                channels = []
                channelsDic = {}
                fPt.seek(hdLen,0)
                chOneFrmLen = 128
                chAllFrmLen = 128*self.ChannelCount
                rawAr = fPt.read(chAllFrmLen)
                allChFrameLen = 0
                for i in range(self.ChannelCount):
                    pos = i*128
                    ch = Channel()
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.ChannelName =''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.ChannelType = ''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+8]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.TransducerType = ''.join(tmpCh)
                    pos+=8
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.PreFilter = ''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+8]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.Unit = ''.join(tmpCh).lower()
                    pos+=8
                    tmp = rawAr[pos:pos+1]
                    ch.BytesOneSample = tmp
                    pos+=1
                    tmp = rawAr[pos:pos+2]
                    if isLE:
                        ch.SampleRate = struct.unpack('<H',tmp)[0]
                    else:
                        ch.SampleRate = struct.unpack('>H',tmp)[0]
                    pos+=2
                    allChFrameLen += ch.SampleRate
                    ch.Bytes = ch.BytesOneSample * ch.SampleRate

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MaxDigital = struct.unpack('<i',tmp)[0]
                    else:
                        ch.MaxDigital = struct.unpack('>i',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MinDigital = struct.unpack('<i',tmp)[0]
                    else:
                        ch.MinDigital = struct.unpack('>i',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MaxPhysical = struct.unpack('<f',tmp)[0]
                    else:
                        ch.MaxPhysical = struct.unpack('>f',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MinPhysical = struct.unpack('<f',tmp)[0]
                    else:
                        ch.MinPhysical = struct.unpack('>f',tmp)[0]
                    pos+=4

                    ch.D2PCoef = (ch.MaxPhysical - ch.MinPhysical)/(ch.MaxDigital-ch.MinDigital)

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.ChannelOffset = struct.unpack('<i',tmp)[0]
                    else:
                        ch.ChannelOffset = struct.unpack('>i',tmp)[0]
                    pos+=4

                    channels.append(ch)
                    channelsDic[ch.ChannelName]=ch

                self.Channels = channels
                self.ChannelsDic = channelsDic

                seekPos = hdLen+128*self.ChannelCount
                fPt.seek(seekPos,0)
                segsLen = 48*self.SegCount
                rawAr = fPt.read(segsLen)
                segFrmLen = 48
                segs = []
                segsStart = []
                segsStop = []
                for i in range(self.SegCount):
                    pos = i*segFrmLen
                    seg = Segment()
                    seg.Index = i
                    seg.FolderName = rawAr[pos:pos+20]
                    pos+=20
                    seg.IsContinous = bool(rawAr[pos])
                    pos+=1
                    if self.IsLE:
                        seg.Start = struct.unpack('<i',rawAr[pos:pos+4])[0]
                        pos+=4
                        seg.Duration = struct.unpack('<i',rawAr[pos:pos+4])[0]
                        seg.Stop = seg.Start + seg.Duration
                    else:
                        seg.Start = struct.unpack('>i',rawAr[pos:pos+4])[0]
                        pos+=4
                        seg.Duration = struct.unpack('>i',rawAr[pos:pos+4])[0]
                        seg.Stop = seg.Start + seg.Duration
                    segs.append(seg)
                    segsStart.append(seg.Start)
                    segsStop.append(seg.Stop)

                self.Segments = segs
                self.SegStart = min(segsStart)
                self.SegStop = max(segsStop)

                fPt.close()
        except:
            print("Failed to open file %s"%self.filePath)


class FileNDF:
    def __init__(self,fPath):
        self.filePath = fPath
        self.parseHeader()

    def parseHeader_origin_v1(self):
        try:
            with open(self.filePath,'rb') as fPt:
                hdLen = 128
                rawAr = fPt.read(hdLen)
                isLEAr = struct.unpack('BB',rawAr[0:2])
                if isLEAr[0] == 0xfe and isLEAr[1] == 0xff:
                    isLE = True
                elif isLEAr[1] == 0xfe and isLEAr[0] == 0xff:
                    isLE = False
                else:
                    print('Failed to parse %s'%self.filePath)
                    return

                fVersion = rawAr[6]
                gender = chr(rawAr[11])
                if gender == 'M':
                    isMale = True
                elif gender =='F':
                    isMale = False


                nameInfo = rawAr[13:33]
                birthday = rawAr[33:43]
                recordDate = rawAr[43:53]
                recordTime = rawAr[53:61]

                if isLE:
                    recordDuration = struct.unpack('<I',rawAr[61:65])[0]
                    channelCount =  struct.unpack('<H',rawAr[65:67])[0]
                    singleDuration =  struct.unpack('<i',rawAr[67:71])[0]
                    maskCount =  struct.unpack('<H',rawAr[71:73])[0]
                else:
                    recordDuration = struct.unpack('>I',rawAr[61:65])[0]
                    channelCount =  struct.unpack('>H',rawAr[65:67])[0]
                    singleDuration =  struct.unpack('>i',rawAr[67:71])[0]
                    maskCount =  struct.unpack('>H',rawAr[71:73])[0]
                subID = rawAr[53:79]
                examID = rawAr[79:85]


                self.IsLE =isLE
                self.fVersion = fVersion
                self.Name = nameInfo
                self.Birthday = birthday
                self.RecordDate = recordDate
                self.RecordTime = recordTime
                self.RecordDuration = recordDuration
                self.ChannelCount = channelCount
                self.SingleDuration = singleDuration
                self.MaskCount = maskCount
                self.SubID = subID
                self.ExamID = examID

                channels = []
                channelsDic = {}
                channelSR = []
                fPt.seek(hdLen,0)
                chOneFrmLen = 128
                chAllFrmLen = 128*self.ChannelCount
                rawAr = fPt.read(chAllFrmLen)
                allChFrameLen = 0
                for i in range(self.ChannelCount):
                    pos = i*128
                    ch = Channel()
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.ChannelName =''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.ChannelType = ''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+8]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.TransducerType = ''.join(tmpCh)
                    pos+=8
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.PreFilter = ''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+8]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.Unit = ''.join(tmpCh).lower()
                    pos+=8
                    tmp = rawAr[pos]
                    ch.BytesOneSample = tmp
                    pos+=1
                    tmp = rawAr[pos:pos+2]
                    if isLE:
                        ch.SampleRate = struct.unpack('<H',tmp)[0]
                    else:
                        ch.SampleRate = struct.unpack('>H',tmp)[0]
                    pos+=2

                    ch.Bytes = ch.BytesOneSample * ch.SampleRate
                    allChFrameLen += ch.Bytes

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MaxDigital = struct.unpack('<i',tmp)[0]
                    else:
                        ch.MaxDigital = struct.unpack('>i',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MinDigital = struct.unpack('<i',tmp)[0]
                    else:
                        ch.MinDigital = struct.unpack('>i',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MaxPhysical = struct.unpack('<f',tmp)[0]
                    else:
                        ch.MaxPhysical = struct.unpack('>f',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MinPhysical = struct.unpack('<f',tmp)[0]
                    else:
                        ch.MinPhysical = struct.unpack('>f',tmp)[0]
                    pos+=4

                    ch.D2PCoef = (ch.MaxPhysical - ch.MinPhysical)/(ch.MaxDigital-ch.MinDigital)


                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.ChannelOffset = struct.unpack('<i',tmp)[0]
                    else:
                        ch.ChannelOffset = struct.unpack('>i',tmp)[0]
                    pos+=4

                    channels.append(ch)
                    channelsDic[ch.ChannelName]=ch
                    channelSR.append(ch.SampleRate)

                self.AllChFrameLen = allChFrameLen + 1 + self.MaskCount
                self.Channels = channels
                self.ChannelsDic = channelsDic
                self.ChannelSR = channelSR
                fPt.close()
        except:
            print("Failed to open file %s"%self.filePath)

    def parseHeader_v1(self,fPt):
        try:
            hdLen = 128
            rawAr = fPt.read(hdLen)
            isLEAr = struct.unpack('BB',rawAr[0:2])
            if isLEAr[0] == 0xfe and isLEAr[1] == 0xff:
                isLE = True
            elif isLEAr[1] == 0xfe and isLEAr[0] == 0xff:
                isLE = False
            else:
                print('Failed to parse %s'%self.filePath)
                return

            fVersion = rawAr[6]
            gender = chr(rawAr[11])
            if gender == 'M':
                isMale = True
            elif gender =='F':
                isMale = False


            nameInfo = rawAr[13:33]
            birthday = rawAr[33:43]
            recordDate = rawAr[43:53]
            recordTime = rawAr[53:61]

            if isLE:
                recordDuration = struct.unpack('<I',rawAr[61:65])[0]
                channelCount =  struct.unpack('<H',rawAr[65:67])[0]
                singleDuration =  struct.unpack('<i',rawAr[67:71])[0]
                maskCount =  struct.unpack('<H',rawAr[71:73])[0]
            else:
                recordDuration = struct.unpack('>I',rawAr[61:65])[0]
                channelCount =  struct.unpack('>H',rawAr[65:67])[0]
                singleDuration =  struct.unpack('>i',rawAr[67:71])[0]
                maskCount =  struct.unpack('>H',rawAr[71:73])[0]
            subID = rawAr[53:79]
            examID = rawAr[79:85]


            self.IsLE =isLE
            self.fVersion = fVersion
            self.Name = nameInfo
            self.Birthday = birthday
            self.RecordDate = recordDate
            self.RecordTime = recordTime
            self.RecordDuration = recordDuration
            self.ChannelCount = channelCount
            self.SingleDuration = singleDuration
            self.MaskCount = maskCount
            self.SubID = subID
            self.ExamID = examID

            channels = []
            channelsDic = {}
            channelSR = []
            fPt.seek(hdLen,0)
            chOneFrmLen = 128
            chAllFrmLen = 128*self.ChannelCount
            rawAr = fPt.read(chAllFrmLen)
            allChFrameLen = 0
            for i in range(self.ChannelCount):
                pos = i*128
                ch = Channel()
                tmp = rawAr[pos:pos+20]
                tmpCh = [chr(each) for each in tmp if each!=0]
                ch.ChannelName =''.join(tmpCh)
                pos+=20
                tmp = rawAr[pos:pos+20]
                tmpCh = [chr(each) for each in tmp if each!=0]
                ch.ChannelType = ''.join(tmpCh)
                pos+=20
                tmp = rawAr[pos:pos+8]
                tmpCh = [chr(each) for each in tmp if each!=0]
                ch.TransducerType = ''.join(tmpCh)
                pos+=8
                tmp = rawAr[pos:pos+20]
                tmpCh = [chr(each) for each in tmp if each!=0]
                ch.PreFilter = ''.join(tmpCh)
                pos+=20
                tmp = rawAr[pos:pos+8]
                tmpCh = [chr(each) for each in tmp if each!=0]
                ch.Unit = ''.join(tmpCh).lower()
                pos+=8
                tmp = rawAr[pos]
                ch.BytesOneSample = tmp
                pos+=1
                tmp = rawAr[pos:pos+2]
                if isLE:
                    ch.SampleRate = struct.unpack('<H',tmp)[0]
                else:
                    ch.SampleRate = struct.unpack('>H',tmp)[0]
                pos+=2

                ch.Bytes = ch.BytesOneSample * ch.SampleRate
                allChFrameLen += ch.Bytes

                tmp = rawAr[pos:pos+4]
                if isLE:
                    ch.MaxDigital = struct.unpack('<i',tmp)[0]
                else:
                    ch.MaxDigital = struct.unpack('>i',tmp)[0]
                pos+=4

                tmp = rawAr[pos:pos+4]
                if isLE:
                    ch.MinDigital = struct.unpack('<i',tmp)[0]
                else:
                    ch.MinDigital = struct.unpack('>i',tmp)[0]
                pos+=4

                tmp = rawAr[pos:pos+4]
                if isLE:
                    ch.MaxPhysical = struct.unpack('<f',tmp)[0]
                else:
                    ch.MaxPhysical = struct.unpack('>f',tmp)[0]
                pos+=4

                tmp = rawAr[pos:pos+4]
                if isLE:
                    ch.MinPhysical = struct.unpack('<f',tmp)[0]
                else:
                    ch.MinPhysical = struct.unpack('>f',tmp)[0]
                pos+=4

                ch.D2PCoef = (ch.MaxPhysical - ch.MinPhysical)/(ch.MaxDigital-ch.MinDigital)


                tmp = rawAr[pos:pos+4]
                if isLE:
                    ch.ChannelOffset = struct.unpack('<i',tmp)[0]
                else:
                    ch.ChannelOffset = struct.unpack('>i',tmp)[0]
                pos+=4

                channels.append(ch)
                channelsDic[ch.ChannelName]=ch
                channelSR.append(ch.SampleRate)

            self.AllChFrameLen = allChFrameLen + 1 + self.MaskCount
            self.Channels = channels
            self.ChannelsDic = channelsDic
            self.ChannelSR = channelSR
        except:
            print("Failed to parse file %s"%self.filePath)

    def parseHeader(self):
        try:
            with open(self.filePath,'rb') as fPt:
                hdLen = 10
                rawAr = fPt.read(hdLen)
                fVersion = rawAr[6]
                self.fVersion = fVersion

                fPt.seek(0)

                if fVersion != 0x02:
                    self.parseHeader_v1(fPt)
                    fPt.close()
                    return

                hdLen = 144
                rawAr = fPt.read(hdLen)
                isLEAr = struct.unpack('BB',rawAr[0:2])
                if isLEAr[0] == 0xfe and isLEAr[1] == 0xff:
                    isLE = True
                elif isLEAr[1] == 0xfe and isLEAr[0] == 0xff:
                    isLE = False
                else:
                    print('Failed to parse %s'%self.filePath)
                    return

                fVersion = rawAr[6]
                gender = chr(rawAr[11])
                if gender == 'M':
                    isMale = True
                elif gender =='F':
                    isMale = False


                nameInfo = rawAr[13:33]
                birthday = rawAr[33:43]
                recordDate = rawAr[43:53]
                recordTime = rawAr[53:61]

                if isLE:
                    recordDuration = struct.unpack('<I',rawAr[61:65])[0]
                    channelCount =  struct.unpack('<H',rawAr[65:67])[0]
                    singleDuration =  struct.unpack('<i',rawAr[67:71])[0]
                    maskCount =  struct.unpack('<H',rawAr[71:73])[0]
                else:
                    recordDuration = struct.unpack('>I',rawAr[61:65])[0]
                    channelCount =  struct.unpack('>H',rawAr[65:67])[0]
                    singleDuration =  struct.unpack('>i',rawAr[67:71])[0]
                    maskCount =  struct.unpack('>H',rawAr[71:73])[0]
                subID = rawAr[53:79]
                examID = rawAr[79:85]


                self.IsLE =isLE
                self.fVersion = fVersion
                self.Name = nameInfo
                self.Birthday = birthday
                self.RecordDate = recordDate
                self.RecordTime = recordTime
                self.RecordDuration = recordDuration
                self.ChannelCount = channelCount
                self.SingleDuration = singleDuration
                self.MaskCount = maskCount
                self.SubID = subID
                self.ExamID = examID

                channels = []
                channelsDic = {}
                channelSR = []
                fPt.seek(hdLen,0)
                chOneFrmLen = 128
                chAllFrmLen = 128*self.ChannelCount
                rawAr = fPt.read(chAllFrmLen)
                allChFrameLen = 0
                for i in range(self.ChannelCount):
                    pos = i*128
                    ch = Channel()
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.ChannelName =''.join(tmpCh)
                    ch.ChannelID = i
                    pos+=20
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.ChannelType = ''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+8]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.TransducerType = ''.join(tmpCh)
                    pos+=8
                    tmp = rawAr[pos:pos+20]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.PreFilter = ''.join(tmpCh)
                    pos+=20
                    tmp = rawAr[pos:pos+8]
                    tmpCh = [chr(each) for each in tmp if each!=0]
                    ch.Unit = ''.join(tmpCh).lower()
                    pos+=8
                    tmp = rawAr[pos]
                    ch.BytesOneSample = tmp
                    pos+=1
                    tmp = rawAr[pos:pos+2]
                    if isLE:
                        ch.SampleRate = struct.unpack('<H',tmp)[0]
                    else:
                        ch.SampleRate = struct.unpack('>H',tmp)[0]
                    pos+=2

                    ch.Bytes = ch.BytesOneSample * ch.SampleRate
                    allChFrameLen += ch.Bytes

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MaxDigital = struct.unpack('<i',tmp)[0]
                    else:
                        ch.MaxDigital = struct.unpack('>i',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MinDigital = struct.unpack('<i',tmp)[0]
                    else:
                        ch.MinDigital = struct.unpack('>i',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MaxPhysical = struct.unpack('<f',tmp)[0]
                    else:
                        ch.MaxPhysical = struct.unpack('>f',tmp)[0]
                    pos+=4

                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.MinPhysical = struct.unpack('<f',tmp)[0]
                    else:
                        ch.MinPhysical = struct.unpack('>f',tmp)[0]
                    pos+=4

                    ch.D2PCoef = (ch.MaxPhysical - ch.MinPhysical)/(ch.MaxDigital-ch.MinDigital)


                    tmp = rawAr[pos:pos+4]
                    if isLE:
                        ch.ChannelOffset = struct.unpack('<i',tmp)[0]
                    else:
                        ch.ChannelOffset = struct.unpack('>i',tmp)[0]
                    pos+=4

                    channels.append(ch)
                    channelsDic[ch.ChannelName]=ch
                    channelSR.append(ch.SampleRate)

                self.AllChFrameLen = allChFrameLen + 1 + self.MaskCount
                self.Channels = channels
                self.ChannelsDic = channelsDic
                self.ChannelSR = channelSR

                arLen = 68
                rawAr = fPt.read(arLen)

                tmp = rawAr[0:4]
                if isLE:
                    self.PageItemCount = struct.unpack('<i',tmp)[0]
                else:
                    self.PageItemCount = struct.unpack('>i',tmp)[0]

                pagesPosition = []
                PAGECOUNT = 8
                pageIndex = 0
                for i in range(PAGECOUNT):
                    pos = i*8 + 4
                    tmp = rawAr[pos:pos+8]
                    if isLE:
                        pageiPos = struct.unpack('<Q',tmp)[0]
                    else:
                        pageiPos= struct.unpack('>Q',tmp)[0]

                    if pageiPos != 0:
                        pageIndex = i
                        pagesPosition.append(pageiPos)
                    else:
                        break



                self.PageUseCount = pageIndex+1
                self.PageOffsets = pagesPosition

                fPt.close()
        except:
            print("Failed to open file %s"%self.filePath)

    def readAll_v1(self):
        try:
            with open(self.filePath,'rb') as fPt:
                chs = self.Channels
                duration = self.RecordDuration
                chCount = self.ChannelCount
                bytesFrame = self.AllChFrameLen
                chNameDataDic = {}
                for i in range(chCount):
                    chNameTmp = chs[i].ChannelName
                    rowTmp = chs[i].SampleRate
                    colTmp = duration
                    #chNameDataDic[chNameTmp] = [[0 for x in range(colTmp)] for y in range(rowTmp)]
                    chNameDataDic[chNameTmp] = []

                bOrder = 'little'
                bOrderLabel = '<'
                if(not self.IsLE):
                    bOrder='big'
                    bOrderLabel = '>'
                for s in range(duration):
                    offset = 128+chCount*128+s* bytesFrame+1+self.MaskCount
                    fPt.seek(offset,0)
                    for j in range(chCount):
                        chTmp = chs[j]
                        chname = chTmp.ChannelName
                        width = chTmp.BytesOneSample
                        sCount = chTmp.SampleRate
                        chRawData = fPt.read(sCount*width)
                        if chTmp.MinDigital >= 0 and width<4:
                            #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder,signed=False) for v in range(sCount)]
                            if width == 3:
                                dtU = np.dtype('u1')
                                chDataD_U = np.frombuffer(chRawData,dtU)
                                chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                                chDataD_U_rsh = chDataD_U_rsh.astype(np.int32)
                                if(self.IsLE):
                                    #chDataAbs = abs(chDataD_I_rsh[:,2])
                                    #coef = chDataD_I_rsh[:,2]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,2]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                                else:
                                    #chDataAbs = abs(chDataD_I_rsh[:,0])
                                    #coef = chDataD_I_rsh[:,0]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,0]
                                    #chDataDT = chDataDT*coef
                                chDataD = np.reshape(chDataDT,(-1,1))
                            else:
                                fStr = '{0}u{1}'.format(bOrderLabel,width)
                                dt = np.dtype(fStr)
                                chDataD = np.frombuffer(chRawData,dt)
                                chDataD = np.double(chDataD)
                        else:
                            #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder,signed=True) for v in range(sCount)]
                            if width == 3:
                                dtU = np.dtype('u1')
                                dtI = np.dtype('i1')
                                chDataD_U = np.frombuffer(chRawData,dtU)
                                chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                                chDataD_I = np.frombuffer(chRawData,dtI)
                                chDataD_I_rsh = np.reshape(chDataD_I,(-1,3))
                                chDataD_U_rsh = chDataD_U_rsh.astype(np.int32)
                                chDataD_I_rsh = chDataD_I_rsh.astype(np.int32)
                                if(self.IsLE):
                                    #chDataAbs = abs(chDataD_I_rsh[:,2])
                                    #coef = chDataD_I_rsh[:,2]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,2]

                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                                else:
                                    #chDataAbs = abs(chDataD_I_rsh[:,0])
                                    #coef = chDataD_I_rsh[:,0]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,0]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                            else:
                                fStr = '{0}i{1}'.format(bOrderLabel,width)
                                dt = np.dtype(fStr)
                                chDataD = np.frombuffer(chRawData,dt)
                                chDataD = np.double(chDataD)
                        #chDataDNp = np.array(chDataD)
                        chDataDNp = np.squeeze(chDataD)
                        #chDataDNp = chDataD
                        chDataPNp = chTmp.MinPhysical + chTmp.D2PCoef*(chDataDNp-chTmp.MinDigital)
                        if chTmp.Unit == 'uv':
                            pass
                        elif chTmp.Unit == 'mv':
                            chDataPNp = chDataPNp*1.0e3
                        elif chTmp.Unit == 'v':
                            chDataPNp = chDataPNp*1.0e6
                        chNameDataDic[chname].append(chDataPNp)

                        #chdataP = [chTmp.MinPhysical + chTmp.D2PCoef*(v-chTmp.MinDigital) for v in chDataD]
                        # if chTmp.Unit == 'uV' or chTmp.Unit == 'uv':
                        #     chdataP = [v*1.0e-6 for v in chdataP]
                        # elif chTmp.Unit == 'mV' or chTmp.Unit == 'mv':
                        #     chdataP = [v*1.0e-3 for v in chdataP]
                        #chdataP = range(sCount)
                        #chdataP = [0 for v in range(sCount)]
                        #chNameDataDic[chname].append(chdataP)
                fPt.close()
                return chNameDataDic
        except:
            print('Failed to read data from %s'%self.filePath)

    def decodedDataD2P(self,dataDec,sampleMax,chNameDataDic):
        chs = self.Channels
        duration = self.RecordDuration
        chCount = self.ChannelCount
        bytesFrame = self.AllChFrameLen;

        for j in range(chCount):
            chTmp = chs[j]
            chname = chTmp.ChannelName
            width = chTmp.BytesOneSample
            sCount = chTmp.SampleRate
            chDataD = dataDec[j*sampleMax:j*sampleMax+sCount]
            chDataDNp = np.squeeze(chDataD)
            #chDataDNp = chDataD
            chDataPNp = chTmp.MinPhysical + chTmp.D2PCoef*(chDataDNp-chTmp.MinDigital)
            if chTmp.Unit == 'uv':
                pass
            elif chTmp.Unit == 'mv':
                chDataPNp = chDataPNp * 1.0e3
            elif chTmp.Unit == 'v':
                chDataPNp = chDataPNp * 1.0e6
            if chname in chNameDataDic:
                chNameDataDic[chname].append(chDataPNp)

        return chNameDataDic

    def rawDataD2P(self,dataDigitalRaw,chNameDataDic):
        chs = self.Channels
        duration = self.RecordDuration
        chCount = self.ChannelCount
        bytesFrame = self.AllChFrameLen;
        bOrder = 'little'
        bOrderLabel = '<'
        if(not self.IsLE):
            bOrder='big'
            bOrderLabel = '>'
        sIndex = 0
        for j in range(chCount):
            chTmp = chs[j]
            chname = chTmp.ChannelName
            width = chTmp.BytesOneSample
            sCount = chTmp.SampleRate
            chBytesCount = sCount*width
            chRawData = dataDigitalRaw[sIndex:sIndex+chBytesCount]
            sIndex += chBytesCount
            if chTmp.MinDigital >= 0 and width<4:
                if width == 3:
                    dtU = np.dtype('u1')
                    chDataD_U = np.frombuffer(chRawData,dtU)
                    chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                    chDataD_U_rsh=chDataD_U_rsh.astype(np.int32)
                    if(self.IsLE):
                        chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,2]
                        chDataD = np.reshape(chDataDT,(-1,1))
                    else:
                        chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,0]
                    chDataD = np.reshape(chDataDT,(-1,1))
                else:
                    fStr = '{0}u{1}'.format(bOrderLabel,width)
                    dt = np.dtype(fStr)
                    chDataD = np.frombuffer(chRawData,dt)
                    chDataD = np.double(chDataD)
            else:
                            #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder,signed=True) for v in range(sCount)]
                if width == 3:
                    dtU = np.dtype('u1')
                    dtI = np.dtype('i1')
                    chDataD_U = np.frombuffer(chRawData,dtU)
                    chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                    chDataD_I = np.frombuffer(chRawData,dtI)
                    chDataD_I_rsh = np.reshape(chDataD_I,(-1,3))
                    chDataD_U_rsh=chDataD_U_rsh.astype(np.int32)
                    chDataD_I_rsh=chDataD_I_rsh.astype(np.int32)
                    if(self.IsLE):
                        chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,2]
                        chDataD = np.reshape(chDataDT,(-1,1))
                    else:
                        chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,0]
                        chDataD = np.reshape(chDataDT,(-1,1))
                else:
                    fStr = '{0}i{1}'.format(bOrderLabel,width)
                    dt = np.dtype(fStr)
                    chDataD = np.frombuffer(chRawData,dt)
                    chDataD = np.double(chDataD)
                chDataDNp = np.squeeze(chDataD)
                chDataPNp = chTmp.MinPhysical + chTmp.D2PCoef*(chDataDNp-chTmp.MinDigital)
                if chTmp.Unit == 'uv':
                    pass
                elif chTmp.Unit == 'mv':
                    chDataPNp = chDataPNp * 1.0e3
                elif chTmp.Unit == 'v':
                    chDataPNp = chDataPNp * 1.0e6
                chNameDataDic[chname].append(chDataPNp)

        return chNameDataDic

    def getAllFileOffset(self,fPointer):
        ChLen  = 128
        HdLen = 144
        PageRCountLen = 4
        PageLen = 64
        fOffset = self.ChannelCount * ChLen + HdLen + PageRCountLen + PageLen
        fPointer.seek(fOffset,0)
        arLen = self.PageItemCount*4
        pInfoAr = []
        for i in range(self.PageUseCount):
            rawAr = fPointer.read(arLen)
            nPInfo = NDFPageInfo()
            nPInfo.PageIndex = i
            nPInfo.PagePosition = self.PageOffsets[i]
            if self.IsLE:
                nPInfo.PageRecordOffsetAr = struct.unpack('<{}I'.format(self.PageItemCount),rawAr)
            else:
                nPInfo.PageRecordOffsetAr= struct.unpack('>{}I'.format(self.PageItemCount),rawAr)
            nPInfo.RecordValidLastIndex = len(nPInfo.PageRecordOffsetAr)
            pInfoAr.append(nPInfo)

        pageValidCount = len(pInfoAr) - 1
        pInfoLast = pInfoAr[pageValidCount]
        prOffsetAr = pInfoLast.PageRecordOffsetAr
        rCount = len(prOffsetAr)
        rLatIndex = -1
        for i in range(rCount):
            if prOffsetAr[i] == 0 and i>0:
                break
            else:
                rLatIndex = i

        pInfoAr[pageValidCount].RecordValidLastIndex = rLatIndex


        self.PageRocordOffset = pInfoAr
        return pInfoAr

    def readAll(self):
        if self.fVersion == 0x01:
            dataAllRes = self.readAll_v1()
            return dataAllRes

        try:
            with open(self.filePath,'rb') as fPt:
                chs = self.Channels
                duration = self.RecordDuration
                chCount = self.ChannelCount
                bytesFrame = self.AllChFrameLen
                chNameDataDic = {}
                sampleMax = 0
                for i in range(chCount):
                    chNameTmp = chs[i].ChannelName
                    rowTmp = chs[i].SampleRate
                    sampleMax = rowTmp if rowTmp>sampleMax else sampleMax
                    colTmp = duration
                    #chNameDataDic[chNameTmp] = [[0 for x in range(colTmp)] for y in range(rowTmp)]
                    chNameDataDic[chNameTmp] = []


                bOrder = 'little'
                bOrderLabel = '<'
                if(not self.IsLE):
                    bOrder='big'
                    bOrderLabel = '>'

                pageAllAr = self.getAllFileOffset(fPt)
                dataHeaderLen = 13
                pageCoursor = 0
                pageRcordCursor = 0
                self.huffmanDicPos = -1
                decoder = NDFDecoder()
                for s in range(duration):
                    pageIn = s//self.PageItemCount
                    recordIn = s%self.PageItemCount
                    if pageIn>self.PageUseCount:
                        break
                    prInfo = self.PageRocordOffset[pageIn]
                    if recordIn >prInfo.RecordValidLastIndex:
                        break

                    offset = prInfo.PagePosition+prInfo.PageRecordOffsetAr[recordIn]
                    fPt.seek(offset,0)
                    rBuffer = fPt.read(dataHeaderLen)
                    if self.IsLE:
                        dataHeaderInfo = struct.unpack('<QBI',rBuffer)
                    else:
                        dataHeaderInfo = struct.unpack('>QBI',rBuffer)

                    hDicPos = dataHeaderInfo[0]
                    comprOption = dataHeaderInfo[1]
                    dtLen = dataHeaderInfo[2]
                    rawDataBuffer = fPt.read(dtLen)
                    if comprOption == 7:
                        #huffman decoding->varaint decoding->zigzag decoding->diff reversing
                        if self.huffmanDicPos != hDicPos:
                            rLen = 1024*4 + 2
                            fPt.seek(hDicPos,0)
                            hDicArRaw = fPt.read(rLen)
                            if self.IsLE:
                                hFreqAr = struct.unpack('<H{0}I'.format(1024),hDicArRaw)
                            else:
                                hFreqAr = struct.unpack('>H{0}I'.format(1024),hDicArRaw)

                            self.hTree = decoder.BuildHuffmanTree(hFreqAr[1:257])
                        dataDec = decoder.HVZD(rawDataBuffer,self.hTree,chCount,sampleMax)
                        self.decodedDataD2P(dataDec,sampleMax,chNameDataDic)
                    elif comprOption == 6:
                        #
                        dataDec = decoder.VZD(rawDataBuffer,chCount,sampleMax)
                        self.decodedDataD2P(dataDec,sampleMax,chNameDataDic)
                    else:
                        self.rawDataD2P(rawDataBuffer,chNameDataDic)

                fPt.close()
                return chNameDataDic
        except:
            print('Failed to read data from %s'%self.filePath)

    def readTimeRange_v1(self,start,duration):
        try:
            with open(self.filePath,'rb') as fPt:
                chs = self.Channels
                duration = self.RecordDuration if self.RecordDuration<duration else duration
                chCount = self.ChannelCount
                bytesFrame = self.AllChFrameLen
                chNameDataDic = {}
                for i in range(chCount):
                    chNameTmp = chs[i].ChannelName
                    rowTmp = chs[i].SampleRate
                    colTmp = duration
                    chNameDataDic[chNameTmp] = []

                bOrder = 'little'
                bOrderLabel = '<'
                if(not self.IsLE):
                    bOrder='big'
                    bOrderLabel = '>'

                # bOrder = 'little'
                # if(not self.IsLE):
                #     bOrder='big'
                for s in range(duration):
                    offset = 128+chCount*128+start*bytesFrame+s* bytesFrame+1+self.MaskCount
                    fPt.seek(offset,0)
                    for j in range(chCount):
                        chTmp = chs[j]
                        chname = chTmp.ChannelName
                        width = chTmp.BytesOneSample
                        sCount = chTmp.SampleRate
                        chRawData = fPt.read(sCount*width)
                        #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder) for v in range(sCount)]
                        if chTmp.MinDigital >= 0 and width<4:
                            #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder,signed=False) for v in range(sCount)]
                            if width == 3:
                                dtU = np.dtype('u1')
                                chDataD_U = np.frombuffer(chRawData,dtU)
                                chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                                chDataD_U_rsh=chDataD_U_rsh.astype(np.int32)
                                if(self.IsLE):
                                    #chDataAbs = abs(chDataD_I_rsh[:,2])
                                    #coef = chDataD_I_rsh[:,2]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,2]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                                else:
                                    #chDataAbs = abs(chDataD_I_rsh[:,0])
                                    #coef = chDataD_I_rsh[:,0]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,0]
                                    #chDataDT = chDataDT*coef
                                chDataD = np.reshape(chDataDT,(-1,1))
                            else:
                                fStr = '{0}u{1}'.format(bOrderLabel,width)
                                dt = np.dtype(fStr)
                                chDataD = np.frombuffer(chRawData,dt)
                                chDataD = np.double(chDataD)
                        else:
                            #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder,signed=True) for v in range(sCount)]
                            if width == 3:
                                dtU = np.dtype('u1')
                                dtI = np.dtype('i1')
                                chDataD_U = np.frombuffer(chRawData,dtU)
                                chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                                chDataD_I = np.frombuffer(chRawData,dtI)
                                chDataD_I_rsh = np.reshape(chDataD_I,(-1,3))
                                chDataD_U_rsh=chDataD_U_rsh.astype(np.int32)
                                chDataD_I_rsh=chDataD_I_rsh.astype(np.int32)
                                if(self.IsLE):
                                    #chDataAbs = abs(chDataD_I_rsh[:,2])
                                    #coef = chDataD_I_rsh[:,2]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,2]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                                else:
                                    #chDataAbs = abs(chDataD_I_rsh[:,0])
                                    #coef = chDataD_I_rsh[:,0]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,0]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                            else:
                                fStr = '{0}i{1}'.format(bOrderLabel,width)
                                dt = np.dtype(fStr)
                                chDataD = np.frombuffer(chRawData,dt)
                                chDataD = np.double(chDataD)

                        #chdataP = [chTmp.MinPhysical + chTmp.D2PCoef*(v-chTmp.MinDigital) for v in chDataD]
                        chDataDNp = np.squeeze(chDataD)
                        chDataDNp = chTmp.MinPhysical + chTmp.D2PCoef*(chDataDNp-chTmp.MinDigital)
                        if chTmp.Unit == 'uv':
                            pass
                        elif chTmp.Unit == 'mv':
                            chDataPNp = chDataPNp * 1.0e3
                        elif chTmp.Unit == 'v':
                            chDataPNp = chDataPNp * 1.0e6
                        chNameDataDic[chname].append(chDataDNp)
                fPt.close()
                return chNameDataDic
        except:
            print('Failed to read data from %s'%self.filePath)

    def getOffsetsByTime(self,fPointer,start,duration):
        ChLen  = 128
        HdLen = 144
        PageRCountLen = 4
        PageLen = 64
        fOffset = self.ChannelCount * ChLen + HdLen + PageRCountLen + PageLen
        timeAr = np.arange(start,start+duration)
        pageAr = timeAr // self.PageItemCount
        pageRecordAr = timeAr % self.PageItemCount
        fOffset += self.PageItemCount*4*pageAr[0]
        fOffset += 4*pageRecordAr[0]
        fPointer.seek(fOffset,0)
        arLen = duration*4
        rawAr = fPointer.read(arLen)
        if self.IsLE:
            RecordOffsetAr = struct.unpack('<{}I'.format(duration),rawAr)
        else:
            RecordOffsetAr= struct.unpack('>{}I'.format(duration),rawAr)

        pagePosAr = np.array(self.PageOffsets)
        recordPosAr = np.array(RecordOffsetAr)
        recordPos = recordPosAr + pagePosAr[pageAr]
        return recordPos

    def readTimeRange(self,start,duration):
        if self.fVersion == 0x01:
            dataRes = self.readTimeRange_v1(start,duration)
            return dataRes
        try:
            with open(self.filePath,'rb') as fPt:
                chs = self.Channels
                duration = self.RecordDuration if self.RecordDuration<duration else duration
                chCount = self.ChannelCount
                bytesFrame = self.AllChFrameLen;
                chNameDataDic = {}
                sampleMax = 0
                for i in range(chCount):
                    chNameTmp = chs[i].ChannelName
                    rowTmp = chs[i].SampleRate
                    sampleMax = rowTmp if rowTmp>sampleMax else sampleMax
                    colTmp = duration
                    chNameDataDic[chNameTmp] = []

                bOrder = 'little'
                bOrderLabel = '<'
                if(not self.IsLE):
                    bOrder='big'
                    bOrderLabel = '>'

                fposAr = self.getOffsetsByTime(fPt,start,duration)
                dataHeaderLen = 13
                self.huffmanDicPos = -1
                decoder = NDFDecoder()
                for s in range(duration):
                    offset = fposAr[s]
                    fPt.seek(offset,0)
                    rBuffer = fPt.read(dataHeaderLen)
                    if self.IsLE:
                        dataHeaderInfo = struct.unpack('<QBI',rBuffer)
                    else:
                        dataHeaderInfo = struct.unpack('>QBI',rBuffer)
                    hDicPos = dataHeaderInfo[0]
                    comprOption = dataHeaderInfo[1]
                    dtLen = dataHeaderInfo[2]
                    rawDataBuffer = fPt.read(dtLen)
                    if comprOption == 7:
                        #huffman decoding->varaint decoding->zigzag decoding->diff reversing
                        if self.huffmanDicPos != hDicPos:
                            rLen = 1024*4 + 2
                            fPt.seek(hDicPos,0)
                            hDicArRaw = fPt.read(rLen)
                            if self.IsLE:
                                hFreqAr = struct.unpack('<H{0}I'.format(1024),hDicArRaw)
                            else:
                                hFreqAr = struct.unpack('>H{0}I'.format(1024),hDicArRaw)

                            self.hTree = decoder.BuildHuffmanTree(hFreqAr[1:257])
                        dataDec = decoder.HVZD(rawDataBuffer,self.hTree,chCount,sampleMax)
                        self.decodedDataD2P(dataDec,sampleMax,chNameDataDic)
                    elif comprOption == 6:
                        #
                        dataDec = decoder.VZD(rawDataBuffer,chCount,sampleMax)
                        self.decodedDataD2P(dataDec,sampleMax,chNameDataDic)
                    else:
                        self.rawDataD2P(rawDataBuffer,chNameDataDic)

                fPt.close()
                return chNameDataDic
        except:
            print('Failed to read data from %s'%self.filePath)

    def readChannelRange_v1(self,start,duration,chList):
        try:
            with open(self.filePath,'rb') as fPt:
                chDic = self.ChannelsDic
                duration = self.RecordDuration
                chCountR = len(chList)
                chCount = self.ChannelCount
                bytesFrame = self.AllChFrameLen;
                chNameDataDic = {}
                for i in range(chCountR):
                    chNameTmp = chList[i]
                    chObj = chDic[chNameTmp]
                    rowTmp = chObj.SampleRate
                    colTmp = duration
                    chNameDataDic[chNameTmp] = []

                bOrder = 'little'
                bOrderLabel = '<'
                if(not self.IsLE):
                    bOrder='big'
                    bOrderLabel = '>'

                for s in range(duration):
                    for j in range(chCountR):
                        chNameTmp = chList[j]
                        chObj = chDic[chNameTmp]
                        offset = 128+chCount*128+start*bytesFrame+s* bytesFrame+1+self.MaskCount
                        fPt.seek(offset,0)
                        width = chObj.BytesOneSample
                        sCount = chObj.SampleRate
                        chRawData = fPt.read(sCount*width)
                        if chObj.MinDigital >= 0 and width<4:
                            #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder,signed=False) for v in range(sCount)]
                            if width == 3:
                                dtU = np.dtype('u1')
                                chDataD_U = np.frombuffer(chRawData,dtU)
                                chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                                chDataD_U_rsh=chDataD_U_rsh.astype(np.int32)
                                if(self.IsLE):
                                    #chDataAbs = abs(chDataD_I_rsh[:,2])
                                    #coef = chDataD_I_rsh[:,2]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,2]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                                else:
                                    #chDataAbs = abs(chDataD_I_rsh[:,0])
                                    #coef = chDataD_I_rsh[:,0]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_U_rsh[:,0]
                                    #chDataDT = chDataDT*coef
                                chDataD = np.reshape(chDataDT,(-1,1))
                            else:
                                fStr = '{0}u{1}'.format(bOrderLabel,width)
                                dt = np.dtype(fStr)
                                chDataD = np.frombuffer(chRawData,dt)
                                chDataD = np.double(chDataD)
                        else:
                            #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder,signed=True) for v in range(sCount)]
                            if width == 3:
                                dtU = np.dtype('u1')
                                dtI = np.dtype('i1')
                                chDataD_U = np.frombuffer(chRawData,dtU)
                                chDataD_U_rsh = np.reshape(chDataD_U,(-1,3))
                                chDataD_I = np.frombuffer(chRawData,dtI)
                                chDataD_I_rsh = np.reshape(chDataD_I,(-1,3))
                                chDataD_U_rsh=chDataD_U_rsh.astype(np.int32)
                                chDataD_I_rsh=chDataD_I_rsh.astype(np.int32)
                                if(self.IsLE):
                                    #chDataAbs = abs(chDataD_I_rsh[:,2])
                                    #coef = chDataD_I_rsh[:,2]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,0]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,2]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                                else:
                                    #chDataAbs = abs(chDataD_I_rsh[:,0])
                                    #coef = chDataD_I_rsh[:,0]/chDataAbs
                                    chDataDT = chDataD_U_rsh[:,2]+256*chDataD_U_rsh[:,1]+65536*chDataD_I_rsh[:,0]
                                    #chDataDT = chDataDT*coef
                                    chDataD = np.reshape(chDataDT,(-1,1))
                            else:
                                fStr = '{0}i{1}'.format(bOrderLabel,width)
                                dt = np.dtype(fStr)
                                chDataD = np.frombuffer(chRawData,dt)
                                chDataD = np.double(chDataD)

                        chData1DNp = np.squeeze(chDataD)
                        chDataDNp = chObj.MinPhysical + chObj.D2PCoef*(chData1DNp-chObj.MinDigital)
                        #chDataD = [int.from_bytes(chRawData[width*v:(v+1)*width],bOrder) for v in range(sCount)]
                        #chdataP = [chObj.MinPhysical + chObj.D2PCoef*(v-chObj.MinDigital) for v in chDataD]
                        if chObj.Unit == 'uv':
                            pass
                        elif chObj.Unit == 'mv':
                            chDataPNp = chDataPNp * 1.0e3
                        elif chObj.Unit == 'v':
                            chDataPNp = chDataPNp * 1.0e6
                        chNameDataDic[chNameTmp].append(chDataDNp)
                fPt.close()
                return chNameDataDic
        except:
            print('Failed to read data from %s'%self.filePath)

    def readChannelRange(self,start,duration,chList):
        if self.fVersion == 0x01:
            dataRes = self.readChannelRange_v1(start,duration,chList)
            return dataRes
        try:
            with open(self.filePath,'rb') as fPt:
                chDic = self.ChannelsDic
                duration = self.RecordDuration
                chCountR = len(chList)
                bytesFrame = self.AllChFrameLen;
                chNameDataDic = {}
                sampleMax = 0
                for i in range(chCountR):
                    chNameTmp = chList[i]
                    chObj = chDic[chNameTmp]
                    chNameDataDic[chNameTmp] = []
                chs = self.Channels
                chCount = self.ChannelCount
                for i in range(chCount):
                    chNameTmp = chs[i].ChannelName
                    rowTmp = chs[i].SampleRate
                    sampleMax = rowTmp if rowTmp>sampleMax else sampleMax

                fposAr = self.getOffsetsByTime(fPt,start,duration)
                dataHeaderLen = 13
                self.huffmanDicPos = -1
                decoder = NDFDecoder() #ChannelID

                bOrder = 'little'
                bOrderLabel = '<'
                if(not self.IsLE):
                    bOrder='big'
                    bOrderLabel = '>'

                for s in range(duration):
                    offset = fposAr[s]
                    fPt.seek(offset,0)
                    rBuffer = fPt.read(dataHeaderLen)
                    if self.IsLE:
                        dataHeaderInfo = struct.unpack('<QBI',rBuffer)
                    else:
                        dataHeaderInfo = struct.unpack('>QBI',rBuffer)
                    hDicPos = dataHeaderInfo[0]
                    comprOption = dataHeaderInfo[1]
                    dtLen = dataHeaderInfo[2]
                    rawDataBuffer = fPt.read(dtLen)
                    if comprOption == 7:
                        #huffman decoding->varaint decoding->zigzag decoding->diff reversing
                        if self.huffmanDicPos != hDicPos:
                            rLen = 1024*4 + 2
                            fPt.seek(hDicPos,0)
                            hDicArRaw = fPt.read(rLen)
                            if self.IsLE:
                                hFreqAr = struct.unpack('<H{0}I'.format(1024),hDicArRaw)
                            else:
                                hFreqAr = struct.unpack('>H{0}I'.format(1024),hDicArRaw)

                            self.hTree = decoder.BuildHuffmanTree(hFreqAr[1:257])
                        dataDec = decoder.HVZD(rawDataBuffer,self.hTree,chCount,sampleMax)
                        self.decodedDataD2P(dataDec,sampleMax,chNameDataDic)
                    elif comprOption == 6:
                        #
                        dataDec = decoder.VZD(rawDataBuffer,chCount,sampleMax)
                        self.decodedDataD2P(dataDec,sampleMax,chNameDataDic)
                    else:
                        self.rawDataD2P(rawDataBuffer,chNameDataDic)

                fPt.close()
                return chNameDataDic
        except:
            print('Failed to read data from %s'%self.filePath)

class FileNEF:
    def __init__(self,fPath):
        self.filePath = fPath
        self.parse()

    def parse(self):
        try:
            with open(self.filePath,'rb') as fPt:
                hdLen = 64
                rawAr = fPt.read(hdLen)
                isLEAr = struct.unpack('BB',rawAr[0:2])
                if isLEAr[0] == 0xfe and isLEAr[1] == 0xff:
                    isLE = True
                elif isLEAr[1] == 0xfe and isLEAr[0] == 0xff:
                    isLE = False
                else:
                    print('Failed to parse %s'%self.filePath)
                    return

                recordDate = rawAr[9:19]
                recordTime = rawAr[19:27]
                if isLE:
                    singleDuration=struct.unpack('<H',rawAr[27:29])[0]
                    eCount=struct.unpack('<I',rawAr[29:33])[0]
                    eDelCount=struct.unpack('<I',rawAr[33:37])[0]
                else:
                    singleDuration=struct.unpack('>H',rawAr[27:29])[0]
                    eCount=struct.unpack('>I',rawAr[29:33])[0]
                    eDelCount=struct.unpack('>I',rawAr[33:37])[0]

                subID = rawAr[37:53]
                fPt.seek(hdLen,0)
                eFrmLen = 1004
                dataLen = eCount*eFrmLen
                rawData = fPt.read(dataLen)
                eventList = []
                for i in range(eCount):
                    lPos = i*(eFrmLen+1)
                    validMark = bool(rawData[lPos])
                    if not validMark:
                        continue

                    uintOrder = '<I'
                    if(not isLE):
                        uintOrder='>I'
                    sPos = lPos+1
                    eID = struct.unpack(uintOrder,rawData[sPos:sPos+4])[0]
                    sPos+=4
                    eOnset = struct.unpack(uintOrder,rawData[sPos:sPos+4])[0]
                    sPos+=6
                    eDuration = struct.unpack(uintOrder,rawData[sPos:sPos+4])[0]
                    sPos+=6
                    eColor = struct.unpack(uintOrder,rawData[sPos:sPos+4])[0]
                    sPos+=6

                    #tmpCh = [chr(each) for each in rawData[sPos:eFrmLen]]
                    tmpCh = []
                    for cNum in rawData[sPos:sPos+982]:
                        c = chr(cNum)
                        if c == '\x16':
                            break
                        tmpCh.append(c)

                    strTmp = ''.join(tmpCh)
                    strSpList = strTmp.split('\x00')
                    strValid = [each for each in strSpList if len(each)>0]
                    eAnno = ''
                    if(len(strValid)>0):
                        eAnno = strValid[0]
                    eRemark = ''
                    if(len(strValid)>1):
                         eRemark = strValid[1]

                    evt = Event()
                    evt.EventIndex = eID
                    evt.Onset = eOnset
                    evt.Duration = eDuration
                    evt.Color = eColor
                    evt.Annotation = eAnno
                    evt.Remark = eRemark
                    eventList.append(evt)

                self.IsLE = isLE
                self.RecordDate = recordDate
                self.RecordTime = recordTime
                self.EventCount = eCount
                self.EventDelCount = eDelCount
                self.EventValidCount = eCount-eDelCount
                self.SingleDuration=singleDuration
                self.Events = eventList

                fPt.close()
        except:
            print("Failed to open file %s"%self.filePath)

class FolderNHF:
    def __init__(self,flPath):
        self.folderPath=flPath
        self.parseHeader()

    def parseHeader(self):
        flCat = NDFUtility.GetFolderCategory(self.folderPath)
        if flCat!= FolderCatergory.NDFFileTypeNHF:
            print("Failed to parse folder %s. "%self.folderPath)
            print("Current folder is not *.NHF or *.NSF folder")
            return
        folder = self.folderPath
        try:
            fAr = []
            fAr += [each for each in os.listdir(folder) if each.endswith('.nhf')]
            fAr += [each for each in os.listdir(folder) if each.endswith('.NHF')]
            fArLen = len(fAr)
            if fArLen<1:
                print("Failed to parse folder %s"%self.folderPath)
                return
            fNHF = fAr[0]
            nhfPath = os.path.join(folder, fNHF)
            nhfObj = FileNHF(nhfPath)
            hCount = nhfObj.HostCount
            hosts = nhfObj.Hosts
            channelNames = []
            channelTypes = []
            chName2NSF = {}
            chName2ChObj = {}
            chName2Host = {}
            sr = []
            dur = []
            segStart = []
            segStop = []
            flNSFObjs = []
            segs = []
            segFolder = []
            isHostFlCN = False
            prStrAr = []
            channelNamesASCII = []
            for j in range(hCount):
                hostObj0 = hosts[j]
                hostFl0 = hostObj0.folder
                hostFlCN = [num0 for num0 in hostFl0 if num0>=128]
                if len(hostFlCN)>0:
                    isHostFlCN = True
                prStr = "Probe%d"%j
                prStrAr.append(prStr)

            for i in range(hCount):
                hostObj = hosts[i]
                hostFl = hostObj.folder

                flName = NDFUtility.Decode(hostFl)
                fdNSF = os.path.join(folder, flName)
                fdNSFObj = FolderNSF(fdNSF)
                flNSFObjs.append(fdNSFObj)
                chs = fdNSFObj.ChannelsDic
                sr.append(fdNSFObj.SR)
                dur.append(fdNSFObj.Duration)
                segStart.append(fdNSFObj.Start)
                segStop.append(fdNSFObj.Stop)
                segCnt = fdNSFObj.SegCount
                if segCnt>1:# segments need to be more than 1 for mne event
                    segs.extend(fdNSFObj.SegmentObj)
                    segsFl = [flName]*segCnt
                    segFolder.extend(segsFl)

                #segs.extend(fdNSFObj.SegmentObj)
                #segsFl = [flName]*fdNSFObj.SegCount
                #segFolder.extend(segsFl)
                for key,value in chs.items():
                    keyNew = key
                    keyNewASCII = key
                    if key in chName2ChObj:
                        keyNew = key+'_'+flName
                        keyNewASCII = key+'_'+prStrAr[i]
                        #keyNewASCII = prStrAr[i]+'_'+flName

                    if isHostFlCN:
                        channelNames.append(keyNewASCII)
                        chName2NSF[keyNewASCII]=fdNSFObj
                        chName2ChObj[keyNewASCII]=value
                        chName2Host[keyNewASCII]=hostObj
                    else:
                        channelNames.append(keyNew)
                        chName2NSF[keyNew]=fdNSFObj
                        chName2ChObj[keyNew]=value
                        chName2Host[keyNew]=hostObj

                    #channelNames.append(keyNew)
                    channelNamesASCII.append(keyNewASCII)
                    # chName2NSF[keyNew]=fdNSFObj
                    # chName2ChObj[keyNew]=value
                    # chName2Host[keyNew]=hostObj
                    channelTypes.append(value.ChannelType)
            self.SegStart = segStart
            self.SegStop = segStop
            self.Start = min(segStart)
            self.Stop =  max(segStop)
            self.SegSR = sr
            self.Duration = self.Stop - self.Start
            self.SR = max(sr)
            self.ChannelNames = channelNames
            self.IsHostFlCN = isHostFlCN
            self.ChannelNamesASCII = channelNamesASCII
            self.ChannelTypes = channelTypes
            self.ChannelCount = len(self.ChannelNames)
            self.ChName2NSF  = chName2NSF
            self.ChName2ChObj = chName2ChObj
            self.ChannelsDic = chName2ChObj
            self.FlNSFObjs = flNSFObjs
            self.HostCount = hCount
            self.Hosts = hosts
            self.ChName2Host = chName2Host
            self.SegmentObj = segs
            self.SegmentFolder = segFolder
            self.RecordDate = nhfObj.RecordDate
            self.RecordTime = nhfObj.RecordTime
        except:
            print("Failed to open folder %s"%folder)

    def readAll(self):
        channelData = {}
        SampleRate = []
        chTypes = []
        chCnt = self.ChannelCount
        for i in range(chCnt):
            chName = self.ChannelNames[i]
            #chDateLen = chObj.SampleRate * duration
            chObj = self.ChName2ChObj[chName]
            sz1D = chObj.SampleRate
            sz2D = self.Duration
            channelData[chName] = [[0]*sz1D for s in range(sz2D)]
        for obj in self.FlNSFObjs:
            obj.readAll()

        for ch in self.ChannelNames:
            nsfObj = self.ChName2NSF[ch]
            dataAll = nsfObj.ChannelData
            if not dataAll:
                dataAll = nsfObj.readAll()
            chObj = self.ChName2ChObj[ch]
            chNameNSFLevel = chObj.ChannelName
            fdNSFObjTmp = self.ChName2NSF[ch]
            #channelData[ch] = dataAll[chNameNSFLevel]
            chData = channelData[ch]
            startTmp = fdNSFObjTmp.Start-self.Start
            stopTmp = fdNSFObjTmp.Stop-self.Start
            chData[startTmp:stopTmp]=dataAll[chNameNSFLevel]
            delRefer = dataAll[chNameNSFLevel]
            del delRefer[:]
            SampleRate.append(chObj.SampleRate)
            chTypes.append(chObj.ChannelType)

        self.ChannelTypes = chTypes
        self.SRAr = SampleRate
        self.ChannelDataDic = channelData
        return channelData

    def readTimeRange(self,start,stop):
        #duration = self.Duration
        durCal = stop - start
        duration = durCal

        channelData = {}
        SampleRate = []
        chTypes = []
        chCnt = self.ChannelCount
        for i in range(chCnt):
            chName = self.ChannelNames[i]

            #chDateLen = chObj.SampleRate * duration
            chObj = self.ChName2ChObj[chName]
            sz1D = chObj.SampleRate
            sz2D = duration
            channelData[chName] = [[0]*sz1D for s in range(sz2D)]

        for obj in self.FlNSFObjs:
            obj.readTimeRange(start,stop)

        for ch in self.ChannelNames:
            nsfObj = self.ChName2NSF[ch]
            if not nsfObj.ChannelData:
                nsfObj.readTimeRange(start,stop)
            dataAll = nsfObj.ChannelData
            chObj = self.chName2ChObj[ch]
            chNameNSFLevel = chObj.ChannelName
            fdNSFObjTmp = self.ChName2NSF[ch]
            #channelData[ch] = dataAll[chNameNSFLevel]
            chData = channelData[ch]
            startTmp = fdNSFObjTmp.Start-self.Start
            stopTmp = fdNSFObjTmp.Stop-self.Start
            chData[startTmp:stopTmp]=dataAll[chNameNSFLevel]
            delRefer = dataAll[chNameNSFLevel]
            del delRefer[:]
            SampleRate.append(chObj.SampleRate)
            chTypes.append(chObj.ChannelType)

        self.ChannelTypes = chTypes
        self.SRAr = SampleRate
        self.ChannelDataDic = channelData
        return channelData

    def readOneChannelTimeRange(self,oneChName,start,stop):
        #duration = self.Duration
        durCal = stop - start
        duration = durCal

        channelData = {}
        SampleRate = []
        chTypes = []
        chCnt = self.ChannelCount
        for i in range(chCnt):
            chName = self.ChannelNames[i]
            #chDateLen = chObj.SampleRate * duration
            chObj = self.ChName2ChObj[chName]
            sz1D = chObj.SampleRate
            sz2D = duration
            channelData[chName] = [[0]*sz1D for s in range(sz2D)]

        for obj in self.FlNSFObjs:
            obj.readTimeRange(start,stop)

        for ch in self.ChannelNames:
            if ch!=oneChName:
                continue
            nsfObj = self.ChName2NSF[ch]
            if not nsfObj.ChannelData:
                nsfObj.readTimeRange(start,stop)
            dataAll = nsfObj.ChannelData
            chObj = self.ChName2ChObj[ch]
            chNameNSFLevel = chObj.ChannelName
            fdNSFObjTmp = self.ChName2NSF[ch]
            #channelData[ch] = dataAll[chNameNSFLevel]
            chData = channelData[ch]
            startTmp = fdNSFObjTmp.Start-self.Start
            stopTmp = fdNSFObjTmp.Stop-self.Start
            chData[startTmp:stopTmp]=dataAll[chNameNSFLevel]
            delRefer = dataAll[chNameNSFLevel]
            del delRefer[:]
            SampleRate.append(chObj.SampleRate)
            chTypes.append(chObj.ChannelType)

        return channelData[oneChName]

class FolderNSF:
    def __init__(self,flPath):
        self.folderPath=flPath
        self.parseHeader()
        self.ChannelData = []

    def parseHeader(self):
        flCat = NDFUtility.GetFolderCategory(self.folderPath)
        if flCat!= FolderCatergory.NDFFileTypeNSF:
            print("Failed to parse folder %s"%self.folderPath)
            return
        folder = self.folderPath
        try:
            fAr = []
            fAr += [each for each in os.listdir(folder) if each.endswith('.nsf')]
            fAr += [each for each in os.listdir(folder) if each.endswith('.NSF')]
            fArLen = len(fAr)
            if fArLen<1:
                print("Failed to parse folder %s"%self.folderPath)
                return
            fNSF = fAr[0]
            nsfPath = os.path.join(folder, fNSF)
            nsfObj = FileNSF(nsfPath)
            segCount = nsfObj.SegCount
            segs = nsfObj.Segments
            segNames = []
            for i in range(segCount):
                segObj = segs[i]
                hostFl = segObj.FolderName
                flName = NDFUtility.Decode(hostFl)
                segNames.append(flName)
                #fdNSF = os.path.join(folder, flName)
            chs = nsfObj.Channels
            chCount = nsfObj.ChannelCount
            chNames = []
            chTypes = []
            chSRs = []
            for i in range(chCount):
                chObj = chs[i]
                chNames.append(chObj.ChannelName)
                chTypes.append(chObj.ChannelType)
                chSRs.append(chObj.SampleRate)
            dur = nsfObj.RecordDuration
            durCal = nsfObj.SegStop - nsfObj.SegStart
            if durCal>dur:
                dur = durCal
            self.Duration = dur
            self.Start = nsfObj.SegStart
            self.Stop = nsfObj.SegStop
            self.PatientName = NDFUtility.Decode(nsfObj.Name)
            self.ChannelNames = chNames
            self.ChannelTypes = chTypes
            self.ChannelSRs = chSRs
            self.SR = max(chSRs)
            self.ChannelCount = nsfObj.ChannelCount
            self.Channels = nsfObj.Channels
            self.ChannelsDic = nsfObj.ChannelsDic
            self.SegCount = nsfObj.SegCount
            self.Segs = nsfObj.Segments
            self.SegmentObj = nsfObj.Segments
            self.NSFObj = nsfObj
            self.SegmentFolder = segNames
            self.RecordDate = nsfObj.RecordDate
            self.ReocrdTime = nsfObj.RecordTime
        except:
            print("Failed to open folder %s"%folder)

    def readChannelData(self,chNames =[],start=[],stop=[]):
        isAllChannel = True
        isAllTime = True
        if not chNames:
            isAllChannel = True
        if not start or not stop:
            isAllTime = False


    def readAll(self):
        duration = self.Duration
        channelData  = {}
        for i in range(self.ChannelCount):
            chObj = self.Channels[i]
            chName = chObj.ChannelName
            #chDateLen = chObj.SampleRate * duration
            sz1D = chObj.SampleRate
            sz2D = duration
            channelData[chName] = [[0]*sz1D for s in range(sz2D)]

        for i in range(self.SegCount):
            segObj = self.Segs[i]
            hostFl = segObj.FolderName
            flName = NDFUtility.Decode(hostFl)
            fNDFPath = os.path.join(self.folderPath, flName)
            segStart = segObj.Start
            segStop = segObj.Stop
            ndfObj = FileNDF(fNDFPath)
            chNameDataDic = ndfObj.readAll()
            for key, value in chNameDataDic.items():
                dItem = channelData[key]
                sr = self.ChannelsDic[key].SampleRate
                #pStart = segStart*sr
                #pStop = segStop*sr
                pStart = segStart - self.Start
                pStop = segStop - self.Start
                dItem[pStart:pStop] = value
                del value[:]
        self.ChannelData = channelData
        return channelData

    def readTimeRange(self,start,stop):
        #duration = self.Duration
        durCal = stop - start
        #if duration>durCal:
        #    duration = durCal
        duration = durCal
        channelData  = {}
        for i in range(self.ChannelCount):
            chObj = self.Channels[i]
            chName = chObj.ChannelName
            #chDateLen = chObj.SampleRate * duration
            sz1D = chObj.SampleRate
            sz2D = duration
            channelData[chName] = [[0]*sz1D for s in range(sz2D)]

        for i in range(self.SegCount):
            segObj = self.Segs[i]
            hostFl = segObj.FolderName
            flName = NDFUtility.Decode(hostFl)
            fNDFPath = os.path.join(self.folderPath, flName)
            segStart = segObj.Start
            segStop = segObj.Stop

            if start<segStart and segStop<stop:
                ndfObj = FileNDF(fNDFPath)
                chNameDataDic = ndfObj.readAll()
                for key, value in chNameDataDic.items():
                    dItem = channelData[key]
                    sr = self.ChannelsDic[key].SampleRate
                    #pStart = segStart*sr
                    #pStop = segStop*sr
                    #dItem[pStart:pStop] = value
                    pStart = segStart
                    pStop = segStop
                    dItem[:,pStart:pStop] = value
                    del value[:]
            else:
                if start >= segStop or stop<=segStart:
                    continue
                else:
                    startValid = start if start>segStart else segStart
                    stopValid = stop if stop<segStop else segStop
                ndfObj = FileNDF(fNDFPath)
                durValid = stopValid - startValid
                if(durValid<1):
                    print('Start time should be greater then stop')
                    continue
                chNameDataDic = ndfObj.readTimeRange(startValid-segStart,durValid)
                for key, value in chNameDataDic.items():
                    dItem = channelData[key]
                    #sr = self.ChannelsDic[key].SampleRate
                    #pStart = (startValid-start)*sr
                    #pStop = (stopValid-start)*sr
                    #dItem[pStart:pStop] = value
                    pStart = (startValid-start)
                    pStop = (stopValid-start)
                    dItem[pStart:pStop] = value
                    del value[:]

        self.ChannelData = channelData
        return channelData

class FolderNTP:
    def __init__(self,flPath):
        self.folderPath=flPath
        self.Events = []

    def parse(self):
        flCat = NDFUtility.GetFolderCategory(self.folderPath)
        if flCat!= FolderCatergory.NDFFileTypeNTP:
            print("Failed to parse folder %s"%self.folderPath)
            return
        folder = self.folderPath
        try:
            fAr = []
            fAr += [each for each in os.listdir(folder) if each.endswith('.ntp')]
            fAr += [each for each in os.listdir(folder) if each.endswith('.Ntp')]
            fArLen = len(fAr)
            if fArLen<1:
                print("Failed to parse folder %s"%self.folderPath)
                return
            fNTP = fAr[0]
            nhfPath = os.path.join(folder, fNTP)
            ntpObj = FileNTP(nhfPath)
            subCount = ntpObj.SubCount
            subs = ntpObj.subInfoAr
            self.NTPObj = ntpObj
            self.SubCount = subCount
            self.SubInfo = subs
            fAr = []
            fAr += [each for each in os.listdir(folder) if each.endswith('.nef')]
            fAr += [each for each in os.listdir(folder) if each.endswith('.NEF')]
            fArLen = len(fAr)
            if fArLen>0:
                fNEF = fAr[0]
                nefPath = os.path.join(folder, fNEF)
                nefObj = FileNEF(nefPath)
                self.Events = nefObj.Events
        except:
            print("Failed to open folder %s"%folder)

    def readAll(self,subIndex = 0):
        if self.SubCount>subIndex:
            subObj = self.SubInfo[subIndex]
            hostFl = subObj.folderName
            flName = NDFUtility.Decode(hostFl)
            fdNHF = os.path.join(self.folderPath, flName)
            fdNSFObj = FolderNHF(fdNHF)
            data = fdNSFObj.readAll()
            self.FlNHFObj = fdNSFObj

            self.ChannelCount = fdNSFObj.ChannelCount
            self.ChannelTypes = fdNSFObj.ChannelTypes
            self.ChannelNames = fdNSFObj.ChannelNames
            return data

