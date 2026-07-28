import mne
import mne.io.pick as pk
import numpy as np
import scipy.signal as sgn
from datetime import datetime,timezone
from .NDFSysParser import FolderNHF,FolderNSF,FileNEF
from .NDFSysUtility import FolderCatergory, NDFUtility




class mneNDF:
    def __init__(self,folder,nefPath=''):
        self.DataFolder = folder
        self.NEFPath = nefPath
        self.parse()

    def parse(self):
        dataFolder = self.DataFolder
        flCat = NDFUtility.GetFolderCategory(dataFolder)
        events = []
        if flCat == FolderCatergory.NDFFileTypeNHF:
            self.FolderType = FolderCatergory.NDFFileTypeNHF
            nhfObj = FolderNHF(dataFolder)
            flObj = nhfObj
        elif flCat == FolderCatergory.NDFFileTypeNSF:
            self.FolderType = FolderCatergory.NDFFileTypeNSF
            flNsfObj = FolderNSF(dataFolder)
            flObj = flNsfObj
        elif flCat == FolderCatergory.NDFFileTypeNTP:
            print('Please select a *.NHF or *.NSF folder')
            return
        else:
            print('Please select a *.NHF or *.NSF folder')
            return
        nefPath = self.NEFPath
        self.Events = []
        self.EventCount = 0
        if nefPath!='':
            nefObj = FileNEF(nefPath)
            self.Events  = nefObj.Events
            self.EventCount = len(self.Events)

        self.FolderObj = flObj
        self.ChannelCount = flObj.ChannelCount
        self.ChannelTypes = flObj.ChannelTypes
        if flObj.IsHostFlCN:
            self.ChannelNames = flObj.ChannelNamesASCII
        else:
            self.ChannelNames = flObj.ChannelNames
        #self.ChannelNames = flObj.ChannelNames
        self.Start = flObj.Start
        self.Stop = flObj.Stop
        self.Segments = flObj.SegmentObj
        self.SegmentFl = flObj.SegmentFolder
        self.RecordDate = flObj.RecordDate
        self.RecordTime = flObj.RecordTime

    def read2MneRaw(self,start = [], stop =[]):
        flObj = self.FolderObj
        n_channels = self.ChannelCount
        ch_names = self.ChannelNames
        ch_types = self.ChannelTypes
        ch_types_lower = [each.lower() for each in ch_types]
        ch_types_mne = []
        for t in ch_types_lower:
            if t !='eeg':
                t = 'misc'
            ch_types_mne.append(t)
        #ch_types_mne = ['misc'] * n_channels

        sfreq = flObj.SR



        chsDic = flObj.ChannelsDic
        if start and stop:
            chDataDic = flObj.readTimeRange(start,stop)
            dur = stop - start
        else:
            chDataDic = flObj.readAll()
            dur = flObj.Duration



        dataRet = np.zeros([n_channels,dur*sfreq])
        ch_names_format = []
        for i in range(n_channels):
            chName = ch_names[i]
            chNameFmt = chName
            chNameLen = len(chName)#limit the length of channel name to 16 by bdf spec
            if chNameLen > 16:
                chNameFmt = chName[:16]
            ch_names_format.append(chNameFmt)
            chObj = chsDic[chName]
            chData = chDataDic[chName]
            chDataNP = np.array(chData)
            del chData[:]
            chDataNP1D = chDataNP.flatten()
            chDataNP1DValid = chDataNP1D
            chSr = chObj.SampleRate
            if chSr != sfreq :
                xSrcLen = dur*chSr
                xSrc = np.arange(1,xSrcLen+1)
                xSrcT = xSrc/chSr
                xTargLen = dur*sfreq
                xTarg = np.arange(1,xTargLen+1)
                xTargT = xTarg/sfreq
                #mne.filter.resample()
                #chDataNP1DValid = mne.filter.resample(
                #           chDataNP1D.astype(np.float64), xTargLen, xSrcLen,
                #            npad=0, axis=-1)
                #chDataNP1DValid = sgn.resample(chDataNP1D.astype(np.float64),xTargLen)
                chDataNP1DValid = np.interp(xTargT,xSrcT,chDataNP1D)

            dataRet[i,:] = chDataNP1DValid

        info = mne.create_info(ch_names=ch_names,ch_types=ch_types_mne,sfreq=sfreq)
        simulated_raw = mne.io.RawArray(dataRet, info)

        encoding = 'utf-8'
        dateStr = str(self.RecordDate, encoding)
        dateStr = dateStr.replace('.','-')
        timetr = str(self.RecordTime, encoding)
        dateTimeStr = '%s %s'%(dateStr, timetr)
        dt = datetime.fromisoformat(dateTimeStr)
        dt = dt.replace(tzinfo=timezone.utc)
        simulated_raw.set_meas_date(dt)

        cnt = self.EventCount
        evts = self.Events
        onset = []
        desc = []
        dur = []
        for i in range(cnt):
            ev = evts[i]
            onset.append(ev.Onset*1.0e-3)
            desc.append(ev.Annotation)
            dur.append(ev.Duration)

        segs = self.Segments
        segCount = len(self.Segments)
        if segCount>1:
            for i in range(segCount):
                sg = segs[i]
                onset.append(sg.Start)
                fl = self.SegmentFl[i]
                ano = '%s record segment %d'%(fl,sg.Index)
                desc.append(ano)
                dur.append(sg.Duration)

        annot = mne.Annotations(onset= onset,duration = dur,description = desc)
        simulated_raw.set_annotations(annot)
        return simulated_raw

