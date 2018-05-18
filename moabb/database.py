from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import Column, Integer, Text, ForeignKey, Float

Base = declarative_base()
engine = create_engine('sqlite:///results.db', echo=True)
sess_factory = sessionmaker(bind=engine)
session = sess_factory()

def add_dataset(d):
    '''Function that takes a dataset, checks if it is in the database, and if not
    adds it to the database file.

    '''
    matches = session.query(DatasetEntry).filter_by(classname=type(d).__name__).\
              count()
    if matches == 0:
        session.add(DatasetEntry(d))
        session.commit()

class DatasetEntry(Base):

    __tablename__ = 'datasets'
    classname = Column(Text, primary_key=True)
    name = Column(Text)
    paradigm = Column(Text)
    nsubjects = Column(Integer)
    nsessions = Column(Integer)
    t0 = Column(Float)
    tend = Column(Float)
    doi = Column(Text)
    events = relationship("Event",backref='dataset')

    def __init__(self, dset):
        self.name = dset.code
        self.classname = type(dset).__name__
        self.paradigm = dset.paradigm
        self.doi = dset.doi
        self.events = [Event(name=k) for k in dset.event_id.keys()]
        self.nsessions = dset.n_sessions
        self.nsubjects = len(dset.subject_list)
        self.t0 = dset.interval[0]
        self.tend = dset.interval[1]
        self.interval = self.tend-self.t0

    def __repr__(self):
        return 'DatasetEntry(name={}, paradigm={}, subjects={})'.format(self.name, self.paradigm, self.nsubjects)


class Event(Base):

    __tablename__ = 'events'
    name = Column(Text)
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.classname'))

    def __repr__(self):
        return 'Event(name={}, dset={})'.format(self.name, self.dataset_id)

Base.metadata.create_all(engine)

if __name__ == '__main__':
    from moabb.datasets.utils import dataset_search

    print(dataset_search('imagery'))
