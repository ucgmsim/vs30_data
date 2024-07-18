"""
Functions developed by Sung Bae to load CPT data from a SQL database.
"""

import numpy as np
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CPTLocation(Base):
    __tablename__ = "cpt_location"
    id = Column(Integer, primary_key=True)
    #    customer_id=Column(Integer, ForeignKey('customers.id'))
    name = Column(String(20), nullable=False)  # 20210427_17
    private = Column(Integer)  # true / false
    type = Column(String(5))  # CPT or SCPT
    nztm_x = Column(Float)
    nztm_y = Column(Float)

    def __iter__(self):  # overridding this to return tuples of (key,value)
        return iter(
            [
                ("id", self.id),
                ("name", self.name),
                ("nztm_x", self.nztm_x),
                ("nztm_y", self.nztm_y),
            ]
        )


class CPTDepthRecord(Base):
    __tablename__ = "cpt_depth_record"
    id = Column(Integer, primary_key=True)
    cpt_name = Column(String(20), nullable=False)  #
    depth = Column(Float)  #
    qc = Column(Float)  #
    fs = Column(Float)
    u = Column(Float)
    loc_id = Column(Integer, ForeignKey("cpt_location.id"))

    def __iter__(self):  # overridding this to return tuples of (key,value)
        return iter(
            [
                ("id", self.id),
                ("depth", self.depth),
                ("qc", self.qc),
                ("fs", self.fs),
                ("u", self.u),
                ("loc_id", self.loc_id),
            ]
        )


# not really useful, but presented as an example
def cpt_records(session, cpt_name):
    res = (
        session.query(CPTDepthRecord).filter(CPTDepthRecord.cpt_name == cpt_name).all()
    )
    return res


# not really useful, but presented as an example
def max_depth_record(session, cpt_name):
    res = (
        session.query(CPTDepthRecord)
        .filter(CPTDepthRecord.cpt_name == cpt_name)
        .order_by(CPTDepthRecord.depth.desc())
        .first()
    )
    return res


# the following 3 functions are actually used
def cpt_locations(session):
    return session.query(CPTLocation).all()


def cpt_records_exists(session, cpt_name):
    res = (
        session.query(CPTDepthRecord)
        .filter(CPTDepthRecord.cpt_name == cpt_name)
        .first()
    )
    return res is not None


def get_cpt_data(session, cpt_name, columnwise=True):
    res = (
        session.query(
            CPTDepthRecord.depth, CPTDepthRecord.qc, CPTDepthRecord.fs, CPTDepthRecord.u
        )
        .filter(CPTDepthRecord.cpt_name == cpt_name)
        .all()
    )
    res_array = np.array(res)
    if columnwise:  # each column is grouped together
        res_array = res_array.T
    return res_array
